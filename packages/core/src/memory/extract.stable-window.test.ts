/**
 * @license
 * Copyright 2026 Qwen Team
 * SPDX-License-Identifier: Apache-2.0
 */

import * as fs from 'node:fs/promises';
import * as os from 'node:os';
import * as path from 'node:path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Config } from '../config/config.js';
import type { Content } from '@google/genai';
import { getAutoMemoryExtractCursorPath } from './paths.js';
import { runAutoMemoryExtract } from './extract.js';
import { runAutoMemoryExtractionByAgent } from './extractionAgentPlanner.js';
import { ensureAutoMemoryScaffold } from './store.js';
import { getCacheSafeParamsSessionId } from '../agents/forkedAgent.js';
import {
  EXTRACT_STABLE_WINDOW_ENV,
  getExtractWindowProcessId,
} from './extract-window.js';
import type { AutoMemoryExtractCursor } from './types.js';

vi.mock('./extractionAgentPlanner.js', () => ({
  runAutoMemoryExtractionByAgent: vi.fn(),
}));

vi.mock('../agents/forkedAgent.js', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../agents/forkedAgent.js')>()),
  getCacheSafeParamsSessionId: vi.fn(),
}));

vi.mock('./indexer.js', () => ({
  rebuildManagedAutoMemoryIndex: vi.fn().mockResolvedValue(''),
  rebuildUserAutoMemoryIndex: vi.fn().mockResolvedValue(''),
}));

vi.mock('./refresh.js', () => ({
  refreshMemoryInstruction: vi.fn().mockResolvedValue(undefined),
}));

const SESSION_ID = 'session-1';

function buildHistory(count: number, label = 'turn'): Content[] {
  return Array.from({ length: count }, (_, index) => ({
    role: index % 2 === 0 ? ('user' as const) : ('model' as const),
    parts: [{ text: `${label} ${index}` }],
  }));
}

describe('auto-memory extraction stable window', () => {
  let tempDir: string;
  let projectRoot: string;
  let rawHistory: Content[];
  let curated: Content[];
  let getHistory: ReturnType<typeof vi.fn>;
  let mockConfig: Config;
  let previousStableWindowEnv: string | undefined;

  beforeEach(async () => {
    previousStableWindowEnv = process.env[EXTRACT_STABLE_WINDOW_ENV];
    delete process.env[EXTRACT_STABLE_WINDOW_ENV];
    tempDir = await fs.mkdtemp(
      path.join(os.tmpdir(), 'auto-memory-stable-window-'),
    );
    projectRoot = path.join(tempDir, 'project');
    await fs.mkdir(projectRoot, { recursive: true });
    await ensureAutoMemoryScaffold(projectRoot);
    rawHistory = buildHistory(100);
    curated = buildHistory(100);
    getHistory = vi.fn(() => curated);
    mockConfig = {
      getSessionId: vi.fn().mockReturnValue(SESSION_ID),
      getLlmClient: vi.fn().mockReturnValue({ getHistory }),
    } as unknown as Config;
    vi.clearAllMocks();
    vi.mocked(getCacheSafeParamsSessionId).mockReturnValue(SESSION_ID);
    vi.mocked(runAutoMemoryExtractionByAgent).mockResolvedValue({
      touchedTopics: [],
      touchedProjectScope: false,
      touchedUserScope: false,
      hasToolActivity: true,
      systemMessage: undefined,
    });
  });

  afterEach(async () => {
    if (previousStableWindowEnv === undefined) {
      delete process.env[EXTRACT_STABLE_WINDOW_ENV];
    } else {
      process.env[EXTRACT_STABLE_WINDOW_ENV] = previousStableWindowEnv;
    }
    await fs.rm(tempDir, {
      recursive: true,
      force: true,
      maxRetries: 3,
      retryDelay: 10,
    });
  });

  async function readCursor(): Promise<AutoMemoryExtractCursor> {
    return JSON.parse(
      await fs.readFile(getAutoMemoryExtractCursorPath(projectRoot), 'utf-8'),
    ) as AutoMemoryExtractCursor;
  }

  function extract() {
    return runAutoMemoryExtract({
      projectRoot,
      sessionId: SESSION_ID,
      config: mockConfig,
      history: rawHistory,
    });
  }

  function lastStableWindow() {
    const call = vi.mocked(runAutoMemoryExtractionByAgent).mock.calls.at(-1);
    return call?.[2]?.stableWindow;
  }

  it('pins a curated window, persists the epoch, and passes the snapshot to the planner', async () => {
    await extract();

    const cursor = await readCursor();
    expect(cursor.epoch).toBe(1);
    expect(cursor.windowStartCurated).toBe(60);
    expect(cursor.coveredCurated).toBe(100);
    expect(cursor.windowHeadHash).toMatch(/^[0-9a-f]{64}$/);
    expect(cursor.windowProcessId).toBe(getExtractWindowProcessId());
    // Raw cursor semantics are untouched.
    expect(cursor.processedOffset).toBe(100);

    const stableWindow = lastStableWindow();
    expect(stableWindow?.curatedHistory).toBe(curated);
    expect(stableWindow?.windowStartCurated).toBe(60);
    expect(getHistory).toHaveBeenCalledWith(true);
  });

  it('keeps the prefix append-only across consecutive runs', async () => {
    await extract();
    const firstRunWindow = lastStableWindow()!;
    const firstWindowSlice = firstRunWindow.curatedHistory.slice(
      firstRunWindow.windowStartCurated,
    );

    // The next turn appends to both the raw and the curated history.
    curated.push(...buildHistory(2, 'appended'));
    rawHistory.push({
      role: 'user',
      parts: [{ text: 'appended user turn' }],
    });
    await extract();

    const secondRunWindow = lastStableWindow()!;
    expect(secondRunWindow.windowStartCurated).toBe(
      firstRunWindow.windowStartCurated,
    );
    const secondWindowSlice = secondRunWindow.curatedHistory.slice(
      secondRunWindow.windowStartCurated,
    );
    expect(secondWindowSlice).toHaveLength(firstWindowSlice.length + 2);
    expect(secondWindowSlice.slice(0, firstWindowSlice.length)).toEqual(
      firstWindowSlice,
    );

    const cursor = await readCursor();
    expect(cursor.epoch).toBe(1);
    expect(cursor.windowStartCurated).toBe(60);
    expect(cursor.coveredCurated).toBe(102);
  });

  it('rebases the epoch when the curated head changes (compaction)', async () => {
    await extract();
    expect((await readCursor()).epoch).toBe(1);

    curated = buildHistory(100, 'compacted');
    rawHistory.push({
      role: 'user',
      parts: [{ text: 'after compaction' }],
    });
    await extract();

    const cursor = await readCursor();
    expect(cursor.epoch).toBe(2);
    expect(cursor.windowStartCurated).toBe(60);
    expect(cursor.coveredCurated).toBe(100);
    expect(lastStableWindow()?.curatedHistory).toBe(curated);
  });

  it('falls back to the legacy 40-entry tail with the flag disabled', async () => {
    process.env[EXTRACT_STABLE_WINDOW_ENV] = '0';

    await extract();

    const cursor = await readCursor();
    expect(cursor.epoch).toBeUndefined();
    expect(cursor.windowStartCurated).toBeUndefined();
    expect(cursor.coveredCurated).toBeUndefined();
    expect(cursor.windowHeadHash).toBeUndefined();
    expect(getHistory).not.toHaveBeenCalled();

    const call = vi.mocked(runAutoMemoryExtractionByAgent).mock.calls.at(-1);
    expect(call).toEqual([mockConfig, projectRoot]);
  });

  it('falls back to the legacy tail when no live curated history is available', async () => {
    (mockConfig as unknown as { getLlmClient?: unknown }).getLlmClient =
      undefined;

    await extract();

    const cursor = await readCursor();
    expect(cursor.epoch).toBeUndefined();
    expect(cursor.windowStartCurated).toBeUndefined();
    const call = vi.mocked(runAutoMemoryExtractionByAgent).mock.calls.at(-1);
    expect(call).toEqual([mockConfig, projectRoot]);
  });

  it('carries the window state through a run that sends no extraction input', async () => {
    await extract();
    const pinned = await readCursor();

    // Only a model turn is appended: no new user message, so the extractor is
    // not invoked and the pinned window must survive the cursor rewrite.
    rawHistory.push({ role: 'model', parts: [{ text: 'assistant only' }] });
    await extract();

    expect(runAutoMemoryExtractionByAgent).toHaveBeenCalledTimes(1);
    const cursor = await readCursor();
    expect(cursor.epoch).toBe(pinned.epoch);
    expect(cursor.windowStartCurated).toBe(pinned.windowStartCurated);
    expect(cursor.coveredCurated).toBe(pinned.coveredCurated);
    expect(cursor.windowHeadHash).toBe(pinned.windowHeadHash);
  });
});
