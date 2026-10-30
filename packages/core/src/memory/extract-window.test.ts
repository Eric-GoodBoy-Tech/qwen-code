/**
 * @license
 * Copyright 2026 Qwen Team
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from 'vitest';
import type { Content } from '@google/genai';
import type { AutoMemoryExtractCursor } from './types.js';
import {
  DEFAULT_EXTRACT_WINDOW_MAX_CHARS,
  EXTRACT_STABLE_WINDOW_ENV,
  EXTRACT_WINDOW_MAX_CHARS_ENV,
  decideExtractWindow,
  estimateExtractWindowChars,
  getExtractWindowHeadHash,
  getExtractWindowProcessId,
  isStableExtractionWindowEnabled,
  resolveExtractWindowMaxChars,
  sliceExtractWindow,
  toExtractWindowCursorFields,
  type ExtractWindowState,
} from './extract-window.js';

function buildHistory(count: number, label = 'turn'): Content[] {
  return Array.from({ length: count }, (_, index) => ({
    role: index % 2 === 0 ? ('user' as const) : ('model' as const),
    parts: [{ text: `${label} ${index}` }],
  }));
}

const SESSION_ID = 'session-1';

/** Simulate persisting a decision the way extract.ts does. */
function persistWindow(
  state: ExtractWindowState,
  overrides: Partial<AutoMemoryExtractCursor> = {},
): AutoMemoryExtractCursor {
  return {
    sessionId: SESSION_ID,
    processedOffset: 0,
    updatedAt: new Date(1).toISOString(),
    ...toExtractWindowCursorFields(state),
    ...overrides,
  };
}

describe('stable extraction window', () => {
  it('defaults to enabled and only explicit falsey values disable it', () => {
    expect(isStableExtractionWindowEnabled({})).toBe(true);
    expect(
      isStableExtractionWindowEnabled({ [EXTRACT_STABLE_WINDOW_ENV]: '' }),
    ).toBe(true);
    for (const value of ['0', 'false', 'off', 'no', 'FALSE', ' Off ']) {
      expect(
        isStableExtractionWindowEnabled({ [EXTRACT_STABLE_WINDOW_ENV]: value }),
      ).toBe(false);
    }
    for (const value of ['1', 'true', 'on', 'yes']) {
      expect(
        isStableExtractionWindowEnabled({ [EXTRACT_STABLE_WINDOW_ENV]: value }),
      ).toBe(true);
    }
  });

  it('resolves the char budget with env override and safe fallbacks', () => {
    expect(resolveExtractWindowMaxChars({})).toBe(
      DEFAULT_EXTRACT_WINDOW_MAX_CHARS,
    );
    expect(
      resolveExtractWindowMaxChars({ [EXTRACT_WINDOW_MAX_CHARS_ENV]: '5000' }),
    ).toBe(5000);
    for (const value of ['nope', '-1', '0', '']) {
      expect(
        resolveExtractWindowMaxChars({ [EXTRACT_WINDOW_MAX_CHARS_ENV]: value }),
      ).toBe(DEFAULT_EXTRACT_WINDOW_MAX_CHARS);
    }
  });

  it('seeds a fresh epoch from the legacy 40-entry tail on a cold start', () => {
    const curated = buildHistory(100);
    const decision = decideExtractWindow({
      cursor: { updatedAt: new Date(0).toISOString() },
      sessionId: SESSION_ID,
      curated,
      maxChars: DEFAULT_EXTRACT_WINDOW_MAX_CHARS,
    });

    expect(decision.rebased).toBe(true);
    expect(decision.reason).toBe('cold_start');
    expect(decision.state.epoch).toBe(1);
    expect(decision.state.windowStartCurated).toBe(100 - 40);
    expect(decision.state.coveredCurated).toBe(100);
    expect(decision.state.windowHeadHash).toBe(
      getExtractWindowHeadHash(curated, 60),
    );
  });

  it('reads a legacy cursor (no window state) without crashing', () => {
    const legacy: AutoMemoryExtractCursor = {
      sessionId: SESSION_ID,
      processedOffset: 42,
      updatedAt: new Date(0).toISOString(),
    };
    const decision = decideExtractWindow({
      cursor: legacy,
      sessionId: SESSION_ID,
      curated: buildHistory(10),
      maxChars: DEFAULT_EXTRACT_WINDOW_MAX_CHARS,
    });

    expect(decision.rebased).toBe(true);
    expect(decision.reason).toBe('cold_start');
    expect(decision.state.epoch).toBe(1);
    expect(decision.state.windowStartCurated).toBe(0);
  });

  it('keeps the pinned start across appended turns (append-only prefix)', () => {
    const firstCurated = buildHistory(100);
    const first = decideExtractWindow({
      cursor: { updatedAt: new Date(0).toISOString() },
      sessionId: SESSION_ID,
      curated: firstCurated,
      maxChars: DEFAULT_EXTRACT_WINDOW_MAX_CHARS,
    });
    const cursor = persistWindow(first.state);

    const secondCurated = [...buildHistory(100), ...buildHistory(12, 'new')];
    const second = decideExtractWindow({
      cursor,
      sessionId: SESSION_ID,
      curated: secondCurated,
      maxChars: DEFAULT_EXTRACT_WINDOW_MAX_CHARS,
    });

    expect(second.rebased).toBe(false);
    expect(second.state.epoch).toBe(first.state.epoch);
    expect(second.state.windowStartCurated).toBe(
      first.state.windowStartCurated,
    );
    // The new run's inherited history strictly extends the old one.
    const firstWindow = sliceExtractWindow(
      firstCurated,
      first.state.windowStartCurated,
    );
    const secondWindow = sliceExtractWindow(
      secondCurated,
      second.state.windowStartCurated,
    );
    expect(secondWindow).toHaveLength(firstWindow.length + 12);
    expect(secondWindow.slice(0, firstWindow.length)).toEqual(firstWindow);
  });

  it('rebases when the curated head at windowStart changes (compaction)', () => {
    const curated = buildHistory(100);
    const first = decideExtractWindow({
      cursor: { updatedAt: new Date(0).toISOString() },
      sessionId: SESSION_ID,
      curated,
      maxChars: DEFAULT_EXTRACT_WINDOW_MAX_CHARS,
    });
    const cursor = persistWindow(first.state);

    const compacted = buildHistory(100, 'compacted');
    const decision = decideExtractWindow({
      cursor,
      sessionId: SESSION_ID,
      curated: compacted,
      maxChars: DEFAULT_EXTRACT_WINDOW_MAX_CHARS,
    });

    expect(decision.rebased).toBe(true);
    expect(decision.reason).toBe('head_changed');
    expect(decision.state.epoch).toBe(first.state.epoch + 1);
    expect(decision.state.windowStartCurated).toBe(100 - 40);
    expect(decision.state.windowHeadHash).toBe(
      getExtractWindowHeadHash(compacted, 60),
    );
  });

  it('rebases on a session change', () => {
    const curated = buildHistory(100);
    const first = decideExtractWindow({
      cursor: { updatedAt: new Date(0).toISOString() },
      sessionId: SESSION_ID,
      curated,
      maxChars: DEFAULT_EXTRACT_WINDOW_MAX_CHARS,
    });
    const decision = decideExtractWindow({
      cursor: persistWindow(first.state),
      sessionId: 'session-2',
      curated: buildHistory(100),
      maxChars: DEFAULT_EXTRACT_WINDOW_MAX_CHARS,
    });

    expect(decision.rebased).toBe(true);
    expect(decision.reason).toBe('session_changed');
    expect(decision.state.epoch).toBe(first.state.epoch + 1);
  });

  it('rebases after a process restart', () => {
    const curated = buildHistory(100);
    const first = decideExtractWindow({
      cursor: { updatedAt: new Date(0).toISOString() },
      sessionId: SESSION_ID,
      curated,
      maxChars: DEFAULT_EXTRACT_WINDOW_MAX_CHARS,
      processId: 'process-before-restart',
    });
    const cursor = persistWindow(first.state);
    // The persisted cursor claims a different process instance than ours.
    const decision = decideExtractWindow({
      cursor: { ...cursor, windowProcessId: 'process-before-restart' },
      sessionId: SESSION_ID,
      curated: buildHistory(104),
      maxChars: DEFAULT_EXTRACT_WINDOW_MAX_CHARS,
      processId: 'process-after-restart',
    });

    expect(decision.rebased).toBe(true);
    expect(decision.reason).toBe('process_restarted');
    expect(decision.state.epoch).toBe(first.state.epoch + 1);
  });

  it('rebases when history shrinks below the pinned start', () => {
    const first = decideExtractWindow({
      cursor: { updatedAt: new Date(0).toISOString() },
      sessionId: SESSION_ID,
      curated: buildHistory(100),
      maxChars: DEFAULT_EXTRACT_WINDOW_MAX_CHARS,
    });
    const decision = decideExtractWindow({
      cursor: persistWindow(first.state),
      sessionId: SESSION_ID,
      curated: buildHistory(30),
      maxChars: DEFAULT_EXTRACT_WINDOW_MAX_CHARS,
    });

    expect(decision.rebased).toBe(true);
    expect(decision.reason).toBe('history_shrank');
    expect(decision.state.windowStartCurated).toBe(0);
    expect(decision.state.coveredCurated).toBe(30);
  });

  it('rebases on budget overflow without dropping the uncovered tail', () => {
    // Epoch pinned at curated[0] and covered only the first 100 entries.
    const firstCurated = buildHistory(100);
    const first = decideExtractWindow({
      cursor: { updatedAt: new Date(0).toISOString() },
      sessionId: SESSION_ID,
      curated: firstCurated,
      maxChars: DEFAULT_EXTRACT_WINDOW_MAX_CHARS,
    });
    expect(first.state.windowStartCurated).toBe(60);

    // A pinned window from entry 0 that covered 100 entries, now 200 long and
    // over budget: the rebase must keep entries 100..199 (never covered).
    const longCurated = buildHistory(200);
    const pinnedAtZero: ExtractWindowState = {
      epoch: 3,
      windowStartCurated: 0,
      coveredCurated: 100,
      windowHeadHash: getExtractWindowHeadHash(longCurated, 0),
    };
    const decision = decideExtractWindow({
      cursor: persistWindow(pinnedAtZero),
      sessionId: SESSION_ID,
      curated: longCurated,
      maxChars: 1,
    });

    expect(decision.rebased).toBe(true);
    expect(decision.reason).toBe('budget_overflow');
    expect(decision.state.epoch).toBe(4);
    // The budget-fitting seed would start after the last covered entry, so the
    // no-loss clamp starts at 100 instead.
    expect(decision.state.windowStartCurated).toBe(100);
    const window = sliceExtractWindow(
      longCurated,
      decision.state.windowStartCurated,
    );
    expect(window[0]).toEqual(longCurated[100]);
    expect(window[window.length - 1]).toEqual(longCurated[199]);
  });

  it('applies the no-loss clamp to head_changed rebases too', () => {
    // A pinned window from entry 0 covered 100 entries; the head at 0 changed
    // (compaction/rewrite) so the epoch must rebase, but entries 100..199 were
    // never covered and must stay in the new window.
    const curated = buildHistory(200);
    const pinnedAtZero: ExtractWindowState = {
      epoch: 2,
      windowStartCurated: 0,
      coveredCurated: 100,
      windowHeadHash: getExtractWindowHeadHash(curated, 0),
    };
    const rewritten = buildHistory(200, 'rewritten');
    const decision = decideExtractWindow({
      cursor: persistWindow(pinnedAtZero),
      sessionId: SESSION_ID,
      curated: rewritten,
      maxChars: DEFAULT_EXTRACT_WINDOW_MAX_CHARS,
    });

    expect(decision.reason).toBe('head_changed');
    // Budget-fitting seed would be rewritten[160], but entry 100 is the oldest
    // never-covered turn.
    expect(decision.state.windowStartCurated).toBe(100);
  });

  it('seeds a 40-entry window that by itself exceeds the budget with budget headroom', () => {
    // A busy session: 40 entries of ~10k chars each = ~400k chars, well over
    // the 200k budget. A legacy 40-entry seed would rebase on every run.
    const perEntryChars = 10_000;
    const curated: Content[] = Array.from({ length: 40 }, (_, index) => ({
      role: index % 2 === 0 ? ('user' as const) : ('model' as const),
      parts: [{ text: 'x'.repeat(perEntryChars) }],
    }));
    const first = decideExtractWindow({
      cursor: { updatedAt: new Date(0).toISOString() },
      sessionId: SESSION_ID,
      curated,
      maxChars: DEFAULT_EXTRACT_WINDOW_MAX_CHARS,
    });

    expect(first.rebased).toBe(true);
    // Half the 200k budget fits 10 entries.
    expect(first.state.windowStartCurated).toBe(40 - 10);
    expect(
      estimateExtractWindowChars(curated, first.state.windowStartCurated),
    ).toBeLessThanOrEqual(DEFAULT_EXTRACT_WINDOW_MAX_CHARS / 2);

    // The next run only appends turns: no immediate rebase loop.
    const withAppendedTurns = [...curated, ...buildHistory(2, 'appended')];
    const second = decideExtractWindow({
      cursor: persistWindow(first.state),
      sessionId: SESSION_ID,
      curated: withAppendedTurns,
      maxChars: DEFAULT_EXTRACT_WINDOW_MAX_CHARS,
    });

    expect(second.rebased).toBe(false);
    expect(second.state.epoch).toBe(first.state.epoch);
    expect(second.state.windowStartCurated).toBe(
      first.state.windowStartCurated,
    );
  });

  it('keeps an oversized single entry as the seed instead of crashing', () => {
    // One 300k-char entry exceeds even half the budget; the seed must still
    // contain at least that newest entry.
    const curated: Content[] = [
      ...buildHistory(4),
      { role: 'model', parts: [{ text: 'y'.repeat(300_000) }] },
    ];
    const decision = decideExtractWindow({
      cursor: { updatedAt: new Date(0).toISOString() },
      sessionId: SESSION_ID,
      curated,
      maxChars: DEFAULT_EXTRACT_WINDOW_MAX_CHARS,
    });

    expect(decision.rebased).toBe(true);
    expect(decision.state.windowStartCurated).toBe(curated.length - 1);
    expect(
      sliceExtractWindow(curated, decision.state.windowStartCurated),
    ).toHaveLength(1);
  });

  it('handles an empty curated history', () => {
    const decision = decideExtractWindow({
      cursor: { updatedAt: new Date(0).toISOString() },
      sessionId: SESSION_ID,
      curated: [],
      maxChars: DEFAULT_EXTRACT_WINDOW_MAX_CHARS,
    });

    expect(decision.rebased).toBe(true);
    expect(decision.state.epoch).toBe(1);
    expect(decision.state.windowStartCurated).toBe(0);
    expect(decision.state.windowHeadHash).toBe('');
  });

  it('keeps the epoch across runs that reuse the window', () => {
    const curated = buildHistory(100);
    const first = decideExtractWindow({
      cursor: { updatedAt: new Date(0).toISOString() },
      sessionId: SESSION_ID,
      curated,
      maxChars: DEFAULT_EXTRACT_WINDOW_MAX_CHARS,
    });
    const persisted = persistWindow(first.state);
    expect(persisted.windowProcessId).toBe(getExtractWindowProcessId());
    expect(persisted.epoch).toBe(first.state.epoch);
    expect(persisted.windowStartCurated).toBe(first.state.windowStartCurated);
    expect(persisted.coveredCurated).toBe(first.state.coveredCurated);

    const secondCurated = buildHistory(105);
    const second = decideExtractWindow({
      cursor: persisted,
      sessionId: SESSION_ID,
      curated: secondCurated,
      maxChars: DEFAULT_EXTRACT_WINDOW_MAX_CHARS,
    });
    expect(second.rebased).toBe(false);
    expect(second.state.coveredCurated).toBe(105);
  });
});
