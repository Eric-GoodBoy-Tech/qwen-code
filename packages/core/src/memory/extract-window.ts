/**
 * @license
 * Copyright 2026 Qwen Team
 * SPDX-License-Identifier: Apache-2.0
 */

import { createHash, randomUUID } from 'node:crypto';
import type { Content } from '@google/genai';
import type { AutoMemoryExtractCursor } from './types.js';

/**
 * Stable extractor window (mechanism A).
 *
 * The legacy extractor always inherits the last 40 curated entries, so every
 * run shifts the prompt prefix by however many turns were appended and no
 * provider prefix cache survives between runs. Here the start index is pinned
 * for an "epoch": each run inherits curated[windowStartCurated..], which only
 * grows at the tail, keeping the inherited history an append-only prefix.
 *
 * There is deliberately no size cap: the window grows with the session and is
 * bounded only by the main session's own compaction. The window is rebased (new
 * epoch) only when the pinned prefix is structurally invalidated: session
 * switch, process restart, history shrink/compaction, or a changed head.
 */
export const EXTRACT_STABLE_WINDOW_ENV =
  'QWEN_CODE_MEMORY_EXTRACT_STABLE_WINDOW';

/** Seed window size at epoch start — the legacy 40-entry extractor tail. */
export const EXTRACT_WINDOW_SEED_ENTRIES = 40;

// Process-scoped marker persisted in the cursor: a different id proves the
// cursor's curated indices came from a previous process and must be rebased.
const PROCESS_INSTANCE_ID = randomUUID();

export function getExtractWindowProcessId(): string {
  return PROCESS_INSTANCE_ID;
}

export type ExtractWindowRebaseReason =
  | 'cold_start'
  | 'session_changed'
  | 'process_restarted'
  | 'history_shrank'
  | 'head_changed';

export interface ExtractWindowState {
  epoch: number;
  windowStartCurated: number;
  coveredCurated: number;
  windowHeadHash: string;
}

export interface ExtractWindowDecision {
  state: ExtractWindowState;
  rebased: boolean;
  reason?: ExtractWindowRebaseReason;
}

/**
 * Stable window is on by default in this build; `0`/`false`/`off`/`no`
 * restores the legacy 40-entry tail exactly.
 */
export function isStableExtractionWindowEnabled(
  env: NodeJS.ProcessEnv = process.env,
): boolean {
  const raw = env[EXTRACT_STABLE_WINDOW_ENV];
  if (raw === undefined || raw.trim() === '') return true;
  const value = raw.trim().toLowerCase();
  if (value === '0' || value === 'false' || value === 'off' || value === 'no') {
    return false;
  }
  return true;
}

export function getExtractWindowHeadHash(
  curated: Content[],
  index: number,
): string {
  const content = curated[index];
  if (!content) return '';
  return createHash('sha256')
    .update(JSON.stringify({ role: content.role, parts: content.parts ?? [] }))
    .digest('hex');
}

/** The slice handed to the extractor for the current epoch. */
export function sliceExtractWindow(
  curated: Content[],
  windowStartCurated: number,
): Content[] {
  return curated.slice(windowStartCurated);
}

/**
 * Pick the persisted window fields out of an existing cursor, if it has a
 * complete set. Used to carry the epoch across runs that fall back to the
 * legacy path (e.g. no live chat available) without destroying it.
 */
export function pickExtractWindowState(
  cursor: AutoMemoryExtractCursor,
): ExtractWindowState | undefined {
  if (
    typeof cursor.epoch !== 'number' ||
    typeof cursor.windowStartCurated !== 'number' ||
    typeof cursor.coveredCurated !== 'number' ||
    typeof cursor.windowHeadHash !== 'string'
  ) {
    return undefined;
  }
  return {
    epoch: cursor.epoch,
    windowStartCurated: cursor.windowStartCurated,
    coveredCurated: cursor.coveredCurated,
    windowHeadHash: cursor.windowHeadHash,
  };
}

/** Cursor fields persisted after an extraction run has used the window. */
export function toExtractWindowCursorFields(
  state: ExtractWindowState,
): Pick<
  AutoMemoryExtractCursor,
  | 'epoch'
  | 'windowStartCurated'
  | 'coveredCurated'
  | 'windowHeadHash'
  | 'windowProcessId'
> {
  return {
    epoch: state.epoch,
    windowStartCurated: state.windowStartCurated,
    coveredCurated: state.coveredCurated,
    windowHeadHash: state.windowHeadHash,
    windowProcessId: PROCESS_INSTANCE_ID,
  };
}

export function decideExtractWindow(params: {
  cursor: AutoMemoryExtractCursor;
  sessionId: string;
  curated: Content[];
  processId?: string;
}): ExtractWindowDecision {
  const { cursor, sessionId, curated } = params;
  const processId = params.processId ?? PROCESS_INSTANCE_ID;
  const length = curated.length;
  const seedStart = Math.max(0, length - EXTRACT_WINDOW_SEED_ENTRIES);
  const existing = pickExtractWindowState(cursor);

  const rebase = (reason: ExtractWindowRebaseReason): ExtractWindowDecision => {
    let windowStartCurated = seedStart;
    // Never skip turns that no earlier run covered: when the previous window's
    // covered index is older than the fresh seed, start there instead. Safe for
    // every structural rebase that still trusts the curated tail ordering
    // (restart, head rewrite, shrink) — a session change is the one case where
    // `coveredCurated` belongs to another transcript.
    if (
      reason !== 'session_changed' &&
      existing !== undefined &&
      existing.coveredCurated > 0 &&
      existing.coveredCurated < seedStart
    ) {
      windowStartCurated = existing.coveredCurated;
    }
    return {
      state: {
        epoch: (existing?.epoch ?? cursor.epoch ?? 0) + 1,
        windowStartCurated,
        coveredCurated: length,
        windowHeadHash: getExtractWindowHeadHash(curated, windowStartCurated),
      },
      rebased: true,
      reason,
    };
  };

  if (!existing) return rebase('cold_start');
  if (cursor.sessionId !== sessionId) return rebase('session_changed');
  if (cursor.windowProcessId !== processId) return rebase('process_restarted');
  if (
    existing.windowStartCurated < 0 ||
    existing.windowStartCurated >= length
  ) {
    return rebase('history_shrank');
  }
  if (
    getExtractWindowHeadHash(curated, existing.windowStartCurated) !==
    existing.windowHeadHash
  ) {
    return rebase('head_changed');
  }
  return {
    state: { ...existing, coveredCurated: length },
    rebased: false,
  };
}
