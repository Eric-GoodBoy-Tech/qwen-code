/**
 * @license
 * Copyright 2026 Qwen Team
 * SPDX-License-Identifier: Apache-2.0
 */

import { createHash, randomUUID } from 'node:crypto';
import type { Content } from '@google/genai';
import {
  DEFAULT_IMAGE_TOKEN_ESTIMATE,
  estimateContentChars,
} from '../services/compactionInputSlimming.js';
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
 * The window is rebased (new epoch) whenever the pinned prefix can no longer
 * be trusted: session switch, process restart, history shrink/compaction,
 * head mismatch, or budget overflow.
 */
export const EXTRACT_STABLE_WINDOW_ENV =
  'QWEN_CODE_MEMORY_EXTRACT_STABLE_WINDOW';

/** Optional char budget override before the pinned window is rebased. */
export const EXTRACT_WINDOW_MAX_CHARS_ENV =
  'QWEN_CODE_MEMORY_EXTRACT_STABLE_WINDOW_MAX_CHARS';

/** Seed window size at epoch start — the legacy 40-entry extractor tail. */
export const EXTRACT_WINDOW_SEED_ENTRIES = 40;

// ~50k tokens. Long enough to amortize a rebase while staying well inside the
// context windows of the models the extractor runs on.
export const DEFAULT_EXTRACT_WINDOW_MAX_CHARS = 200_000;

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
  | 'head_changed'
  | 'budget_overflow';

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

export function resolveExtractWindowMaxChars(
  env: NodeJS.ProcessEnv = process.env,
): number {
  const raw = env[EXTRACT_WINDOW_MAX_CHARS_ENV];
  if (raw === undefined || raw.trim() === '') {
    return DEFAULT_EXTRACT_WINDOW_MAX_CHARS;
  }
  const parsed = Number(raw.trim());
  return Number.isFinite(parsed) && parsed > 0
    ? Math.floor(parsed)
    : DEFAULT_EXTRACT_WINDOW_MAX_CHARS;
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

export function estimateExtractWindowChars(
  curated: Content[],
  windowStartCurated: number,
): number {
  let total = 0;
  for (let i = windowStartCurated; i < curated.length; i++) {
    total += estimateContentChars(curated[i], DEFAULT_IMAGE_TOKEN_ESTIMATE);
  }
  return total;
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

/**
 * Budget-fitting epoch seed.
 *
 * The legacy seed (the newest 40 entries) can by itself exceed the window
 * budget in a busy session — measured curated windows there reach ~228k chars
 * — and a seed over budget would rebase on every single run, i.e. exactly the
 * zero-reuse loop this mode exists to remove. Seed instead from the newest
 * entries that fit into HALF the budget (always at least one entry, never more
 * than the legacy 40): the other half is headroom for appended turns before the
 * next rebase, so a rebase is an occasional event rather than the steady state.
 */
function computeSeedWindowStart(curated: Content[], maxChars: number): number {
  const budget = Math.max(1, Math.floor(maxChars / 2));
  let usedChars = 0;
  let entries = 0;
  let start = curated.length;
  for (let i = curated.length - 1; i >= 0; i--) {
    if (entries >= EXTRACT_WINDOW_SEED_ENTRIES) break;
    const chars = estimateContentChars(
      curated[i],
      DEFAULT_IMAGE_TOKEN_ESTIMATE,
    );
    if (entries > 0 && usedChars + chars > budget) break;
    usedChars += chars;
    entries++;
    start = i;
  }
  return start;
}

export function decideExtractWindow(params: {
  cursor: AutoMemoryExtractCursor;
  sessionId: string;
  curated: Content[];
  maxChars: number;
  processId?: string;
}): ExtractWindowDecision {
  const { cursor, sessionId, curated, maxChars } = params;
  const processId = params.processId ?? PROCESS_INSTANCE_ID;
  const length = curated.length;
  const seedStart = computeSeedWindowStart(curated, maxChars);
  const existing = pickExtractWindowState(cursor);

  const rebase = (reason: ExtractWindowRebaseReason): ExtractWindowDecision => {
    let windowStartCurated = seedStart;
    // Never skip turns that no earlier run covered: when the previous window's
    // covered index is older than the budget-fitting seed, start there instead.
    // Safe for every rebase that still trusts the curated tail ordering
    // (restart, head rewrite, shrink, budget) — a session change is the one
    // case where `coveredCurated` belongs to another transcript.
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
  if (
    estimateExtractWindowChars(curated, existing.windowStartCurated) > maxChars
  ) {
    return rebase('budget_overflow');
  }
  return {
    state: { ...existing, coveredCurated: length },
    rebased: false,
  };
}
