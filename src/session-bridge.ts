import type {
  LanguageModelV3,
  LanguageModelV3CallOptions,
  LanguageModelV3FinishReason,
  LanguageModelV3Prompt,
  LanguageModelV3StreamPart,
  LanguageModelV3Usage,
} from "@ai-sdk/provider"
import type { Plugin } from "@opencode/plugin/effect"
import { Effect } from "effect"
import {
  AcpProcess,
  AgyNotFoundError,
  FALLBACK_MODELS,
  buildModelCatalog,
  discoverAcpModels,
  resolveAcpModelTarget,
  type AcpDiscoveredModel,
  type AcpPermissionOutcome,
  type AcpPermissionRequestParams,
  type AcpProcessOptions,
  type AcpPromptResult,
  type AcpSessionUpdate,
  type AcpContentBlock,
} from "./acp-client.js"
import {
  ACP_PROVIDER_ID,
  API_ERROR_ABORT_PATTERN,
  buildTextOnlyHandoffPromptBlocks,
  buildThinAcpPromptBlocks,
  extractToolResultsFromPrompt,
  hasUnseenNonAcpAssistantHistory,
  isSyntheticRestartAfterError,
} from "./history-sanitizer.js"
import { McpRelayServer, type McpCallToolResult, type McpToolDescriptor } from "./mcp-relay.js"

export type AcpMode = "auto" | "accept-edits" | "default" | "plan"

class PluginDisposedError extends Error {
  constructor() {
    super("Antigravity ACP plugin instance was disposed")
    this.name = "PluginDisposedError"
  }
}

/**
 * Thrown when an in-flight `ensureAcpSession` start is invalidated by a
 * concurrent `setCodeMode()` / `resetSession()` (mode flip or generation bump).
 * The stale proc is disposed without being installed; the caller retries once
 * with the current mode.
 */
class StaleAcpStartError extends Error {
  constructor() {
    super("ACP session start superseded by a mode change; retrying")
    this.name = "StaleAcpStartError"
  }
}

const NATIVE_SHADOWED_TOOLS = new Set([
  "read",
  "write",
  "edit",
  "multiedit",
  "apply_patch",
  "bash",
  "shell",
  "grep",
  "glob",
  "ls",
  "list",
])

export interface SessionContextSnapshot {
  readonly systemText: string
  readonly agent: string
  readonly toolSignature: string
  readonly tools: Record<string, { readonly description: string; readonly input: any }>
  readonly mcpSafeToOriginal: Map<string, string>
  readonly mcpDescriptors: McpToolDescriptor[]
}

export interface SessionPublicState {
  readonly sessionID: string
  readonly mode: AcpMode
  readonly codeModeEnabled: boolean
  readonly starting: boolean
  readonly acpSessionId: string | null
  readonly connected: boolean
  readonly busy: boolean
  readonly modelCount: number
  readonly relayedToolCount: number
  readonly lastError: string | null
  readonly cooldownUntil: number | null
}

interface PendingMcpCall {
  readonly toolCallId: string
  readonly mcpName: string
  readonly openCodeToolName: string
  readonly args: unknown
  emittedToOpenCode: boolean
  resolved: boolean
  readonly resolve: (res: McpCallToolResult) => void
}

interface ActiveTurn {
  controller?: ReadableStreamDefaultController<LanguageModelV3StreamPart>
  controllerCleanup?: () => void
  readonly bufferedParts: LanguageModelV3StreamPart[]
  openTextId?: string
  openReasoningId?: string
  partSeq: number
  readonly activeNativeTools: Set<string>
  readonly relayedUpdateIds: Set<string>
  readonly pendingMcpCalls: Map<string, PendingMcpCall>
  readonly dispatchedCallIds: Set<string>
  flushTimer?: ReturnType<typeof setTimeout>
  promptSettled: boolean
  abortedOnApiError?: boolean
  promptResult?: AcpPromptResult
  promptError?: Error
}

interface BridgeSession {
  readonly sessionID: string
  cwd: string
  mode: AcpMode
  codeModeEnabled: boolean
  acpSessionId?: string
  proc?: AcpProcess
  lastModelId?: string
  lastModeApplied?: AcpMode
  lastError: string | null
  lastTitleSent?: string
  contextSnapshot?: SessionContextSnapshot
  activeTurn?: ActiveTurn
  startLock?: Promise<void>
  /** Bumped by resetSession to invalidate in-flight ensureAcpSession starts. */
  startGeneration: number
  /**
   * Settles when the latest resetSession cleanup (including runStorageRemove)
   * finishes. Joiners and stale-start retries await it before reading storage
   * or spawning, so a replacement session never resumes a half-torn-down id.
   */
  resetLock?: Promise<void>
  /**
   * Set when a fresh `agy acp` session is installed while the OpenCode thread
   * already has prior turns (e.g. resume failed during a background warm).
   * Consumed by the next real turn, which restores context via text handoff.
   */
  pendingFreshHandoff?: boolean
  /**
   * Settles when the latest `proc.prompt()` so a follow-up turn can wait for a
   * cancelled prompt to settle server-side before sending a new prompt on
   * the same live proc. Never rejects; cleared on settle.
   */
  settlingPrompt?: Promise<void>
  /** Foreground reservations held by live doStream turns (excluded from idle LRU). */
  foregroundCount?: number
  /**
   * Antigravity `ask_question` call that was re-routed to OpenCode's local
   * interactive `question` tool. When the matching tool result comes back in
   * the next doStream prompt, it is forwarded to agy as a plain-text answer
   * prompt (the same way the Antigravity CLI itself delivers answers).
   */
  pendingQuestion?: PendingQuestion
}

interface PendingQuestion {
  readonly callId: string
  readonly questions: ReadonlyArray<{ readonly question: string }>
}

interface WarmSpare {
  proc: AcpProcess
  cwd: string
  holder: { owner?: BridgeSession }
}

const STORAGE_KEY_PREFIX = "antigravity-acp:session:"
const COOLDOWN_STORAGE_KEY = "antigravity-acp:cooldown"
const MODEL_CATALOG_STORAGE_KEY = "antigravity-acp:model-catalog:v1"
const MODEL_CATALOG_TTL_MS = 6 * 60 * 60 * 1000

/**
 * Upper bound on idle pre-warmed sessions. Clicking across many sessions must
 * not accumulate unbounded `agy acp` children; the LRU victim is torn down via
 * resetSession (never a session with an active turn or an in-flight start).
 */
const MAX_IDLE_WARMED_SESSIONS = 2

export type QuotaBucket = "claude-gpt" | "gemini" | "other"

type CooldownBucket = QuotaBucket | "global"

const COOLDOWN_BUCKETS: readonly CooldownBucket[] = ["global", "claude-gpt", "gemini", "other"]

/**
 * Upper bound for waiting on a previously cancelled ACP prompt to settle
 * server-side before a follow-up turn sends a new prompt on the same live
 * proc. If it hangs, the proc is torn down so the replacement starts clean.
 */
const SETTLING_PROMPT_TIMEOUT_MS = 2000

/**
 * Maps a model ID to its upstream quota pool. Claude & GPT models share one
 * pool, Gemini & Orchestrator models share another, everything else is "other".
 */
export function quotaBucketForModel(modelID: string): QuotaBucket {
  const id = (modelID ?? "").toLowerCase()
  if (id.includes("claude") || id.includes("gpt")) return "claude-gpt"
  if (id.includes("gemini") || id.includes("orchestrator")) return "gemini"
  return "other"
}

/**
 * Fatal/auth-level errors trip the "global" cooldown (all models blocked).
 * Everything else reaching the breaker is pool quota exhaustion.
 */
function isGlobalBreakerError(message: string): boolean {
  return /\(code 403\)|PERMISSION_DENIED/i.test(message)
}

let sharedCatalog: readonly AcpDiscoveredModel[] | undefined
let sharedCatalogUpdatedAt = 0
let sharedDiscoveryPromise: Promise<readonly AcpDiscoveredModel[]> | undefined

export function resetSharedCatalogForTests(): void {
  sharedCatalog = undefined
  sharedCatalogUpdatedAt = 0
  sharedDiscoveryPromise = undefined
}

function isValidCatalog(raw: unknown): raw is readonly AcpDiscoveredModel[] {
  if (!Array.isArray(raw) || raw.length === 0) return false
  return raw.every(
    (item) =>
      item &&
      typeof item === "object" &&
      typeof (item as any).id === "string" &&
      typeof (item as any).name === "string" &&
      typeof (item as any).defaultAcpModelId === "string" &&
      Array.isArray((item as any).variants),
  )
}

function normalizeCatalog(models: readonly AcpDiscoveredModel[]): readonly AcpDiscoveredModel[] {
  return models.map((m) => ({
    ...m,
    family: m.family || m.id,
  }))
}

function catalogSignature(models: readonly AcpDiscoveredModel[]): string {
  return JSON.stringify(models.map((m) => [m.id, m.name, m.family, m.defaultAcpModelId, m.variants]))
}

// Antigravity `session/prompt` usage semantics (verified against agy 1.5.14):
//   inputTokens       = NEW (non-cached) input tokens for the turn
//   cachedReadTokens  = cached portion of the input
//   cachedWriteTokens = cache-write portion of the input
//   outputTokens      = output text + thought tokens
//   thoughtTokens     = reasoning portion of the output
//   totalTokens       = inputTokens + cachedRead + output (matches usage_update.used)
// OpenCode derives the displayed per-message "input" from inputTokens.noCache,
// so subtracting the cached portions here (as upstream did) zeroed the display.
function zeroUsage(raw?: AcpPromptResult["usage"]): LanguageModelV3Usage {
  const noCache = raw?.inputTokens
  const cacheRead = raw?.cachedReadTokens
  const cacheWrite = raw?.cachedWriteTokens
  const outputTotal = raw?.outputTokens
  const reasoning = raw?.thoughtTokens
  const inputTotal =
    noCache !== undefined || cacheRead !== undefined || cacheWrite !== undefined
      ? (noCache ?? 0) + (cacheRead ?? 0) + (cacheWrite ?? 0)
      : undefined
  const text =
    outputTotal !== undefined ? Math.max(0, outputTotal - (reasoning ?? 0)) : undefined
  return {
    inputTokens: {
      total: inputTotal,
      noCache,
      cacheRead,
      cacheWrite,
    },
    outputTokens: {
      total: outputTotal,
      text,
      reasoning,
    },
  }
}

function sanitizeMcpToolName(name: string, used: Set<string>): string {
  let base = name.replace(/[^a-zA-Z0-9_-]/g, "_")
  if (!base) base = "tool"
  let candidate = base
  let n = 2
  while (used.has(candidate)) {
    candidate = `${base}_${n++}`
  }
  used.add(candidate)
  return candidate
}

/**
 * Translates Antigravity's native `ask_question` input
 * ({ questions: [{ question, options: string[], is_multi_select }] }) into
 * OpenCode's local `question` tool input
 * ({ questions: [{ question, header, options: [{ label, description }], multiple }] }).
 * Returns undefined when the shape is not recognized; the caller then falls
 * back to the generic provider-executed rendering.
 */
function translateAskQuestionInput(
  raw: unknown,
): { input: Record<string, unknown>; questions: ReadonlyArray<{ question: string }> } | undefined {
  if (!raw || typeof raw !== "object") return undefined
  const list = (raw as Record<string, unknown>).questions
  if (!Array.isArray(list) || list.length === 0) return undefined
  const out: Record<string, unknown>[] = []
  const originals: { question: string }[] = []
  for (const entry of list) {
    if (!entry || typeof entry !== "object") return undefined
    const q = entry as Record<string, unknown>
    const question = String(q.question ?? "").trim()
    if (!question) return undefined
    const rawOptions = Array.isArray(q.options) ? q.options : []
    const options = rawOptions.map((o) => ({
      label: String(o),
      description: "",
    }))
    const header = question.length > 24 ? `${question.slice(0, 24).trimEnd()}…` : question
    originals.push({ question })
    out.push({
      question,
      header,
      options,
      multiple: q.is_multi_select === true || q.multiple === true,
    })
  }
  return { input: { questions: out }, questions: originals }
}

/**
 * Finds the tool result for a re-routed ask_question call in the next prompt
 * and formats it as a plain-text answer for agy. agy ended its turn when it
 * asked, so the answer travels as the next user prompt. Returns a dismissal
 * note when the user cancelled, and undefined when no matching result exists
 * (e.g. the user typed a normal message instead).
 */
function buildQuestionAnswerBlocks(
  pending: PendingQuestion,
  prompt: LanguageModelV3Prompt,
): AcpContentBlock[] | undefined {
  for (let i = prompt.length - 1; i >= 0; i--) {
    const msg = prompt[i]
    if (msg.role !== "tool" || !Array.isArray(msg.content)) continue
    for (const part of msg.content as ReadonlyArray<any>) {
      if (part?.type !== "tool-result" || part.toolCallId !== pending.callId) continue
      const answers = extractQuestionAnswers(part.output)
      if (!answers) {
        return [
          {
            type: "text",
            text: "(The user dismissed the question without answering. Proceed with your recommended option(s).)",
          },
        ]
      }
      const lines: string[] = ["The user answered your question(s):"]
      for (let n = 0; n < pending.questions.length; n++) {
        const q = pending.questions[n]?.question ?? `Question ${n + 1}`
        const a = answers[n] && answers[n].length > 0 ? answers[n].join(", ") : "(no answer)"
        lines.push(`Q${n + 1}: ${q}`)
        lines.push(`A${n + 1}: ${a}`)
      }
      return [{ type: "text", text: lines.join("\n") }]
    }
  }
  return undefined
}

/** OpenCode's question tool returns { answers: string[][] } as its output. */
function extractQuestionAnswers(output: unknown): string[][] | undefined {
  let value: unknown
  if (output && typeof output === "object") {
    const o = output as Record<string, unknown>
    if (o.type === "json") value = o.value
    else if (o.type === "text" && typeof o.value === "string") {
      try {
        value = JSON.parse(o.value)
      } catch {
        return undefined
      }
    } else if (Array.isArray((o as any).answers)) {
      value = o
    }
  }
  const answers = (value as any)?.answers
  if (!Array.isArray(answers)) return undefined
  return answers.map((a: unknown) =>
    Array.isArray(a) ? a.map(String) : a === null || a === undefined ? [] : [String(a)],
  )
}

function buildDescriptors(
  tools: Record<string, { readonly description: string; readonly input: any }>,
  includeNativeShadowed: boolean,
): { mcpDescriptors: McpToolDescriptor[]; mcpSafeToOriginal: Map<string, string> } {
  const used = new Set<string>()
  const mcpSafeToOriginal = new Map<string, string>()
  const mcpDescriptors: McpToolDescriptor[] = []

  for (const [toolName, info] of Object.entries(tools)) {
    if (toolName === ACP_PROVIDER_ID) continue
    if (!includeNativeShadowed && NATIVE_SHADOWED_TOOLS.has(toolName)) continue
    const safeName = sanitizeMcpToolName(toolName, used)
    mcpSafeToOriginal.set(safeName, toolName)
    mcpDescriptors.push({
      name: safeName,
      description: info.description || `OpenCode tool: ${toolName}`,
      inputSchema: info.input,
    })
  }
  return { mcpDescriptors, mcpSafeToOriginal }
}

function extractToolContentText(update: AcpSessionUpdate): unknown {
  if (update.rawOutput !== undefined) return update.rawOutput
  if (Array.isArray(update.content)) {
    const texts: string[] = []
    for (const block of update.content as any[]) {
      if (block?.type === "text" && typeof block.text === "string") texts.push(block.text)
      else if (block?.content?.type === "text" && typeof block.content.text === "string") {
        texts.push(block.content.text)
      } else {
        texts.push(JSON.stringify(block))
      }
    }
    if (texts.length > 0) return texts.join("\n")
  }
  return { status: update.status ?? "completed", title: update.title }
}

export class SessionBridgeManager {
  readonly mcpRelay: McpRelayServer
  private sessions = new Map<string, BridgeSession>()
  private catalog: readonly AcpDiscoveredModel[] = FALLBACK_MODELS
  private onStateChanged?: (sessionID: string) => void
  private onCatalogChanged?: () => void
  private discoverModelsFn: (cwd: string) => Promise<readonly AcpDiscoveredModel[]>
  private createProcFn: (options: AcpProcessOptions) => AcpProcess
  private cooldownsByBucket = new Map<CooldownBucket, { until: number; reason: string }>()
  private cooldownLoadPromise?: Promise<void> | null
  private disposed = false
  private activeProcs = new Set<AcpProcess>()
  private intentionallyDisposed = new WeakSet<AcpProcess>()
  private warmSpare?: WarmSpare
  private warmSpareLock?: Promise<void>
  /**
   * OpenCode session IDs whose proc was installed by a background warm and
   * which have not run a real turn yet (LRU order, oldest first). Bounded by
   * MAX_IDLE_WARMED_SESSIONS so clicking across sessions can't accumulate
   * unbounded idle `agy acp` children.
   */
  private warmedIdleSessions: string[] = []

  constructor(
    private readonly ctx: Plugin.Context,
    private readonly defaultCwd: string,
    onStateChanged?: (sessionID: string) => void,
    options?: {
      readonly onCatalogChanged?: () => void
      readonly discoverModels?: (cwd: string) => Promise<readonly AcpDiscoveredModel[]>
      /** Injectable proc factory for tests; defaults to real `AcpProcess`. */
      readonly createProc?: (options: AcpProcessOptions) => AcpProcess
    },
  ) {
    this.onStateChanged = onStateChanged
    this.onCatalogChanged = options?.onCatalogChanged
    this.discoverModelsFn = options?.discoverModels ?? ((cwd) => discoverAcpModels(cwd))
    this.createProcFn = options?.createProc ?? ((opts) => new AcpProcess(opts))
    if (sharedCatalog && sharedCatalog.length > 0) {
      this.catalog = sharedCatalog
    }
    this.mcpRelay = new McpRelayServer({
      getTools: (sessionID) => {
        const s = this.sessions.get(sessionID)
        return s?.contextSnapshot?.mcpDescriptors ?? []
      },
      callTool: (sessionID, toolName, args, signal) =>
        this.handleIncomingMcpCall(sessionID, toolName, args, signal),
    })
  }

  async start(): Promise<void> {
    // Lazy: only start the loopback relay when at least one session has
    // opted into Code Mode. Pure ACP sessions never need it.
    for (const s of this.sessions.values()) {
      if (s.codeModeEnabled) {
        await this.mcpRelay.start()
        return
      }
    }
  }

  getModels(): readonly AcpDiscoveredModel[] {
    return this.catalog
  }

  async loadCachedModels(): Promise<{ readonly stale: boolean; readonly changed: boolean }> {
    if (sharedCatalog && sharedCatalog.length > 0) {
      const prevSig = catalogSignature(this.catalog)
      this.catalog = sharedCatalog
      return {
        stale: Date.now() - sharedCatalogUpdatedAt > MODEL_CATALOG_TTL_MS,
        changed: prevSig !== catalogSignature(sharedCatalog),
      }
    }

    try {
      const raw = (await Effect.runPromise(
        this.ctx.storage.get(MODEL_CATALOG_STORAGE_KEY),
      )) as { readonly updatedAt?: unknown; readonly models?: unknown } | undefined
      if (raw && typeof raw === "object" && isValidCatalog(raw.models)) {
        const normalized = normalizeCatalog(raw.models)
        const updatedAt = typeof raw.updatedAt === "number" ? raw.updatedAt : 0
        const prevSig = catalogSignature(this.catalog)
        sharedCatalog = normalized
        sharedCatalogUpdatedAt = updatedAt
        this.catalog = normalized
        return {
          stale: Date.now() - updatedAt > MODEL_CATALOG_TTL_MS,
          changed: prevSig !== catalogSignature(normalized),
        }
      }
    } catch {}

    return { stale: true, changed: false }
  }

  private async persistCatalog(
    models: readonly AcpDiscoveredModel[],
    updatedAt: number,
  ): Promise<void> {
    try {
      await Effect.runPromise(
        this.ctx.storage.set(MODEL_CATALOG_STORAGE_KEY, {
          version: 1,
          updatedAt,
          models: models as unknown as any,
        }),
      )
    } catch {}
  }

  async refreshModels(options?: {
    readonly force?: boolean
  }): Promise<{ readonly models: readonly AcpDiscoveredModel[]; readonly changed: boolean }> {
    if (
      !options?.force &&
      sharedCatalog &&
      sharedCatalog.length > 0 &&
      Date.now() - sharedCatalogUpdatedAt <= MODEL_CATALOG_TTL_MS
    ) {
      const changed = catalogSignature(this.catalog) !== catalogSignature(sharedCatalog)
      this.catalog = sharedCatalog
      return { models: this.catalog, changed }
    }

    if (!sharedDiscoveryPromise) {
      sharedDiscoveryPromise = (async () => {
        const discovered = await this.discoverModelsFn(this.defaultCwd)
        const normalized = normalizeCatalog(discovered)
        if (normalized.length > 0) {
          const updatedAt = Date.now()
          sharedCatalog = normalized
          sharedCatalogUpdatedAt = updatedAt
          await this.persistCatalog(normalized, updatedAt)
        }
        return normalized
      })().finally(() => {
        sharedDiscoveryPromise = undefined
      })
    }

    const normalized = await sharedDiscoveryPromise
    if (normalized.length === 0) {
      return { models: this.catalog, changed: false }
    }

    const prevSig = catalogSignature(this.catalog)
    this.catalog = normalized
    const changed = prevSig !== catalogSignature(normalized)
    return { models: normalized, changed }
  }

  async refreshModelsInBackground(): Promise<boolean> {
    try {
      const { changed } = await this.refreshModels({ force: false })
      if (!this.disposed && changed) {
        this.onCatalogChanged?.()
      }
      return changed
    } catch {
      return false
    }
  }

  hasSession(sessionID: string): boolean {
    return this.sessions.has(sessionID)
  }

  /**
   * Loads persisted per-bucket cooldowns once and shares the in-flight read:
   * concurrent callers await the same promise, so nobody proceeds on a
   * half-hydrated map and the storage read happens exactly once.
   */
  private loadPersistedCooldown(): Promise<void> {
    if (!this.cooldownLoadPromise) {
      this.cooldownLoadPromise = (async () => {
        const read = async (key: string): Promise<unknown> => {
          try {
            return await Effect.runPromise(this.ctx.storage.get(key))
          } catch {
            return undefined
          }
        }
        const applyRecord = (bucket: CooldownBucket, val: unknown) => {
          if (val && typeof val === "object") {
            const { until, reason } = val as { readonly until?: unknown; readonly reason?: unknown }
            if (typeof until === "number" && until > Date.now()) {
              const cur = this.cooldownsByBucket.get(bucket)
              if (!cur || until > cur.until) {
                this.cooldownsByBucket.set(bucket, {
                  until,
                  reason: typeof reason === "string" ? reason : "upstream error",
                })
              }
            }
          }
        }
        // Legacy single-key shape (pre-bucket) migrates into the global bucket.
        applyRecord("global", await read(COOLDOWN_STORAGE_KEY))
        for (const bucket of COOLDOWN_BUCKETS) {
          applyRecord(bucket, await read(`${COOLDOWN_STORAGE_KEY}:${bucket}`))
        }
      })()
    }
    return this.cooldownLoadPromise
  }

  private persistCooldownForBucket(bucket: CooldownBucket, until: number, reason: string): void {
    try {
      Effect.runFork(
        this.ctx.storage
          .set(`${COOLDOWN_STORAGE_KEY}:${bucket}`, { until, reason })
          .pipe(Effect.catch(() => Effect.void)),
      )
    } catch {}
  }

  private tripCircuitBreaker(
    reason: string,
    bucket: CooldownBucket = "global",
  ): { until: number; reason: string } {
    const until = Date.now() + 60_000
    const trimmed = reason.trim().slice(0, 180)
    this.cooldownsByBucket.set(bucket, { until, reason: trimmed })
    this.persistCooldownForBucket(bucket, until, trimmed)
    return { until, reason: trimmed }
  }

  /** Effective cooldown for a model: the global breaker wins, else its quota pool. */
  private cooldownForModel(modelID: string): { until: number; reason: string } | undefined {
    const now = Date.now()
    const g = this.cooldownsByBucket.get("global")
    if (g && g.until > now) return g
    const b = this.cooldownsByBucket.get(quotaBucketForModel(modelID))
    if (b && b.until > now) return b
    return undefined
  }

  private activeCooldownForSession(s: BridgeSession | undefined): { until: number; reason: string } | undefined {
    const now = Date.now()
    const g = this.cooldownsByBucket.get("global")
    if (g && g.until > now) return g
    const bucket = s?.lastModelId ? quotaBucketForModel(s.lastModelId) : undefined
    if (bucket) {
      const b = this.cooldownsByBucket.get(bucket)
      if (b && b.until > now) return b
    }
    return undefined
  }

  private getOrCreateSession(sessionID: string): BridgeSession {
    let s = this.sessions.get(sessionID)
    if (!s) {
      s = {
        sessionID,
        cwd: this.defaultCwd,
        mode: "auto",
        codeModeEnabled: false,
        lastError: null,
        startGeneration: 0,
      }
      this.sessions.set(sessionID, s)
    }
    return s
  }

  updateContextSnapshot(
    sessionID: string,
    input: {
      readonly systemText: string
      readonly agent: string
      readonly tools: Record<string, { readonly description: string; readonly input: any }>
    },
  ): void {
    const s = this.getOrCreateSession(sessionID)
    // Fingerprint keys AND descriptions AND schemas: a same-keyed tool whose
    // description or input schema changed must rebuild the descriptors.
    const toolSignature = Object.entries(input.tools)
      .map(([name, info]) => {
        let schema: string
        try {
          const stringified = JSON.stringify(info.input ?? null)
          schema = typeof stringified === "string" ? stringified : String(typeof info.input)
        } catch {
          schema = String(typeof info.input)
        }
        return `${name}\0${info.description ?? ""}\0${schema}`
      })
      .sort()
      .join("\0")
    const prev = s.contextSnapshot
    if (
      prev &&
      prev.systemText === input.systemText &&
      prev.agent === input.agent &&
      prev.toolSignature === toolSignature
    ) {
      return
    }
    const { mcpDescriptors, mcpSafeToOriginal } = s.codeModeEnabled
      ? buildDescriptors(input.tools, false)
      : { mcpDescriptors: [], mcpSafeToOriginal: new Map<string, string>() }
    s.contextSnapshot = {
      systemText: input.systemText,
      agent: input.agent,
      toolSignature,
      tools: input.tools,
      mcpDescriptors,
      mcpSafeToOriginal,
    }
  }

  isCircuitOpen(): boolean {
    const now = Date.now()
    for (const rec of this.cooldownsByBucket.values()) {
      if (rec.until > now) return true
    }
    return false
  }

  /**
   * Pre-warm `agy acp` in the background so the first real turn doesn't pay
   * a cold spawn. With a real session ID the session's own `ensureAcpSession`
   * runs (deduplicated via `s.startLock`); with an empty ID (unsaved new tab)
   * a spare initialized + authenticated proc is parked for the first real
   * `ses_...` session to claim. Never throws — the foreground turn surfaces
   * and retries any failure itself.
   */
  async warmSession(sessionID?: string): Promise<SessionPublicState> {
    if (this.disposed) return this.getState(sessionID ?? "")
    try {
      await this.loadPersistedCooldown()
    } catch {}
    if (this.isCircuitOpen()) return this.getState(sessionID ?? "")
    if (sessionID && sessionID.length > 0) {
      const s = this.getOrCreateSession(sessionID)
      if (!s.proc && !s.startLock) {
        // Pre-warm in the background; a real turn joins via startLock or reuses the install.
        // No retry on stale: if a reset invalidates this warm, it stays torn
        // down (the foreground turn re-warms on demand) instead of eagerly
        // reinstalling behind the reset. markFreshHandoff arms the text-handoff
        // restore in case resume fails and the fresh session owns no history.
        void this.ensureAcpSession(s, { retryOnStale: false, markFreshHandoff: true })
          .then(() => this.trackWarmedIdle(sessionID))
          .catch(() => {})
      }
      return this.getState(sessionID)
    }
    void this.ensureWarmSpare().catch(() => {})
    return this.getState("")
  }

  private ensureWarmSpare(): Promise<void> {
    if (this.disposed) return Promise.resolve()
    const existing = this.warmSpare
    if (existing && !existing.proc.isClosed) return Promise.resolve()
    if (existing) this.warmSpare = undefined
    if (this.warmSpareLock) return this.warmSpareLock
    const p = (async () => {
      const holder: { owner?: BridgeSession } = {}
      const proc = this.createProcFn({
        cwd: this.defaultCwd,
        dangerouslySkipPermissions: false,
        onUpdate: (_sid, update) => {
          if (holder.owner) this.handleAcpUpdate(holder.owner, update)
        },
        onPermissionRequest: (params) =>
          holder.owner
            ? this.handlePermissionRequest(holder.owner, params)
            : Promise.resolve({ outcome: "cancelled" as const }),
        onExit: (code, signal, tail) => {
          this.activeProcs.delete(proc)
          const owner = holder.owner
          if (owner && owner.proc === proc) {
            // Claimed and installed: standard session cleanup. Otherwise the
            // spare was superseded before install — just drop the slot.
            this.handleProcExit(owner, proc, code, signal, tail)
          } else if (this.warmSpare?.proc === proc) {
            this.warmSpare = undefined
          }
        },
      })
      this.activeProcs.add(proc)
      try {
        await proc.initialize()
        if (this.disposed) throw new PluginDisposedError()
        try {
          await proc.authenticate()
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err)
          if (!/already (authenticated|logged in|signed in)/i.test(msg)) throw err
        }
        if (this.disposed) throw new PluginDisposedError()
        if (proc.isClosed) throw new Error("warmed agy acp closed before use")
        this.warmSpare = { proc, cwd: this.defaultCwd, holder }
      } catch {
        this.activeProcs.delete(proc)
        try {
          await proc.dispose()
        } catch {}
        if (this.warmSpare?.proc === proc) this.warmSpare = undefined
      }
    })()
    this.warmSpareLock = p.finally(() => {
      this.warmSpareLock = undefined
    })
    return this.warmSpareLock
  }

  private async trackWarmedIdle(sessionID: string): Promise<void> {
    if (this.disposed) return
    const s = this.sessions.get(sessionID)
    // Only genuinely idle warmed sessions count: an in-flight start, a live
    // turn, a foreground reservation, a dead proc, or a torn-down session
    // means there's nothing to bound.
    if (!s || (s.foregroundCount ?? 0) > 0 || !s.proc || s.proc.isClosed || s.activeTurn || s.startLock || !s.acpSessionId) return
    this.untrackWarmedIdle(sessionID)
    this.warmedIdleSessions.push(sessionID)
    while (this.warmedIdleSessions.length > MAX_IDLE_WARMED_SESSIONS) {
      const victimID = this.warmedIdleSessions.find((id) => {
        const v = this.sessions.get(id)
        return (
          !!v &&
          !v.activeTurn &&
          !v.startLock &&
          !(v.foregroundCount && v.foregroundCount > 0) &&
          v.proc &&
          !v.proc.isClosed
        )
      })
      if (!victimID) break
      // Re-verify immediately before teardown so a turn that just started is
      // never interrupted by eviction.
      const victim = this.sessions.get(victimID)
      if (!victim || victim.activeTurn || victim.startLock || (victim.foregroundCount && victim.foregroundCount > 0)) break
      this.untrackWarmedIdle(victimID)
      try {
        // LRU eviction disposes ONLY the in-memory proc and clears s.proc —
        // it must NOT call resetSession / runStorageRemove, which would erase
        // the persisted acpSessionId needed for a later resume.
        const proc = victim.proc
        if (proc) {
          this.intentionallyDisposed.add(proc)
          this.activeProcs.delete(proc)
          if (victim.proc === proc) victim.proc = undefined
          if (!proc.isClosed) {
            try {
              await proc.dispose()
            } catch {}
          }
        }
      } catch {}
    }
  }

  private untrackWarmedIdle(sessionID: string): void {
    const idx = this.warmedIdleSessions.indexOf(sessionID)
    if (idx !== -1) this.warmedIdleSessions.splice(idx, 1)
  }

  getState(sessionID: string): SessionPublicState {
    const s = this.sessions.get(sessionID)
    return {
      sessionID,
      mode: s?.mode ?? "auto",
      codeModeEnabled: s?.codeModeEnabled ?? false,
      starting: Boolean(s?.startLock),
      acpSessionId: s?.acpSessionId ?? null,
      connected: Boolean(s?.proc && !s.proc.isClosed && s.acpSessionId),
      busy: Boolean(s?.activeTurn),
      modelCount: this.catalog.length,
      relayedToolCount: s?.contextSnapshot?.mcpDescriptors.length ?? 0,
      lastError: s?.lastError ?? null,
      cooldownUntil: this.activeCooldownForSession(s)?.until ?? null,
    }
  }

  async setMode(sessionID: string, mode: AcpMode): Promise<SessionPublicState> {
    const s = this.getOrCreateSession(sessionID)
    s.mode = mode
    if (s.proc && !s.proc.isClosed && s.acpSessionId) {
      try {
        await s.proc.setMode(s.acpSessionId, mode)
        s.lastModeApplied = mode
      } catch (err) {
        s.lastError = err instanceof Error ? err.message : String(err)
      }
    }
    this.notifyChanged(sessionID)
    return this.getState(sessionID)
  }

  async setCodeMode(sessionID: string, codeModeEnabled: boolean): Promise<SessionPublicState> {
    const s = this.getOrCreateSession(sessionID)
    s.codeModeEnabled = codeModeEnabled
    if (s.contextSnapshot) {
      const { mcpDescriptors, mcpSafeToOriginal } = codeModeEnabled
        ? buildDescriptors(s.contextSnapshot.tools, false)
        : { mcpDescriptors: [], mcpSafeToOriginal: new Map<string, string>() }
      s.contextSnapshot = {
        ...s.contextSnapshot,
        mcpDescriptors,
        mcpSafeToOriginal,
      }
    }
    // Reset ACP session so agy re-fetches MCP tools/list cleanly on next turn
    await this.resetSession(sessionID)
    return this.getState(sessionID)
  }

  async resetSession(sessionID: string): Promise<SessionPublicState> {
    const s = this.sessions.get(sessionID)
    if (s) {
      // Publish a completion barrier BEFORE doing anything async: joiners in
      // ensureAcpSession (and its stale-start retry) await this lock, so a
      // replacement session never reads storage or spawns until our cleanup —
      // including runStorageRemove below — has fully finished.
      let releaseReset!: () => void
      const resetGate = new Promise<void>((r) => {
        releaseReset = r
      })
      const prevReset = s.resetLock
      s.resetLock = resetGate
      // A torn-down session is idle-warmed no longer; drop it so stale
      // entries can't inflate the cap or get evicted later.
      this.untrackWarmedIdle(sessionID)
      try {
        if (prevReset) {
          try {
            await prevReset
          } catch {}
        }
        // Invalidate any in-flight ensureAcpSession start, then wait for it to
        // settle so it cannot install a stale proc after we clean up.
        // Note: a foreground turn already awaiting ensureAcpSession transparently
        // retries after this reset and installs with the current mode (seamless
        // setCodeMode handoff) — that post-reset install is by design. Background
        // warms never retry, so a user reset during a pure warm stays torn down.
        s.startGeneration++
        const inFlight = s.startLock
        if (inFlight) {
          try {
            await inFlight
          } catch {}
        }
        await this.handleInterrupt(sessionID, "Session reset by user")
        if (s.proc) {
          const proc = s.proc
          const acpSid = s.acpSessionId
          this.intentionallyDisposed.add(proc)
          s.proc = undefined
          s.acpSessionId = undefined
          if (acpSid) await proc.closeSession(acpSid)
          await proc.dispose()
          this.activeProcs.delete(proc)
        }
        s.acpSessionId = undefined
        s.lastModelId = undefined
        s.lastModeApplied = undefined
        s.lastError = null
        s.pendingFreshHandoff = undefined
        s.settlingPrompt = undefined
        this.untrackWarmedIdle(sessionID)
        await this.runStorageRemove(sessionID)
      } finally {
        if (s.resetLock === resetGate) s.resetLock = undefined
        releaseReset()
      }
    }
    this.notifyChanged(sessionID)
    return this.getState(sessionID)
  }

  async handleInterrupt(sessionID: string, reason = "Cancelled"): Promise<void> {
    const s = this.sessions.get(sessionID)
    if (!s?.activeTurn) return
    const turn = s.activeTurn
    if (turn.flushTimer) {
      clearTimeout(turn.flushTimer)
      turn.flushTimer = undefined
    }
    // NOTE: do NOT detach the controller here — finishControllerWithError
    // detaches it itself after enqueueing the error + close. Detaching first
    // would orphan the stream: the error would sit in bufferedParts and any
    // reader would hang forever on a never-closing stream.
    for (const call of turn.pendingMcpCalls.values()) {
      if (!call.resolved) {
        call.resolved = true
        call.resolve({
          content: [{ type: "text", text: reason }],
          isError: true,
        })
      }
    }
    turn.pendingMcpCalls.clear()
    turn.dispatchedCallIds.clear()

    if (s.proc && !s.proc.isClosed && s.acpSessionId) {
      try {
        await s.proc.cancel(s.acpSessionId)
      } catch {}
    }
    this.finishControllerWithError(turn, new Error(reason))
    s.activeTurn = undefined
    this.notifyChanged(sessionID)
  }

  private finishTurnCleanStop(
    s: BridgeSession,
    turn: ActiveTurn,
    text: string,
    rawFinish: string,
  ): void {
    if (turn.flushTimer) {
      clearTimeout(turn.flushTimer)
      turn.flushTimer = undefined
    }
    for (const call of turn.pendingMcpCalls.values()) {
      if (!call.resolved) {
        call.resolved = true
        call.resolve({
          content: [{ type: "text", text }],
          isError: true,
        })
      }
    }
    turn.pendingMcpCalls.clear()
    turn.dispatchedCallIds.clear()

    this.closeOpenBlocks(turn)
    const textId = `t_${++turn.partSeq}`
    this.emitPart(turn, {
      type: "text-start",
      id: textId,
      providerMetadata: { [ACP_PROVIDER_ID]: { acp: true } },
    })
    this.emitPart(turn, {
      type: "text-delta",
      id: textId,
      delta: text,
    })
    this.emitPart(turn, {
      type: "text-end",
      id: textId,
      providerMetadata: { [ACP_PROVIDER_ID]: { acp: true } },
    })
    this.emitPart(turn, {
      type: "finish",
      finishReason: { unified: "stop", raw: rawFinish },
      usage: zeroUsage(),
      providerMetadata: { [ACP_PROVIDER_ID]: { acp: true } },
    })
    const ctrl = this.detachController(turn)
    try {
      ctrl?.close()
    } catch {}
    if (s.activeTurn === turn) {
      s.activeTurn = undefined
    }
  }

  async disposeAll(): Promise<void> {
    this.disposed = true
    // Let an in-flight spare warm settle, then dispose it so zero orphan procs leak.
    const warmLock = this.warmSpareLock
    if (warmLock) {
      try {
        await warmLock
      } catch {}
    }
    const spare = this.warmSpare
    if (spare) {
      this.warmSpare = undefined
      this.intentionallyDisposed.add(spare.proc)
      this.activeProcs.delete(spare.proc)
      try {
        await spare.proc.dispose()
      } catch {}
    }
    const snapshot = [...this.sessions.values()]
    for (const s of snapshot) {
      if (s.activeTurn) {
        this.finishTurnCleanStop(
          s,
          s.activeTurn,
          "Antigravity ACP plugin was disabled or reloaded. Re-enable or send a new message to continue.",
          "plugin_unloaded",
        )
      }
      if (s.proc) {
        const proc = s.proc
        const acpSid = s.acpSessionId
        this.intentionallyDisposed.add(proc)
        s.proc = undefined
        s.acpSessionId = undefined
        if (!proc.isClosed && acpSid) {
          proc.notifyCancel(acpSid)
        }
        await proc.dispose()
        this.activeProcs.delete(proc)
      }
    }
    for (const proc of [...this.activeProcs]) {
      this.intentionallyDisposed.add(proc)
      await proc.dispose()
    }
    this.activeProcs.clear()
    this.sessions.clear()
    this.warmedIdleSessions.length = 0
    await this.mcpRelay.stop()
  }

  private notifyChanged(sessionID: string): void {
    try {
      this.onStateChanged?.(sessionID)
    } catch {}
  }

  private async runStorageGet(sessionID: string): Promise<string | undefined> {
    try {
      const val = await Effect.runPromise(this.ctx.storage.get(`${STORAGE_KEY_PREFIX}${sessionID}`))
      return typeof val === "string" && val.length > 0 ? val : undefined
    } catch {
      return undefined
    }
  }

  private async runStorageSet(sessionID: string, acpSessionId: string): Promise<void> {
    try {
      await Effect.runPromise(
        this.ctx.storage.set(`${STORAGE_KEY_PREFIX}${sessionID}`, acpSessionId),
      )
    } catch {}
  }

  private async runStorageRemove(sessionID: string): Promise<void> {
    try {
      await Effect.runPromise(this.ctx.storage.remove(`${STORAGE_KEY_PREFIX}${sessionID}`))
    } catch {}
  }

  private async ensureAcpSession(
    s: BridgeSession,
    opts?: { readonly retryOnStale?: boolean; readonly markFreshHandoff?: boolean },
  ): Promise<{ proc: AcpProcess; acpSessionId: string; freshSession: boolean }> {
    if (this.disposed) throw new PluginDisposedError()
    // Join in-flight work instead of fanning out: every waiter that survives a
    // superseded lock re-reads state, so only one spawner ever exists and no
    // orphan procs are left behind. Each continue follows a settled await, so
    // the loop always terminates (return or single-spawner break).
    for (;;) {
      if (this.disposed) throw new PluginDisposedError()
      // A reset in progress tears down proc + storage; wait for its completion
      // barrier before reading storage or spawning a replacement.
      const reset = s.resetLock
      if (reset) {
        try {
          await reset
        } catch {}
        continue
      }
      const lock = s.startLock
      if (lock) {
        try {
          await lock
        } catch (err) {
          if (!(err instanceof StaleAcpStartError)) throw err
        }
        continue
      }
      if (s.proc && !s.proc.isClosed && s.acpSessionId) {
        return { proc: s.proc, acpSessionId: s.acpSessionId, freshSession: false }
      }
      break
    }
    if (this.disposed) throw new PluginDisposedError()

    // Capture the mode/generation up front: a concurrent setCodeMode() /
    // resetSession() invalidates this start before it installs s.proc.
    const startCodeMode = s.codeModeEnabled
    const startGeneration = s.startGeneration
    // Re-checked after every await below: an invalidation landing mid-start
    // must abort before any further side effects (spawn, storage read, resume,
    // session/new), not just before install.
    const throwIfStale = () => {
      if (s.startGeneration !== startGeneration || s.codeModeEnabled !== startCodeMode) {
        throw new StaleAcpStartError()
      }
    }
    let freshSession = false
    // Boxed so the adoption below survives analysis: assignments made inside
    // the start closure are invisible to outer control-flow narrowing, which
    // would otherwise shrink a plain `let` to its `undefined` initializer.
    const adoption: { winner?: { proc: AcpProcess; acpSessionId: string } } = {}
    const startPromise = (async () => {
      try {
        const info = await Effect.runPromise(
          this.ctx.session.get({ sessionID: s.sessionID as any }),
        )
        const dir = (info as any)?.location?.directory ?? (info as any)?.directory
        if (typeof dir === "string" && dir.length > 0) {
          s.cwd = dir
        }
      } catch {}

      if (this.disposed) throw new PluginDisposedError()
      const codeModeEnabled = s.codeModeEnabled
      let relayUrl = ""
      if (codeModeEnabled) {
        await this.mcpRelay.start()
        if (this.disposed) throw new PluginDisposedError()
        relayUrl = this.mcpRelay.urlForSession(s.sessionID)
      }
      if (this.disposed) throw new PluginDisposedError()
      throwIfStale()

      let disposedByBridge = false
      // Claim a pre-warmed spare when it matches this session's cwd (already
      // initialize()d + authenticate()d, mode-agnostic: MCP servers attach at
      // session/new). Otherwise spawn fresh.
      let proc: AcpProcess
      let claimedSpare = false
      const spare = this.warmSpare
      if (spare && !spare.proc.isClosed && spare.cwd === s.cwd && !spare.holder.owner) {
        this.warmSpare = undefined
        spare.holder.owner = s
        proc = spare.proc
        claimedSpare = true
      } else {
        proc = this.createProcFn({
          cwd: s.cwd,
          dangerouslySkipPermissions: false,
          onUpdate: (_sid, update) => this.handleAcpUpdate(s, update),
          onPermissionRequest: (params) => this.handlePermissionRequest(s, params),
          onExit: (code, signal, tail) => {
            this.handleProcExit(s, proc, code, signal, tail, disposedByBridge)
          },
        })
      }
      this.activeProcs.add(proc)

      try {
        if (!claimedSpare) {
          await proc.initialize()
          if (this.disposed) throw new PluginDisposedError()
          // Explicit ACP auth: gemini-api-key when GEMINI_API_KEY is set,
          // else ANTIGRAVITY_ACP_AUTH_METHOD override, else oauth-personal.
          // Benign already-authenticated errors are ignored so cached/env auth works.
          try {
            await proc.authenticate()
          } catch (err) {
            const msg = err instanceof Error ? err.message : String(err)
            if (!/already (authenticated|logged in|signed in)/i.test(msg)) throw err
          }
        }
        if (this.disposed) throw new PluginDisposedError()
        // Unbounded waits above (initialize/authenticate, possibly gated for a
        // long time): re-check before touching storage or the ACP session.
        throwIfStale()
        // Pure Thin ACP by default: zero bridged MCPs, 100% native agy acp.
        // Opt-in Code Mode attaches only the single 'oc' Code Mode MCP server.
        const mcpServers = codeModeEnabled
          ? [
              {
                type: "http" as const,
                name: "oc",
                url: relayUrl,
                headers: this.mcpRelay.authHeaders(),
              },
            ]
          : []

        const storedId = s.acpSessionId ?? (await this.runStorageGet(s.sessionID))
        if (s.startGeneration !== startGeneration) throw new StaleAcpStartError()
        if (this.disposed) throw new PluginDisposedError()
        let res: any
        if (storedId) {
          try {
            res = await proc.resumeSession({
              sessionId: storedId,
              cwd: s.cwd,
              mcpServers,
            })
          } catch {
            res = undefined
          }
          if (s.startGeneration !== startGeneration) throw new StaleAcpStartError()
        }

        if (this.disposed) throw new PluginDisposedError()
        if (!res?.sessionId) {
          freshSession = true
          res = await proc.newSession({
            cwd: s.cwd,
            mcpServers,
          })
        }

        if (this.disposed) throw new PluginDisposedError()
        // Mode flipped (or session reset) while this start was in flight:
        // let the catch block dispose the stale proc without installing it.
        throwIfStale()
        if (s.proc && !s.proc.isClosed && s.acpSessionId) {
          // Lost a concurrent-start race after the entry check (defense in
          // depth — the join loop above normally prevents this): dispose our
          // just-created proc and adopt the winner instead of orphaning it.
          disposedByBridge = true
          this.activeProcs.delete(proc)
          await proc.dispose()
          adoption.winner = { proc: s.proc, acpSessionId: s.acpSessionId }
          return
        }
        s.proc = proc
        s.acpSessionId = String(res.sessionId)
        s.lastError = null
        // A warm-driven fresh *replacement* (a stored id existed but resume
        // failed) owns no server-side history: arm the text-handoff restore
        // consumed by the first real turn. Persisting is deferred until that
        // first turn consumes `pendingFreshHandoff` (see doStream): a bridge
        // restart before turn 1 then still resumes the prior id, fails resume,
        // and re-arms the handoff instead of silently adopting an empty fresh
        // session with the flag lost. Brand-new warms (no prior stored id)
        // persist immediately so LRU eviction keeps a resumable id.
        if (freshSession && opts?.markFreshHandoff && storedId) {
          s.pendingFreshHandoff = true
        } else {
          await this.runStorageSet(s.sessionID, s.acpSessionId)
        }

        if (res?.models?.availableModels?.length || res?.configOptions?.length) {
          const live = normalizeCatalog(buildModelCatalog(res))
          if (live.length > 0 && catalogSignature(live) !== catalogSignature(this.catalog)) {
            const updatedAt = Date.now()
            this.catalog = live
            sharedCatalog = live
            sharedCatalogUpdatedAt = updatedAt
            void this.persistCatalog(live, updatedAt)
            if (!this.disposed) {
              this.onCatalogChanged?.()
            }
          }
        }

        if (s.mode !== "default") {
          await proc.setMode(s.acpSessionId, s.mode).catch(() => {})
          s.lastModeApplied = s.mode
        }
        if (this.disposed) throw new PluginDisposedError()
      } catch (err) {
        disposedByBridge = true
        this.activeProcs.delete(proc)
        this.intentionallyDisposed.add(proc)
        if (s.proc === proc) {
          s.proc = undefined
          s.acpSessionId = undefined
        }
        await proc.dispose()
        throw err
      }
    })()

    s.startLock = startPromise.finally(() => {
      s.startLock = undefined
    })
    try {
      await s.startLock
    } catch (err) {
      // Our own start was superseded mid-flight (mode flip / reset): the stale
      // proc was already disposed without being installed. Foreground turns
      // retry once with the current mode (each retry needs a fresh
      // invalidation, so no loop); background warms stay quiet so a reset
      // really tears down instead of eagerly reinstalling behind it.
      if (err instanceof StaleAcpStartError && !this.disposed) {
        if (opts?.retryOnStale === false) throw err
        return this.ensureAcpSession(s, opts)
      }
      throw err
    }
    // Adopted a concurrent winner that installed while we were in flight
    // (defense-in-depth; the join loop above normally prevents this).
    const winner = adoption.winner
    if (winner) {
      if (this.disposed) throw new PluginDisposedError()
      this.notifyChanged(s.sessionID)
      return { proc: winner.proc, acpSessionId: winner.acpSessionId, freshSession: false }
    }
    if (this.disposed || !s.proc || !s.acpSessionId) {
      throw new PluginDisposedError()
    }
    this.notifyChanged(s.sessionID)
    return { proc: s.proc, acpSessionId: s.acpSessionId, freshSession }
  }

  private handleProcExit(
    s: BridgeSession,
    proc: AcpProcess,
    code: number | null,
    signal: NodeJS.Signals | null,
    tail: string,
    disposedByBridge = false,
  ): void {
    this.activeProcs.delete(proc)
    this.untrackWarmedIdle(s.sessionID)
    if (
      disposedByBridge ||
      this.disposed ||
      this.intentionallyDisposed.has(proc) ||
      (s.proc && s.proc !== proc)
    ) {
      return
    }
    s.lastError = `agy acp exited (code=${code}, signal=${signal})${tail ? `: ${tail.split("\n").at(-1)}` : ""}`
    if (s.activeTurn) {
      this.finishControllerWithError(
        s.activeTurn,
        new Error(s.lastError),
      )
      for (const c of s.activeTurn.pendingMcpCalls.values()) {
        if (!c.resolved) {
          c.resolved = true
          c.resolve({ content: [{ type: "text", text: "ACP process exited" }], isError: true })
        }
      }
      s.activeTurn = undefined
    }
    if (s.proc === proc) s.proc = undefined
    this.notifyChanged(s.sessionID)
  }

  private async handlePermissionRequest(
    s: BridgeSession,
    params: AcpPermissionRequestParams,
  ): Promise<AcpPermissionOutcome> {
    const opts = params.options ?? []
    const allowOpt =
      opts.find(
        (o) =>
          o.kind === "allow_once" ||
          o.kind === "allow_always" ||
          o.optionId.includes("allow"),
      ) ?? opts[0]
    const denyOpt = opts.find(
      (o) => o.kind === "reject_once" || o.kind === "reject_always" || o.optionId.includes("deny"),
    )

    const toolName = String(params.toolCall?.name ?? params.toolCall?.kind ?? "unknown")
    // Always allow the internal MCP relay inside ACP: OpenCode itself enforces real permissions when running the tool!
    if (toolName.startsWith("mcp_oc_")) {
      return allowOpt
        ? { outcome: "selected", optionId: allowOpt.optionId }
        : { outcome: "cancelled" }
    }

    if (s.mode === "auto") {
      return allowOpt
        ? { outcome: "selected", optionId: allowOpt.optionId }
        : { outcome: "cancelled" }
    }

    const title = String(params.toolCall?.title ?? toolName)
    const resources: string[] = []
    if (Array.isArray(params.toolCall?.locations)) {
      for (const loc of params.toolCall.locations) {
        if (loc?.path) resources.push(loc.path)
      }
    }
    if (resources.length === 0) resources.push(title)

    try {
      await Effect.runPromise(
        this.ctx.permission.assert({
          sessionID: s.sessionID as any,
          agent: (s.contextSnapshot?.agent ?? "build") as any,
          action: `antigravity.${params.toolCall?.kind ?? toolName}`,
          resources,
          metadata: {
            tool: toolName,
            title,
            rawInput: params.toolCall?.rawInput,
          },
        }),
      )
      return allowOpt
        ? { outcome: "selected", optionId: allowOpt.optionId }
        : { outcome: "cancelled" }
    } catch {
      return denyOpt
        ? { outcome: "selected", optionId: denyOpt.optionId }
        : { outcome: "cancelled" }
    }
  }

  private async handleIncomingMcpCall(
    sessionID: string,
    mcpToolName: string,
    args: unknown,
    signal: AbortSignal,
  ): Promise<McpCallToolResult> {
    const s = this.sessions.get(sessionID)
    const turn = s?.activeTurn
    if (!s || !turn) {
      return {
        content: [{ type: "text", text: "No active Antigravity turn to execute OpenCode tool." }],
        isError: true,
      }
    }

    const openCodeToolName = s.contextSnapshot?.mcpSafeToOriginal.get(mcpToolName) ?? mcpToolName
    const toolCallId = `oc_relay_${++turn.partSeq}_${Math.random().toString(36).slice(2, 8)}`

    return new Promise<McpCallToolResult>((resolve) => {
      const entry: PendingMcpCall = {
        toolCallId,
        mcpName: mcpToolName,
        openCodeToolName,
        args,
        emittedToOpenCode: false,
        resolved: false,
        resolve,
      }
      turn.pendingMcpCalls.set(toolCallId, entry)

      const onAbort = () => {
        if (!entry.resolved) {
          entry.resolved = true
          turn.pendingMcpCalls.delete(toolCallId)
          resolve({
            content: [{ type: "text", text: "MCP tool call aborted by ACP host." }],
            isError: true,
          })
        }
      }
      signal.addEventListener("abort", onAbort, { once: true })

      this.scheduleFlushToolCalls(s, turn)
    })
  }

  private scheduleFlushToolCalls(s: BridgeSession, turn: ActiveTurn): void {
    if (turn.flushTimer) clearTimeout(turn.flushTimer)
    turn.flushTimer = setTimeout(() => {
      turn.flushTimer = undefined
      this.tryFlushToolCalls(s, turn)
    }, 60)
  }

  private tryFlushToolCalls(s: BridgeSession, turn: ActiveTurn): void {
    if (!turn.controller) return
    // Wait for any concurrent Antigravity native hosted tool calls to settle first
    if (turn.activeNativeTools.size > 0) return

    const unsent = [...turn.pendingMcpCalls.values()].filter((c) => !c.emittedToOpenCode)
    if (unsent.length === 0) return

    this.closeOpenBlocks(turn)
    for (const call of unsent) {
      call.emittedToOpenCode = true
      turn.dispatchedCallIds.add(call.toolCallId)
      const jsonInput = JSON.stringify(call.args ?? {})
      this.emitPart(turn, {
        type: "tool-input-start",
        id: call.toolCallId,
        toolName: call.openCodeToolName,
        providerExecuted: false,
      })
      this.emitPart(turn, {
        type: "tool-input-delta",
        id: call.toolCallId,
        delta: jsonInput,
      })
      this.emitPart(turn, {
        type: "tool-input-end",
        id: call.toolCallId,
      })
      this.emitPart(turn, {
        type: "tool-call",
        toolCallId: call.toolCallId,
        toolName: call.openCodeToolName,
        input: jsonInput,
        providerExecuted: false,
        providerMetadata: { [ACP_PROVIDER_ID]: { acp: true } },
      })
    }

    const finishReason: LanguageModelV3FinishReason = {
      unified: "tool-calls",
      raw: "tool_calls",
    }
    this.emitPart(turn, {
      type: "finish",
      finishReason,
      usage: zeroUsage(),
      providerMetadata: { [ACP_PROVIDER_ID]: { acp: true } },
    })
    const ctrl = this.detachController(turn)
    try {
      ctrl?.close()
    } catch {}
  }

  private detachController(turn: ActiveTurn): ReadableStreamDefaultController<LanguageModelV3StreamPart> | undefined {
    const ctrl = turn.controller
    turn.controller = undefined
    const cleanup = turn.controllerCleanup
    turn.controllerCleanup = undefined
    cleanup?.()
    return ctrl
  }

  private createBoundStream(
    s: BridgeSession,
    turn: ActiveTurn,
    abortSignal: AbortSignal | undefined,
    onStart: (controller: ReadableStreamDefaultController<LanguageModelV3StreamPart>) => void,
  ): ReadableStream<LanguageModelV3StreamPart> {
    let stepController: ReadableStreamDefaultController<LanguageModelV3StreamPart> | undefined
    return new ReadableStream<LanguageModelV3StreamPart>({
      start: (controller) => {
        stepController = controller
        turn.controller = controller
        if (abortSignal) {
          const onAbort = () => {
            if (turn.controller === stepController) {
              void this.handleInterrupt(s.sessionID, "Aborted")
            }
          }
          if (abortSignal.aborted) {
            onAbort()
          } else {
            abortSignal.addEventListener("abort", onAbort, { once: true })
            turn.controllerCleanup = () => abortSignal.removeEventListener("abort", onAbort)
          }
        }
        controller.enqueue({ type: "stream-start", warnings: [] })
        onStart(controller)
      },
      cancel: () => {
        if (turn.controller === stepController) {
          void this.handleInterrupt(s.sessionID, "Stream cancelled")
        }
      },
    })
  }

  private emitPart(turn: ActiveTurn, part: LanguageModelV3StreamPart): void {
    if (turn.controller) {
      try {
        turn.controller.enqueue(part)
        return
      } catch {}
    }
    turn.bufferedParts.push(part)
  }

  private closeOpenBlocks(turn: ActiveTurn): void {
    if (turn.openReasoningId) {
      const id = turn.openReasoningId
      turn.openReasoningId = undefined
      this.emitPart(turn, {
        type: "reasoning-end",
        id,
        providerMetadata: { [ACP_PROVIDER_ID]: { acp: true } },
      })
    }
    if (turn.openTextId) {
      const id = turn.openTextId
      turn.openTextId = undefined
      this.emitPart(turn, {
        type: "text-end",
        id,
        providerMetadata: { [ACP_PROVIDER_ID]: { acp: true } },
      })
    }
  }

  private finishControllerWithError(turn: ActiveTurn, error: Error): void {
    const ctrl = this.detachController(turn)
    if (!ctrl) {
      turn.bufferedParts.push({ type: "error", error })
      return
    }
    try {
      ctrl.enqueue({ type: "error", error })
      ctrl.close()
    } catch {}
  }

  private handleAcpUpdate(s: BridgeSession, update: AcpSessionUpdate): void {
    const kind = update.sessionUpdate
    if (kind === "session_info_update" && typeof update.title === "string" && update.title.trim()) {
      const title = update.title.trim()
      if (s.lastTitleSent !== title) {
        s.lastTitleSent = title
        Effect.runFork(
          this.ctx.session.update({ sessionID: s.sessionID as any, title }).pipe(
            Effect.catch(() => Effect.void),
          ),
        )
      }
      return
    }

    const turn = s.activeTurn
    if (!turn || turn.abortedOnApiError) return

    if (kind === "agent_thought_chunk") {
      const text = (update.content as any)?.text
      if (typeof text !== "string" || text.length === 0) return
      if (turn.openTextId) {
        const tid = turn.openTextId
        turn.openTextId = undefined
        this.emitPart(turn, {
          type: "text-end",
          id: tid,
          providerMetadata: { [ACP_PROVIDER_ID]: { acp: true } },
        })
      }
      if (!turn.openReasoningId) {
        turn.openReasoningId = `r_${++turn.partSeq}`
        this.emitPart(turn, {
          type: "reasoning-start",
          id: turn.openReasoningId,
          providerMetadata: { [ACP_PROVIDER_ID]: { acp: true } },
        })
      }
      this.emitPart(turn, {
        type: "reasoning-delta",
        id: turn.openReasoningId,
        delta: text,
      })
      return
    }

    if (kind === "agent_message_chunk") {
      const text = (update.content as any)?.text
      if (typeof text !== "string" || text.length === 0) return

      if (API_ERROR_ABORT_PATTERN.test(text)) {
        turn.abortedOnApiError = true
        if (isGlobalBreakerError(text)) {
          // Fatal/auth-level: trip the global breaker and tear the session down.
          const rec = this.tripCircuitBreaker(text, "global")
          s.lastError = rec.reason
          void this.runStorageRemove(s.sessionID)

          const proc = s.proc
          const sid = s.acpSessionId
          s.proc = undefined
          s.acpSessionId = undefined
          s.lastModelId = undefined
          s.lastModeApplied = undefined
          s.settlingPrompt = undefined
          if (proc) {
            this.intentionallyDisposed.add(proc)
            this.activeProcs.delete(proc)
            if (!proc.isClosed) {
              if (sid) proc.notifyCancel(sid)
              proc.killImmediate()
            }
          }

          this.finishTurnCleanStop(
            s,
            turn,
            `Antigravity ACP halted immediately after 1st upstream error to prevent retry loops:\n${text.trim()}`,
            "api_error_halted",
          )
        } else {
          // Model-pool quota: trip only this model's bucket and keep the live
          // session intact (proc, acpSessionId, and storage all preserved) so
          // switching to a healthy model in another pool resumes seamlessly.
          const bucket = quotaBucketForModel(s.lastModelId ?? "")
          const rec = this.tripCircuitBreaker(text, bucket)
          s.lastError = rec.reason
          if (s.proc && !s.proc.isClosed && s.acpSessionId) {
            s.proc.notifyCancel(s.acpSessionId)
          }

          this.finishTurnCleanStop(
            s,
            turn,
            `Antigravity ACP halted this turn on a ${bucket} quota error (60s cooldown for ${bucket} models — other model pools are unaffected, switch models to continue immediately):\n${text.trim()}`,
            "quota_cooldown",
          )
        }
        this.notifyChanged(s.sessionID)
        return
      }

      if (turn.openReasoningId) {
        const rid = turn.openReasoningId
        turn.openReasoningId = undefined
        this.emitPart(turn, {
          type: "reasoning-end",
          id: rid,
          providerMetadata: { [ACP_PROVIDER_ID]: { acp: true } },
        })
      }
      if (!turn.openTextId) {
        turn.openTextId = `t_${++turn.partSeq}`
        this.emitPart(turn, {
          type: "text-start",
          id: turn.openTextId,
          providerMetadata: { [ACP_PROVIDER_ID]: { acp: true } },
        })
      }
      this.emitPart(turn, {
        type: "text-delta",
        id: turn.openTextId,
        delta: text,
      })
      return
    }

    if (kind === "tool_call" || kind === "tool_call_update") {
      const callId = String(update.toolCallId ?? "")
      if (!callId) return
      const name = String(update.name ?? update.kind ?? "acp_tool")

      // Relayed OpenCode MCP tools are already emitted natively from handleIncomingMcpCall!
      if (name.startsWith("mcp_oc_") || turn.relayedUpdateIds.has(callId)) {
        turn.relayedUpdateIds.add(callId)
        return
      }

      // Antigravity's native ask_question has no resolver over ACP: agy emits
      // tool_call(in_progress) -> tool_call_update(pending), ends its turn and
      // expects the answer as the next user prompt. As a provider-executed
      // acp_* part it just hangs until aborted. Re-route it to OpenCode's
      // local interactive `question` tool instead: emit a client-executed tool
      // call, finish the step with tool-calls, and let OpenCode collect the
      // answer. buildQuestionAnswerBlocks forwards it on the next turn.
      if (name === "ask_question") {
        if (kind === "tool_call" && !s.pendingQuestion) {
          const translated = translateAskQuestionInput(update.rawInput ?? update.args ?? {})
          if (translated) {
            s.pendingQuestion = { callId, questions: translated.questions }
            turn.relayedUpdateIds.add(callId) // swallow the pending update too
            this.closeOpenBlocks(turn)
            const jsonInput = JSON.stringify(translated.input)
            this.emitPart(turn, {
              type: "tool-input-start",
              id: callId,
              toolName: "question",
              providerExecuted: false,
            })
            this.emitPart(turn, {
              type: "tool-input-delta",
              id: callId,
              delta: jsonInput,
            })
            this.emitPart(turn, {
              type: "tool-input-end",
              id: callId,
            })
            this.emitPart(turn, {
              type: "tool-call",
              toolCallId: callId,
              toolName: "question",
              input: jsonInput,
              providerExecuted: false,
              providerMetadata: { [ACP_PROVIDER_ID]: { acp: true } },
            })
            this.emitPart(turn, {
              type: "finish",
              finishReason: { unified: "tool-calls", raw: "tool_calls" },
              usage: zeroUsage(),
              providerMetadata: { [ACP_PROVIDER_ID]: { acp: true } },
            })
            const ctrl = this.detachController(turn)
            try {
              ctrl?.close()
            } catch {}
            this.notifyChanged(s.sessionID)
            return
          }
        } else if (s.pendingQuestion?.callId === callId) {
          turn.relayedUpdateIds.add(callId)
          return
        }
      }

      const status = String(update.status ?? "")
      if (kind === "tool_call" && !turn.activeNativeTools.has(callId)) {
        this.closeOpenBlocks(turn)
        turn.activeNativeTools.add(callId)
        const rawInput = update.rawInput ?? update.args ?? {}
        const jsonInput = JSON.stringify(rawInput)
        const displayName = `acp_${name}`
        this.emitPart(turn, {
          type: "tool-input-start",
          id: callId,
          toolName: displayName,
          providerExecuted: true,
        })
        this.emitPart(turn, {
          type: "tool-input-delta",
          id: callId,
          delta: jsonInput,
        })
        this.emitPart(turn, {
          type: "tool-input-end",
          id: callId,
        })
        this.emitPart(turn, {
          type: "tool-call",
          toolCallId: callId,
          toolName: displayName,
          input: jsonInput,
          providerExecuted: true,
          providerMetadata: { [ACP_PROVIDER_ID]: { acp: true } },
        })
        return
      }

      if (
        (status === "completed" || status === "failed" || status === "error" || status === "cancelled") &&
        turn.activeNativeTools.has(callId)
      ) {
        turn.activeNativeTools.delete(callId)
        const displayName = `acp_${name}`
        const isError = status !== "completed"
        const resultPayload = extractToolContentText(update)
        this.emitPart(turn, {
          type: "tool-result",
          toolCallId: callId,
          toolName: displayName,
          result: (resultPayload ?? { status }) as any,
          isError,
          providerMetadata: { [ACP_PROVIDER_ID]: { acp: true } },
        })

        // If any relayed OpenCode tool calls were waiting for this native tool to settle, flush them now!
        if (turn.activeNativeTools.size === 0 && turn.pendingMcpCalls.size > 0) {
          this.tryFlushToolCalls(s, turn)
        }
      }
    }
  }

  private finalizeTurnStop(s: BridgeSession, turn: ActiveTurn): void {
    // Emit any remaining buffered native chunks, close open text/reasoning, emit finish:stop
    this.closeOpenBlocks(turn)
    if (turn.activeNativeTools.size > 0) {
      for (const id of turn.activeNativeTools) {
        this.emitPart(turn, {
          type: "tool-result",
          toolCallId: id,
          toolName: "acp_tool",
          result: { status: "completed" },
          isError: false,
          providerMetadata: { [ACP_PROVIDER_ID]: { acp: true } },
        })
      }
      turn.activeNativeTools.clear()
    }

    const stopReason = turn.promptResult?.stopReason
    const unified: LanguageModelV3FinishReason["unified"] =
      stopReason === "max_tokens" ? "length" : "stop"

    this.emitPart(turn, {
      type: "finish",
      finishReason: { unified, raw: stopReason ?? "end_turn" },
      usage: zeroUsage(turn.promptResult?.usage),
      providerMetadata: { [ACP_PROVIDER_ID]: { acp: true } },
    })

    const ctrl = this.detachController(turn)
    try {
      ctrl?.close()
    } catch {}
    if (s.activeTurn === turn) {
      s.activeTurn = undefined
    }
    this.notifyChanged(s.sessionID)
  }

  private createImmediateTextStream(
    text: string,
    rawFinish: string,
  ): ReadableStream<LanguageModelV3StreamPart> {
    // Local harness statuses (cooldown blocks, restarts, missing CLI), not
    // turns that ran through `agy acp`. Tagged with a localStatus marker so
    // hasUnseenNonAcpAssistantHistory skips them: they are neither a completed
    // ACP handoff checkpoint nor a foreign model turn.
    const id = "t_1"
    const localMeta = { providerMetadata: { [ACP_PROVIDER_ID]: { localStatus: true } } } as const
    return new ReadableStream<LanguageModelV3StreamPart>({
      start: (controller) => {
        controller.enqueue({ type: "stream-start", warnings: [] })
        controller.enqueue({
          type: "text-start",
          id,
          ...localMeta,
        })
        controller.enqueue({ type: "text-delta", id, delta: text })
        controller.enqueue({
          type: "text-end",
          id,
          ...localMeta,
        })
        controller.enqueue({
          type: "finish",
          finishReason: { unified: "stop", raw: rawFinish },
          usage: zeroUsage(),
          ...localMeta,
        })
        controller.close()
      },
    })
  }

  createLanguageModel(modelID: string): LanguageModelV3 {
    const bridge = this
    return {
      specificationVersion: "v3",
      provider: ACP_PROVIDER_ID,
      modelId: modelID,
      supportedUrls: {},
      async doGenerate(options: LanguageModelV3CallOptions) {
        const { stream } = await this.doStream(options)
        const reader = stream.getReader()
        const texts: string[] = []
        const reasonings: string[] = []
        let finishReason: LanguageModelV3FinishReason = { unified: "stop", raw: "stop" }
        let usage: LanguageModelV3Usage = zeroUsage()

        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          if (value.type === "text-delta") texts.push(value.delta)
          else if (value.type === "reasoning-delta") reasonings.push(value.delta)
          else if (value.type === "finish") {
            finishReason = value.finishReason
            usage = value.usage
          } else if (value.type === "error") {
            throw value.error
          }
        }
        const content: any[] = []
        if (reasonings.length > 0) content.push({ type: "reasoning", text: reasonings.join("") })
        if (texts.length > 0) content.push({ type: "text", text: texts.join("") })
        return {
          content,
          finishReason,
          usage,
          warnings: [],
        }
      },
      async doStream(options: LanguageModelV3CallOptions) {
        if (bridge.disposed) {
          return {
            stream: bridge.createImmediateTextStream(
              "Antigravity ACP plugin was disabled or reloaded. Send a new message to continue.",
              "plugin_disposed",
            ),
          }
        }
        const hdrs = options.headers ?? {}
        const sessionID = String(hdrs["x-opencode-session-id"] ?? "__default__")
        const variant = String(hdrs["x-opencode-variant"] ?? "") || undefined
        const s = bridge.getOrCreateSession(sessionID)

        // Check if this is an OpenCode tool-result continuation step for an active ACP turn
        const existingTurn = s.activeTurn
        if (existingTurn && existingTurn.dispatchedCallIds.size > 0) {
          const matchedResults = extractToolResultsFromPrompt(
            options.prompt,
            existingTurn.dispatchedCallIds,
          )
          if (matchedResults.size > 0) {
            const stream = bridge.createBoundStream(s, existingTurn, options.abortSignal, (controller) => {
              while (existingTurn.bufferedParts.length > 0) {
                controller.enqueue(existingTurn.bufferedParts.shift()!)
              }

              // Feed OpenCode's tool results back into the waiting HTTP MCP calls
              for (const callId of [...existingTurn.dispatchedCallIds]) {
                existingTurn.dispatchedCallIds.delete(callId)
                const pending = existingTurn.pendingMcpCalls.get(callId)
                if (!pending) continue
                existingTurn.pendingMcpCalls.delete(callId)
                if (!pending.resolved) {
                  pending.resolved = true
                  const res = matchedResults.get(callId) ?? {
                    text: "Tool returned no output",
                    isError: false,
                  }
                  pending.resolve({
                    content: [{ type: "text", text: res.text }],
                    isError: res.isError,
                  })
                }
              }

              if (existingTurn.promptSettled) {
                if (existingTurn.promptError) {
                  bridge.finishControllerWithError(existingTurn, existingTurn.promptError)
                  s.activeTurn = undefined
                } else {
                  bridge.finalizeTurnStop(s, existingTurn)
                }
              } else if (!existingTurn.flushTimer && existingTurn.pendingMcpCalls.size > 0) {
                bridge.scheduleFlushToolCalls(s, existingTurn)
              }
            })
            return { stream }
          }

          // Mismatched turn (user interrupted & sent new prompt): tear down stale activeTurn
          await bridge.handleInterrupt(sessionID, "Superseded by new prompt")
        } else if (existingTurn) {
          await bridge.handleInterrupt(sessionID, "Superseded by new prompt")
        }

        if (isSyntheticRestartAfterError(options.prompt)) {
          return {
            stream: bridge.createImmediateTextStream(
              "Skipped auto-resume after server restart because the previous Antigravity turn halted on an API error. Send a new message when ready.",
              "restart_skipped",
            ),
          }
        }

        await bridge.loadPersistedCooldown()

        // Per-pool breaker: the global cooldown blocks everything, otherwise
        // only the current turn's quota pool is gated — switching pools works
        // with zero lockout.
        const activeCooldown = bridge.cooldownForModel(modelID)
        if (activeCooldown) {
          const secs = Math.ceil((activeCooldown.until - Date.now()) / 1000)
          return {
            stream: bridge.createImmediateTextStream(
              `Antigravity ACP cooldown active (${secs}s remaining) after upstream API error (${activeCooldown.reason}) to protect your account from retry loops.`,
              "circuit_open",
            ),
          }
        }

        // A real turn is starting on this session: it is no longer an idle
        // warmed session, so drop it from the LRU pool (eviction only bounds
        // sessions that never ran a turn).
        bridge.untrackWarmedIdle(sessionID)

        // Foreground reservation: protects `s.proc` from idle-LRU eviction
        // while ensureAcpSession / setModel await. Released once s.activeTurn
        // is installed (which then protects the session) or on early exit.
        s.foregroundCount = (s.foregroundCount ?? 0) + 1
        try {
          // A quota-cancelled prompt resolves asynchronously server-side; don't
        // fire a new prompt on the same live proc until the previous one has
        // settled. `s.settlingPrompt` stays installed as the shared recovery
        // barrier until the cancelled prompt settles OR the wedged proc is
        // replaced (including storage cleanup) — it is never cleared before
        // awaiting, so concurrent follow-ups all join the same barrier instead
        // of racing past it. Bounded wait — a wedged proc is torn down instead.
        // The wait is tied to the exact proc + startGeneration that owned the
        // prompt: only that owner is disposed, and only if still current.
        const settling = s.settlingPrompt
        const settlingProc = s.proc
        const settlingGeneration = s.startGeneration
        if (settling) {
          let settleTimer: ReturnType<typeof setTimeout> | undefined
          const hung = await Promise.race([
            settling.then(
              () => false,
              () => false,
            ),
            new Promise<true>((resolve) => {
              settleTimer = setTimeout(() => resolve(true), SETTLING_PROMPT_TIMEOUT_MS)
            }),
          ])
          if (settleTimer) clearTimeout(settleTimer)
          if (hung) {
            if (
              settlingProc &&
              !settlingProc.isClosed &&
              s.proc === settlingProc &&
              s.startGeneration === settlingGeneration &&
              s.settlingPrompt === settling
            ) {
              bridge.intentionallyDisposed.add(settlingProc)
              bridge.activeProcs.delete(settlingProc)
              s.proc = undefined
              s.acpSessionId = undefined
              // Drop cached model/mode targets so the replacement proc below
              // is (re)configured instead of inheriting the wedged state.
              s.lastModelId = undefined
              s.lastModeApplied = undefined
              s.lastError =
                "Previous Antigravity prompt did not settle in time; started a fresh session."
              // Await the removal so the replacement session below never
              // resumes the wedged id, even if storage is slow.
              await bridge.runStorageRemove(sessionID)
              try {
                await settlingProc.dispose()
              } catch {}
              // Replaced (including storage cleanup): release the barrier.
              if (s.settlingPrompt === settling) s.settlingPrompt = undefined
            }
          }
          // Settled path: the tracked prompt clears `s.settlingPrompt` itself
          // on settle; the barrier stays until it does, so a concurrent
          // follow-up still joins it rather than racing past.
        }

        // The settling prompt may have tripped a pool breaker while we waited:
        // recheck AFTER the barrier (and before ensureAcpSession) so the
        // follow-up is blocked for that pool while unrelated pools proceed.
        {
          const postSettleCooldown = bridge.cooldownForModel(modelID)
          if (postSettleCooldown) {
            const secs = Math.ceil((postSettleCooldown.until - Date.now()) / 1000)
            return {
              stream: bridge.createImmediateTextStream(
                `Antigravity ACP cooldown active (${secs}s remaining) after upstream API error (${postSettleCooldown.reason}) to protect your account from retry loops.`,
                "circuit_open",
              ),
            }
          }
        }

        // T3-Code-style create_with_handoff (v0.2.0): never hit a thread lock.
        // Normal ACP-to-ACP flow (no unseen non-ACP turns) uses the raw thin
        // prompt; switching back after normal-model turns starts a fresh agy
        // session and sends a plain-text-only handoff (no tool/reasoning replay).
        const needsHandoff = hasUnseenNonAcpAssistantHistory(options.prompt)
        if (needsHandoff) {
          await bridge.resetSession(sessionID)
        }

        let ensured: { proc: AcpProcess; acpSessionId: string; freshSession: boolean }
        try {
          ensured = await bridge.ensureAcpSession(s)
        } catch (err) {
          if (err instanceof PluginDisposedError || bridge.disposed) {
            return {
              stream: bridge.createImmediateTextStream(
                "Antigravity ACP plugin was disabled or reloaded. Send a new message to continue.",
                "plugin_disposed",
              ),
            }
          }
          if (err instanceof AgyNotFoundError || (err as NodeJS.ErrnoException)?.code === "ENOENT") {
            const msg = err instanceof Error ? err.message : String(err)
            s.lastError = msg
            bridge.notifyChanged(sessionID)
            return {
              stream: bridge.createImmediateTextStream(msg, "agy_unavailable"),
            }
          }
          throw err
        }
        let { proc, acpSessionId, freshSession } = ensured
        let ensuredGeneration = s.startGeneration

        // Apply model + thinking level target
        const target = resolveAcpModelTarget(bridge.catalog, modelID, variant)
        const applyModelTarget = async (): Promise<void> => {
          if (s.lastModelId !== target.acpModelId) {
            try {
              await proc.setModel(acpSessionId, target.acpModelId)
            } catch {
              // Fallback to configOptions if combined ID isn't accepted on a model update
              await proc.setConfigOption(acpSessionId, "model", target.baseModelId).catch(() => {})
              if (target.thoughtLevel) {
                await proc
                  .setConfigOption(acpSessionId, "thought_level", target.thoughtLevel)
                  .catch(() => {})
              }
            }
            s.lastModelId = target.acpModelId
          }
        }
        await applyModelTarget()

        // A resetSession landing during setModel disposes `proc` and clears the
        // session: revalidate BEFORE installing s.activeTurn or calling
        // proc.prompt. If the session moved on, retry setup with the current
        // session rather than prompting a disposed process. Empty session
        // slots (test-injected ensure stubs that return without installing)
        // are not stale — only a generation bump or a *different* installed
        // proc/id indicates a reset raced us.
        const isStaleInstall = (): boolean =>
          s.startGeneration !== ensuredGeneration ||
          (s.proc !== undefined && s.proc !== proc) ||
          (s.acpSessionId !== undefined && s.acpSessionId !== acpSessionId)
        if (isStaleInstall()) {
          const retry = await bridge.ensureAcpSession(s)
          proc = retry.proc
          acpSessionId = retry.acpSessionId
          freshSession = freshSession || retry.freshSession
          ensuredGeneration = s.startGeneration
          await applyModelTarget()
          if (isStaleInstall()) {
            throw new StaleAcpStartError()
          }
        }

        // T3-Code-style create_with_handoff: plain-text-only history when
        // switching back after normal-model turns, else raw thin prompt.
        // Zero outer system prompt, zero tool schemas, zero tool/reasoning replay.
        // Fallback: if session/resume failed on an existing ACP thread, the fresh
        // session/new owns no history — hand off prior turns as plain text so the
        // earlier conversation isn't silently lost. Brand-new conversations (a
        // single user turn, nothing prior) stay on the raw thin prompt.
        // pendingFreshHandoff covers the warm path: the fresh install happened
        // in an earlier call, so this turn consumes the flag to restore history.
        const nonSystemTurnCount = options.prompt.filter((m) => m.role !== "system").length
        const hadPendingFresh = s.pendingFreshHandoff === true
        const hadFreshStart = freshSession || hadPendingFresh
        s.pendingFreshHandoff = undefined
        if (hadPendingFresh && !freshSession) {
          // Warm-installed fresh session deferred its storage write: persist
          // now that the first real turn consumed the handoff flag.
          await bridge.runStorageSet(sessionID, acpSessionId)
        }
        // Question-answer turn: the previous step re-routed an agy
        // ask_question to OpenCode's local question tool, and this prompt
        // carries its tool result. Forward the answer as plain text instead
        // of the generic thin prompt (which would resend the last user text).
        const pendingQuestion = s.pendingQuestion
        s.pendingQuestion = undefined
        const answerBlocks = pendingQuestion
          ? buildQuestionAnswerBlocks(pendingQuestion, options.prompt)
          : undefined
        const promptBlocks =
          answerBlocks ??
          (needsHandoff || (hadFreshStart && nonSystemTurnCount > 1)
            ? buildTextOnlyHandoffPromptBlocks(options.prompt, { maxChars: 16_000 })
            : buildThinAcpPromptBlocks(options.prompt))

        const turn: ActiveTurn = {
          bufferedParts: [],
          partSeq: 0,
          activeNativeTools: new Set(),
          relayedUpdateIds: new Set(),
          pendingMcpCalls: new Map(),
          dispatchedCallIds: new Set(),
          promptSettled: false,
        }
        s.activeTurn = turn
        bridge.notifyChanged(sessionID)

        const stream = bridge.createBoundStream(s, turn, options.abortSignal, () => {
          const pending = proc.prompt(acpSessionId, promptBlocks)
          // Track the in-flight prompt so a follow-up turn waits for a
          // cancelled prompt to settle server-side before sending a new one
          // on the same live proc. Never rejects; clears itself on settle.
          const tracked: Promise<void> = pending.then(
            () => {
              if (s.settlingPrompt === tracked) s.settlingPrompt = undefined
            },
            () => {
              if (s.settlingPrompt === tracked) s.settlingPrompt = undefined
            },
          )
          s.settlingPrompt = tracked
          pending
            .then((res) => {
              if (turn.abortedOnApiError || bridge.disposed) return
              turn.promptSettled = true
              turn.promptResult = res
              s.lastError = null
              if (turn.flushTimer) {
                clearTimeout(turn.flushTimer)
                turn.flushTimer = undefined
              }
              // Wait if any OpenCode tool call is still mid-roundtrip (shouldn't happen before HTTP reply, unless aborted)
              if (turn.dispatchedCallIds.size === 0 && turn.controller) {
                bridge.finalizeTurnStop(s, turn)
              }
            })
            .catch((err) => {
              if (turn.abortedOnApiError || bridge.disposed) return
              turn.promptSettled = true
              const error = err instanceof Error ? err : new Error(String(err))
              turn.promptError = error

              if (API_ERROR_ABORT_PATTERN.test(error.message)) {
                turn.abortedOnApiError = true
                if (isGlobalBreakerError(error.message)) {
                  // Fatal/auth-level: trip the global breaker and tear the session down.
                  const rec = bridge.tripCircuitBreaker(error.message, "global")
                  s.lastError = rec.reason
                  void bridge.runStorageRemove(sessionID)
                  bridge.intentionallyDisposed.add(proc)
                  bridge.activeProcs.delete(proc)
                  if (s.proc === proc) {
                    s.proc = undefined
                    s.acpSessionId = undefined
                  }
                  s.lastModelId = undefined
                  s.lastModeApplied = undefined
                  s.settlingPrompt = undefined
                  proc.killImmediate()
                  bridge.finishTurnCleanStop(
                    s,
                    turn,
                    `Antigravity ACP halted immediately after upstream API error to prevent retry loops:\n${error.message}`,
                    "api_error_halted",
                  )
                } else {
                  // Model-pool quota: trip only this turn's bucket; keep the
                  // live proc, session id, and storage so a healthy pool
                  // resumes the same session seamlessly.
                  const rec = bridge.tripCircuitBreaker(
                    error.message,
                    quotaBucketForModel(s.lastModelId ?? modelID),
                  )
                  s.lastError = rec.reason
                  if (!proc.isClosed) {
                    proc.notifyCancel(acpSessionId)
                  }
                  bridge.finishTurnCleanStop(
                    s,
                    turn,
                    `Antigravity ACP halted this turn on a quota error (60s cooldown for this model pool — other pools are unaffected, switch models to continue immediately):\n${error.message}`,
                    "quota_cooldown",
                  )
                }
                bridge.notifyChanged(sessionID)
                return
              }

              if (turn.flushTimer) {
                clearTimeout(turn.flushTimer)
                turn.flushTimer = undefined
              }
              for (const c of turn.pendingMcpCalls.values()) {
                if (!c.resolved) {
                  c.resolved = true
                  c.resolve({ content: [{ type: "text", text: error.message }], isError: true })
                }
              }
              turn.pendingMcpCalls.clear()
              turn.dispatchedCallIds.clear()

              s.lastError = error.message
              bridge.finishControllerWithError(turn, error)
              if (s.activeTurn === turn) s.activeTurn = undefined
              bridge.notifyChanged(sessionID)
            })
        })

        return { stream }
        } finally {
          s.foregroundCount = Math.max(0, (s.foregroundCount ?? 1) - 1)
        }
      },
    }
  }
}
