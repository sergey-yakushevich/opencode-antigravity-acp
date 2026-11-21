import { Plugin } from "@opencode/plugin/effect"
import { Effect, Stream } from "effect"
import type { AcpDiscoveredModel } from "./acp-client.js"
import {
  ACP_PROVIDER_ID,
  sanitizeForAcpModel,
  sanitizeForNonAcpModel,
} from "./history-sanitizer.js"
import { AcpRpc } from "./rpc.js"
import { SessionBridgeManager } from "./session-bridge.js"

function buildModelEntries(models: readonly AcpDiscoveredModel[]): any[] {
  return models.map((m) => ({
    id: m.id,
    modelID: m.id,
    providerID: ACP_PROVIDER_ID,
    type: "chat",
    family: m.family ?? m.id,
    name: m.name,
    package: "aisdk:antigravity-acp",
    capabilities: {
      tools: true,
      input: ["text", "image"],
      output: ["text"],
    },
    variants: m.variants.map((v) => ({ id: v.id })),
    time: { released: 0 },
    cost: [],
    status: "active",
    enabled: true,
    limit: {
      context: 1048576,
      output: 65536,
    },
  }))
}

function summarizeTitleFromMessages(messages: readonly any[]): string | undefined {
  for (const msg of messages) {
    if (msg?.role !== "user" || !Array.isArray(msg.content)) continue
    const text = msg.content
      .filter((p: any) => p?.type === "text" && typeof p.text === "string")
      .map((p: any) => p.text)
      .join(" ")
      .replace(/\s+/g, " ")
      .trim()
    if (!text) continue
    return text.length > 60 ? `${text.slice(0, 57)}...` : text
  }
  return undefined
}

export default Plugin.define({
  id: ACP_PROVIDER_ID,
  effect: (ctx) =>
    Effect.gen(function* () {
      const bridge = new SessionBridgeManager(ctx, ctx.location.directory, undefined, {
        onCatalogChanged: () => {
          Effect.runFork(ctx.provider.reload().pipe(Effect.catch(() => Effect.void)))
        },
      })

      yield* Effect.addFinalizer(() => Effect.promise(() => bridge.disposeAll()))

      const cacheStatus = yield* Effect.promise(() => bridge.loadCachedModels())

      yield* ctx.provider.transform((editor) => {
        if (editor.get(ACP_PROVIDER_ID as any)) {
          editor.remove(ACP_PROVIDER_ID as any)
        }
        editor.add({
          info: {
            id: ACP_PROVIDER_ID as any,
            name: "Antigravity (ACP)",
            activation: "enabled",
            package: "aisdk:antigravity-acp",
          } as any,
          models: buildModelEntries(bridge.getModels()) as any,
        })
      })

      // Both sdk and language AISDK hooks are required when using "aisdk:antigravity-acp"
      yield* ctx.aisdk.hook(
        "sdk",
        (input) =>
          Effect.sync(() => {
            input.sdk = {
              languageModel: (id: string) => bridge.createLanguageModel(id),
            }
          }),
        { providerID: ACP_PROVIDER_ID as any },
      )

      yield* ctx.aisdk.hook(
        "language",
        (input) =>
          Effect.sync(() => {
            const id = String(input.model.modelID ?? input.model.id)
            input.language = bridge.createLanguageModel(id)
          }),
        { providerID: ACP_PROVIDER_ID as any },
      )

      // Strict Thin ACP Mode: zero outer-harness prompt injection.
      // Snapshot the live tool catalog for the optional Code Mode MCP relay,
      // then clear outer system instructions so OpenCode does not inject
      // anything into the ACP model call.
      //
      // NOTE: input.tools is intentionally NOT cleared. The bridge never
      // forwards options.tools to agy (prompts are thin text-only), so no
      // tool schemas reach the ACP side either way — but OpenCode resolves
      // client-executed tool calls (MCP relay callbacks, the re-routed local
      // `question` tool) against the request's tool definitions and fails
      // them with "Tool is not available for this request" when the map is
      // empty. Keeping the map local-only preserves thin mode and execution.
      yield* ctx.session.hook("context", (input) =>
        Effect.sync(() => {
          if (input.model.providerID === ACP_PROVIDER_ID) {
            sanitizeForAcpModel(input.messages as any[])
            bridge.updateContextSnapshot(input.sessionID, {
              systemText: "",
              agent: input.agent,
              tools: input.tools,
            })
            input.system = []
          } else {
            sanitizeForNonAcpModel(input.messages as any[])
          }
        }),
      )

      // Pass exact sessionID, request kind, and selected variant into LanguageModelV3.doStream via headers
      yield* ctx.session.hook(
        "model.request",
        (input) =>
          Effect.sync(() => {
            input.headers["x-opencode-session-id"] = input.sessionID
            input.headers["x-opencode-request-kind"] = input.kind
            input.headers["x-opencode-agent"] = input.agent
            input.headers["x-opencode-variant"] = input.model.variant ?? ""
          }),
        { providerID: ACP_PROVIDER_ID as any },
      )

      // Avoid burning an ACP session/turn just to auto-title new threads
      yield* ctx.session.hook(
        "title",
        (input) =>
          Effect.sync(() => {
            if (input.result) return
            const derived = summarizeTitleFromMessages(input.messages as any[])
            if (derived) input.result = derived
          }),
        { providerID: ACP_PROVIDER_ID as any },
      )

      yield* ctx.rpc.register(AcpRpc, {
        getState: (input) => {
          const { sessionID } = input as { sessionID: string }
          return Effect.sync(() => bridge.getState(sessionID))
        },
        setMode: (input) => {
          const { sessionID, mode } = input as {
            sessionID: string
            mode: "auto" | "accept-edits" | "default" | "plan"
          }
          return Effect.promise(() => bridge.setMode(sessionID, mode))
        },
        setCodeMode: (input) => {
          const { sessionID, codeModeEnabled } = input as {
            sessionID: string
            codeModeEnabled: boolean
          }
          return Effect.promise(() => bridge.setCodeMode(sessionID, codeModeEnabled))
        },
        resetSession: (input) => {
          const { sessionID } = input as { sessionID: string }
          return Effect.promise(() => bridge.resetSession(sessionID))
        },
        warmSession: (input) => {
          const { sessionID } = input as { sessionID?: string }
          return Effect.promise(() => bridge.warmSession(sessionID))
        },
        reloadModels: () =>
          Effect.gen(function* () {
            const { models } = yield* Effect.promise(() => bridge.refreshModels({ force: true }))
            yield* ctx.provider.reload()
            return { modelCount: models.length }
          }).pipe(Effect.orDie),
      })

      if (cacheStatus.stale) {
        yield* Effect.forkScoped(
          Effect.promise(() => bridge.refreshModelsInBackground()).pipe(
            Effect.catch(() => Effect.void),
          ),
        )
      }

      // Ensure Stop / Esc / failures / session deletion interrupt or dispose `agy acp` cleanly
      yield* Effect.forkScoped(
        ctx.event
          .subscribe({
            types: ["session.execution.interrupted", "session.execution.failed", "session.deleted"],
          })
          .pipe(
            Stream.runForEach((evt: any) => {
              const sid = evt?.data?.sessionID
              if (typeof sid !== "string" || sid.length === 0 || !bridge.hasSession(sid)) {
                return Effect.void
              }
              return Effect.promise(async () => {
                if (evt.type === "session.deleted") {
                  await bridge.resetSession(sid)
                  return
                }
                await bridge.handleInterrupt(
                  sid,
                  evt.type === "session.execution.interrupted"
                    ? "Interrupted by user"
                    : "Session execution failed",
                )
              })
            }),
            Effect.catch(() => Effect.void),
          ),
      )

      yield* Effect.logInfo("Antigravity ACP plugin loaded", {
        directory: ctx.location.directory,
        models: bridge.getModels().length,
      })
    }).pipe(Effect.orDie),
})
