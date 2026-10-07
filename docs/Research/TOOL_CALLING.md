# LLM tool calling (`HartsyInference.Tools`)

Source snapshot 2026-09-30: formats read from the local GGUF checkpoints' own `tokenizer.chat_template` and vocab
(Qwen3-4B, Llama-3.2-1B-Instruct, Mistral-7B-Instruct-v0.3, gemma-4-E2B-it). Engine verification status lives in
[MODEL_STATUS_LLM.md](../Checklists/MODEL_STATUS_LLM.md#tool-calling).

## Split

| Layer | Holds |
|---|---|
| Engine (`HartsyInference.Engine`) | `TextRequest.Tools` → chat template (`tools` in Jinja, Qwen2.5-shaped block in the ChatML fallback) and the `<tool_call>` sentinel grammar; `ITextStreamFilter` seam (`OnDelta`/`OnEnd` → `TextFilterResult{ForwardText, ToolCall, Stop}`) created per request by `EngineOptions.TextStreamFilterFactory`; `TextChunkKind.NativeToolCall` emission, `StopReason.ToolCall`, `TextResult.ToolCall`; OpenAI-compat mapping. No format knowledge. |
| Tools (`HartsyInference.Tools`, opt-in, references Engine only, not in the meta package) | `ToolCallFormat` rules, `ToolCallParser`, `ToolCallStreamFilter`, `ToolCalling.Install`, `ToolRegistry`/`IToolHandler`, `ToolSchema.FromDelegate`, `ToolLoop`. |

Without the package installed a request with `Tools` still renders them into the prompt and arms the grammar, but no
call is parsed: the model's call text streams as plain content.

## Install

```csharp
EngineOptions options = new();
ToolCalling.Install(options, ToolCallFormats.Detect("Qwen3-4B-Q4_K_M.gguf"));   // format is per engine; null = Hermes
using InferenceEngine engine = new("cuda", options);

ToolRegistry tools = new ToolRegistry()
    .Add("hang_up", () => "call ended", "Ends the current phone call immediately.")
    .Add("weather", ([Description("City name")] string city, int days = 3) => Forecast(city, days));

TextRequest request = new() { Messages = [user("please hang up now")], EnableThinking = false };
await foreach (TextChunk chunk in ToolLoop.RunAsync(engine.Text, spec, request, tools, maxRounds: 4, ct)) { … }
```

`ToolRegistry` is thread-safe (concurrent `Add`/`Remove`/lookup; `Definitions` is an immutable insertion-ordered
snapshot per read), so a host can build or edit a registry per call. `Remove(name)` unregisters a tool. Handlers have
no timeout by default; set `ToolRegistry.DefaultTimeout`, or pass a `TimeSpan` to an `Add` overload for one tool, and a
handler that runs past it has its token cancelled and the call returns `{"error": "... timed out ..."}` (a handler that
ignores its token is abandoned, not awaited). Cancelling the turn's token still throws.

`Install` sets the factory to return a `ToolCallStreamFilter` only when `request.Tools` is non-empty; every other
request keeps the untouched text path. `StopAfterFirstCall` (default true) ends generation at the first completed
call; false lets the model emit several calls, forwarding the text between them.

## Formats

| `ToolCallFormat` | Marker form (when the tokenizer emits the marker as text) | Bare form (what a GGUF delivers) | Arguments key |
|---|---|---|---|
| `Hermes` (Qwen, GLM, DeepSeek, Hermes fine-tunes; default) | `<tool_call>{"name": …, "arguments": {…}}</tool_call>`, closing tag optional | `{` at line start, or `{"name"` anywhere | `arguments`, then `parameters` |
| `Llama3` | `<|python_tag|>{"name": …, "parameters": {…}}` | `{` at line start, `{"name"` anywhere | `parameters`, then `arguments` |
| `Gemma` (Gemma 4) | `<|tool_call>call:name{key:value,text:<|"|>string<|"|>}<tool_call|>` | `call:` anywhere followed by `name{…}` | block converted to a JSON object |
| `Mistral` | `[TOOL_CALLS][{"name": …, "arguments": {…}}, …]`, a single object, or `name{…}` | `[` at line start, `{` at line start, `{"name"` anywhere, `name{` at line start | `arguments`, then `parameters` |

Rules that hold for every format:

- A span ends when its JSON (or Gemma block) value balances; a Hermes/Gemma closing marker after it, and the
  whitespace around the span, are consumed. Multiple spans per turn are allowed and numbered `call_0`, `call_1`, …
- The bare forms are *strict*: a bare JSON span is released at its first key unless that key is `"name"` (a code
  block or a JSON answer with tools on streams normally instead of stalling until its braces balance), and when the
  parser knows the offered tool names (`ToolCalling.Install` passes `request.Tools`; `ToolCallParser`/
  `ToolCallStreamFilter` take `knownTools`) a bare span naming anything else is text; the bare `name{` forms
  (Mistral at line start, Gemma after `call:`) are released at the brace unless the name is a complete offered one,
  so a word that only prefixes a tool name does not hold the text after it. The tagged forms
  (`<tool_call>`, `<|python_tag|>`, `[TOOL_CALLS]`, `<|tool_call>`) stay permissive so a mistyped tool name reaches
  the host as a call it can answer with an error result. Without a known-name list every bare `{"name": …}` object
  is a call.
- Anything that does not resolve into a call is forwarded as plain text: invalid JSON, an object without a string
  `name`, a closing marker before the value balanced, a span longer than `ToolCallParser.DefaultMaxSpanChars`
  (64 KiB), an unterminated span at `Flush`. The parser never throws on model output.
- `"arguments"` given as a JSON string is passed through as the JSON text; OpenAI's `{"function": {…}}` nesting is
  unwrapped; a missing arguments object becomes `{}`.

### Why the bare forms are primary

In every local GGUF the markers are CONTROL (type 3) or USER_DEFINED (type 4) tokens: Qwen3-4B `<tool_call>` = 151657
type 4, `</tool_call>` = 151658 type 4; Llama-3.2 `<|python_tag|>` = 128010 type 3; Mistral-7B-v0.3 `[TOOL_CALLS]`
= 5 type 3; gemma-4-E2B `<|tool_call>` = 48, `<tool_call|>` = 49, `<|"|>` = 52, all type 4. `GgufTokenizer` marks
both types special and `PassthroughOutputParser` decodes with `includeSpecial:false`, so the filter sees only the
payload between the markers. The text-marker rules cover tokenizers that emit the tags as ordinary pieces.

Gemma consequence: with `<|"|>` dropped, a string value is a bare word up to the next `,`/`}`/`]`, so string values
containing those characters cannot be recovered (`{city:Paris, FR}` reads as `city:"Paris"` plus a malformed rest).
Keys are unquoted in the DSL (`escape_keys=False` in the template's own `format_argument`).

Same token typing, different symptom: `JsonVocabText` builds its table with `Decode([id])`, which skips special ids,
so the `<tool_call>` sentinel never appears in `SentinelJsonGrammarStep`'s tail and the grammar does not activate on
these checkpoints (found while building this package; the engine side is unchanged here).

## Seam limits

`TextFilterResult` carries one `ToolCall`. When one delta closes several calls (a Mistral array), the filter emits the
first and queues the rest for the following deltas and `OnEnd`, and with `StopAfterFirstCall` it requests the stop
only once the queue has drained (one extra decode step per queued call), so every completed call reaches the host;
`ToolCallStreamFilter.Calls` lists all of them.

## Loop message shapes

Per round `ToolLoop` sends `request with { Messages = conversation, Tools = request.Tools ?? registry.Definitions }`.
After a round that ends in `StopReason.ToolCall` with calls `c0..cn`:

```
assistant: { Content = <round's visible text>, ToolCalls = [c0..cn] }        // template renders tool_calls[].function
tool:      { Content = <result of c0>, ToolCallId = c0.Id, Name = c0.Name }   // one per call, in order
…
```

Jinja receives `tool_calls[i] = {id, type:"function", function:{name, arguments}}` and `tool_call_id`/`name` on the
tool turn; Qwen's template folds consecutive tool turns into one `<tool_response>` user turn.

Chunk stream: each round's `Chunk`, `NativeToolCall`, `Reasoning`, `Status`, `ToolCallDelta`, `ToolCallAbort` and
`Usage` chunks pass through as they arrive; each round's own `Result` and `ToolCall` stop are suppressed; the loop ends
with one `Result` (visible text of all rounds) and one `StopReason`. Tool results are `TextChunkKind.Status` chunks
(no Engine kind fits and this package adds none): `Text = "tool_result:" + result`, `Status.Phase = "tool_result"`,
`ToolCall` = the call, `ToolCallIndex` = running index. When the round limit is hit while the model still asks for a
tool, the loop yields `Status` `tool_loop:max_rounds=N` (phase `tool_loop`), the result and `StopReason.ToolCall`;
that last call is not dispatched. `Error`/`Cancelled` stops are relayed and end the loop.

Dispatch errors are results, not exceptions: an unknown tool or a throwing handler yields
`{"error": "…"}` (exception type and message) for the model to read; cancellation propagates. That text is meant
for the model and the host; a host relaying tool results to a remote end user should rewrite it. An `Error` or
`Cancelled` stop from the service ends the loop without a final `Result` chunk.

## Open

- `ForceToolId` (OpenAI `tool_choice` = named function) is carried but not enforced anywhere.
- Llama 3.2's pythonic `[f(a=1), g()]` call syntax and Gemma 3's ```` ```tool_code ```` prompt-only convention are
  not parsed.
- Surfacing marker tokens to the filter (or a tokenizer-aware sentinel table for the grammar) is an Engine/LLM
  decision, not taken here.
