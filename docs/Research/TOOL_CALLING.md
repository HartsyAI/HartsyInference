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
ignores its token is abandoned, not awaited). With a timeout set the handler starts on the thread pool, not on the
caller's synchronization context. Cancelling the turn's token still throws.

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

Chunk stream: each round's `Chunk`, `NativeToolCall`, `Reasoning`, `Status`, `Usage` and other pass-through chunks stream
as they arrive; each round's own `Result` and `ToolCall` stop are suppressed; the run ends with one `Result` (the visible
text of all rounds) and one `StopReason`. A dispatched call streams as `TextChunkKind.ToolResult`: `Text` is the result,
`ToolCall` the call, `ToolCallIndex` its position in the run. When the round limit is hit while the model still asks for
a tool, the run yields a `Status` chunk `tool_loop:max_rounds=N` (phase `tool_loop`), the result and
`StopReason.ToolCall`; that last call is not dispatched. `Error`/`Cancelled` stops are relayed and end the run.

`ToolLoop.Create` takes the engine's text service or any model stream (`Func<TextRequest, CancellationToken,
IAsyncEnumerable<TextChunk>>`), an `IToolDispatcher` (`ToolRegistry` implements it; a host can supply its own executor)
and `ToolLoopOptions`. The run is single-use and exposes `Conversation`, `ToolResults`, `Rounds`, `Stop` and
`VisibleText` once it completes. `ToolLoopOptions.OnBeforeToolCall` allows or denies each call; a denial is not
dispatched and its result is what the model reads. Calls dispatch on any non-error stop, so a plain `Stop` with a call
still runs it.

Dispatch errors are results, not exceptions: an unknown tool or a throwing handler yields
`{"error": "…"}` (exception type and message) for the model to read; cancellation propagates. That text is meant
for the model and the host; a host relaying tool results to a remote end user should rewrite it. An `Error` or
`Cancelled` stop from the service ends the loop without a final `Result` chunk.

## Status (alpha.335 to alpha.341)

- Per-request format: `ToolCalling.Install` resolves the format from the model's chat template first, then its
  architecture or path, then Hermes. A structured parser (DeepSeek-V4.1) gets no text filter.
- Control-token markers reach the parser: a filter lists its markers (`ITextStreamFilter.MarkerLiterals`) and the engine
  decodes them as text. Message content is escaped against the control literals, so user or tool text cannot open a turn
  or emit a marker token. `<think>`/`</think>` stream as text.
- Dialects: Hermes JSON (including the name-on-one-line form GLM-4-0414 uses), Llama 3 `<|python_tag|>`, Gemma 4,
  Mistral `[TOOL_CALLS]`, Qwen3.5 and Qwen3-Coder `<function=…><parameter=…>` (`QwenXml`), GLM-4.5 `<arg_key>` pairs
  (`GlmXml`), and DeepSeek-R1 fenced JSON blocks (`DeepSeekR1`). Markup values are typed by the offered tool's schema.
- A template with no `tools` slot gets the Hermes tool prompt written into the conversation, and the template gets no
  tools. `ForceToolId` must name an offered tool (refused before model work), is stated in the system prompt, and
  restricts the parser to that tool. It is not grammar-forced: a model that ignores the instruction can answer in prose.
- The sentinel grammar (`<tool_call>` JSON forcing) is no longer armed for tool requests. It never fired on GGUF text
  and it disabled graph and speculative decode for every tool turn.

Real-weight results on this machine (CPU; `ToolLoopRealWeightTheoryTests` and `ToolCallCpuProbeTests`):

| Checkpoint | Dialect | Result |
|---|---|---|
| Qwen3-0.6B Q4_K_M | Hermes | `get_time` call, second round, passes |
| Gemma-3-1B Q4_K_M | Hermes, injected prompt | `get_time` call, second round, passes |
| Llama-3.2-1B Q8_0 | Llama 3 | `get_time` call, second round, passes |
| Qwen2.5-0.5B Q4_K_M | Hermes | `hang_up` call passes; `get_time` gets `<tool_call>` then the turn ends, so the matrix case fails (a model limit on this prompt, kept visible) |
| Qwen3.5-0.8B Q4_K_M | QwenXml | not run on CPU: its Q5_K dense weights have no CPU matmul path |
| Phi-3.5-mini, Mistral-7B, GLM-4-9B | Hermes | not run on CPU: the CPU path widens weights to F32 (about 15 GB or more) and the OOM killer ends the test host |
| DeepSeek-R1-Distill-Qwen-1.5B | DeepSeekR1 | not on this machine |
| GLM-4.5 | GlmXml | not on this machine; the dialect is covered by unit tests only |

Run the CUDA rows with `TOOLCALL_PROBE_DEVICE=cuda` and the checkpoint's `*_GGUF_PATH` variable.

## Open

- Llama 3.2's pythonic `[f(a=1), g()]` call syntax and Gemma 3's ```` ```tool_code ```` prompt-only convention are
  not parsed.
- `ForceToolId` is an instruction plus a parser restriction, not a grammar. A model can still answer in prose.
- Small models echo a tool's schema into `arguments` instead of an empty object (seen on Gemma-3-1B); the parser
  faithfully reports what the model wrote.
- The CPU path cannot run Q5_K dense weights or 3.8B-and-larger checkpoints without an F32 widen; those need CUDA or a
  quantized CPU kernel.
