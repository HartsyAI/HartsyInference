# Chat-template fixtures

Each `.jinja` file is the exact, unmodified `tokenizer.chat_template` metadata string from the named local
GGUF checkpoint — byte-for-byte (no added trailing newline, no re-encoding), read with
`HartsyInference.ModelAssets.Gguf.GgufLoader.Load` + `Metadata.GetString("tokenizer.chat_template")`
(metadata only — tensor data is never touched). `ToolCallTemplateDetectionTests`'s guarded
`Category=Integration` test re-reads each source GGUF directly and asserts it still matches the committed
fixture byte-for-byte, so this file and the real checkpoint can't silently drift apart.

| Fixture | Source GGUF (local path) | `general.name` | sha256 (first 16) |
| --- | --- | --- | --- |
| `qwen3-4b.jinja` | `/mnt/model-storage/Models/llm/qwen3/Qwen3-4B-Q4_K_M.gguf` | Qwen3 4B Instruct Awq | `57f1fd00f0013a2b` |
| `qwen2.5-1.5b.jinja` | `/mnt/model-storage/Models/llm/qwen25-1.5b/Qwen2.5-1.5B-Instruct-Q8_0.gguf` | Qwen2.5 1.5B Instruct | `cd8e9439f0570856` |
| `qwen3.5-0.8b.jinja` | `/mnt/model-storage/Models/llm/qwen35/Qwen3.5-0.8B-Q4_K_M.gguf` | Qwen3.5-0.8B | `7f0e529032c25183` |
| `deepseek-r1-distill-qwen-1.5b.jinja` | `/mnt/model-storage/Models/llm/deepseek-r1/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf` | DeepSeek R1 Distill Qwen 1.5B | `b6835114b7303ddd` |
| `glm-4-9b-0414.jinja` | `/mnt/model-storage/Models/llm/glm4/GLM-4-9B-0414-Q4_K_M.gguf` | Glm-4-9B-0414 | `e81f91f841610ef3` |
| `llama-3.2-1b-instruct.jinja` | `/mnt/model-storage/Models/llm/llama32-1b/llama-3.2-1b-instruct-q8_0.gguf` | Llama 3.2 1B Instruct | `5816fce10444e03c` |
| `mistral-7b-instruct-v0.3.jinja` | `/mnt/model-storage/Models/llm/mistral/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf` | Mistral-7B-Instruct-v0.3 | `26a59556925c9873` |
| `gemma-4-e2b-it.jinja` | `/mnt/model-storage/Models/llm/gemma4/gemma-4-E2B-it-Q4_K_M.gguf` | (gemma4 architecture) | `241c50d86bdfe5e4` |

## What each one actually proves

Only three of the eight instruct a format `ToolCallFormats` has a rule table for — the point of committing
all eight is that "references `tools`" is not sufficient by itself, and the failure modes are family-specific,
not hypothetical:

- **Hermes (`true`)**: `qwen3-4b`, `qwen2.5-1.5b` — `{%- if tools %}` ... `<tool_call>\n{"name": ..., "arguments": ...}\n</tool_call>`.
  The literal template source escapes the embedded JSON example as `\"name\"` / `\"arguments\"` (it's a Jinja
  string literal), which is what `TryDetectFromTemplate`'s `\"` → `"` normalization exists for.
- **Gemma (`true`)**: `gemma-4-e2b-it` — `{%- if tools -%}` ... `'<|tool_call>call:' + function['name'] + '{' ...`.
- **Llama-3 (`false`)**: `llama-3.2-1b-instruct` references `tools` (and `tools_in_user_message`) but this
  checkpoint's template never renders `<|python_tag|>` — custom tools get a bare
  `{"name": ..., "parameters": ...}` instruction instead, which isn't one of the four supported envelopes.
  A real Llama model, included specifically to show the detector doesn't assume "family is Llama" implies
  "template uses `<|python_tag|>`".
- **Mistral (`false`)**: `mistral-7b-instruct-v0.3` doesn't reference `tools` at all — this build's template is
  the plain `[INST] ... [/INST]` conversational format, no tool-calling support baked in.
- **DeepSeek (`false`)**: `deepseek-r1-distill-qwen-1.5b` renders a *previous* `message['tool_calls']` turn
  (`<｜tool▁calls▁begin｜>` markers) but never loops over the caller-supplied `tools` list, so there's no
  instruction to detect in the first place.
- **GLM (`false`)**: `glm-4-9b-0414` references `tools` and lists each function via `{%- for tool in tools %}`,
  but only tells the model to "use JSON format for the arguments" in prose — it never renders one of the four
  literal envelope markers. This is a different GLM generation from the `<tool_call>name\n<arg_key>…` XML
  dialect the review named (that's GLM-4.5); `ToolCallTemplateDetectionTests` covers the XML dialect with a
  synthetic template since no local GGUF carries it.
- **Qwen3.5 (`false`)**: `qwen3.5-0.8b` references `tools` and opens with the same `<tool_call>` tag as the
  Hermes pair, but instructs the XML `<function=name><parameter=...>` form (Qwen3-Coder style) — it contains
  neither `"name"` nor `"arguments"` as literal substrings anywhere in the template.
