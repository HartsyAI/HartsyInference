# Structured Prompt Builder — Research & Design Notes

> **Status:** Design complete (no code yet) | **Last Updated:** 2026-06-07 | **Needed Before:** `Ideogram4Pipeline` usability; reused by future regional-prompting models
>
> **Motivation:** Ideogram 4 (and a growing set of layout-controllable models) is trained on **structured captions** — a scene summary, a style block, and per-object descriptions with bounding boxes and hex color palettes — not free text. Community feedback (ComfyUI tutorials, forum threads) is that "you have to LLM-prompt each region," and people hand-roll JSON per model and get it subtly wrong (key order, hex casing, coordinate convention). This document designs **one model-agnostic structured-prompt data model** with **per-model serializer dialects**, so a user (or our SwarmUI extension) builds the prompt once and we emit the exact format each model wants — plus an optional pluggable LLM "magic prompt" expander.
>
> **Sources:**
> - `ideogram-oss/ideogram4` `docs/prompting.md` + `src/ideogram4/magic_prompt.py` + `magic_prompt_system_prompts/`
> - [IDEOGRAM4_ARCHITECTURE.md](IDEOGRAM4_ARCHITECTURE.md) (the consumer)
> - Existing regional-prompting conventions (attention-coupled region masks) used by SD/Flux community tooling — relevant to the non-JSON dialects.

## Summary

The builder is a small, dependency-free library: a **`StructuredPrompt` data model** (scene description, style block, ordered list of elements each with optional bounding box + color palette), a set of **`IPromptDialect` serializers** (one per target model/format), and an optional **`IMagicPromptExpander`** hook (plug an LLM via the native `HartsyInference.LLM` package, the Claude API, or a local Gemma/Qwen, to turn a plain string into a `StructuredPrompt`). The first dialect is **`Ideogram4Dialect`**, which emits the exact JSON Ideogram 4 was trained on (correct key ordering, compact separators, uppercase `#RRGGBB`, `0–1000` normalized `[y_min,x_min,y_max,x_max]` boxes). Later dialects cover **regional-attention prompting** (for models that take per-region prompt + mask instead of JSON) and **plain natural language** (flatten the structure to prose).

This lives in **`src/HartsyInference.Diffusion/Prompting/`** (image-generation specific; not Core). It is **pure data + serialization** — no backend, no tensors, no hot path — so it has no performance constraints and no package-boundary concerns beyond Diffusion. The LLM expander is an interface only; concrete expanders live behind it so the core builder never depends on any LLM SDK.

## The universal data model

```csharp
namespace HartsyInference.Diffusion.Prompting;

/// <summary>Model-agnostic structured prompt: a scene, a style, and spatially-placed elements.</summary>
public record StructuredPrompt
{
    /// <summary>One- or two-sentence summary of the whole image (Ideogram: high_level_description).</summary>
    public string? Summary { get; init; }
    public StyleBlock? Style { get; init; }
    /// <summary>Background / environment description (required by Ideogram's compositional_deconstruction).</summary>
    public string? Background { get; init; }
    public IReadOnlyList<PromptElement> Elements { get; init; } = [];
    /// <summary>Up to 16 overall-palette hex colors (#RRGGBB).</summary>
    public IReadOnlyList<string> ColorPalette { get; init; } = [];
}

public record StyleBlock
{
    public string? Aesthetics { get; init; }
    public string? Lighting { get; init; }
    public string? Medium { get; init; }
    /// <summary>Photographic descriptor; mutually exclusive with ArtStyle.</summary>
    public string? Photo { get; init; }
    /// <summary>Illustration/painting/3D descriptor; mutually exclusive with Photo.</summary>
    public string? ArtStyle { get; init; }
    public IReadOnlyList<string> ColorPalette { get; init; } = [];
}

/// <summary>Base for a placed element. Bbox is normalized 0–1000 [y_min, x_min, y_max, x_max].</summary>
public abstract record PromptElement
{
    public BoundingBox? Bbox { get; init; }
    public string Description { get; init; } = "";
    /// <summary>Up to 5 per-element hex colors.</summary>
    public IReadOnlyList<string> ColorPalette { get; init; } = [];
}

public record ObjectElement : PromptElement;

public record TextElement : PromptElement
{
    /// <summary>Literal text to render in the image.</summary>
    public required string Text { get; init; }
}

/// <summary>Normalized 0–1000 box. Order matches Ideogram: [y_min, x_min, y_max, x_max].</summary>
public readonly record struct BoundingBox(int YMin, int XMin, int YMax, int XMax)
{
    public void Validate() { /* 0 ≤ min < max ≤ 1000 on both axes; throw HartsyInferenceException otherwise */ }
}
```

A fluent `StructuredPromptBuilder` is convenient for callers and the SwarmUI extension:

```csharp
StructuredPrompt prompt = new StructuredPromptBuilder()
    .Summary("A neon-lit ramen shop at night in the rain")
    .Style(s => s.ArtStyle("anime").Lighting("moody neon").Medium("digital illustration"))
    .Background("wet city street, glowing signage reflections")
    .AddObject("a steaming bowl of ramen on the counter",
               bbox: new BoundingBox(600, 350, 950, 700), palette: ["#FF6B35", "#F7C59F"])
    .AddText("らーめん", bbox: new BoundingBox(50, 100, 200, 900), desc: "glowing red shop sign")
    .Palette("#FF6B35", "#004E89", "#1A659E", "#2B2D42")
    .Build();
```

## Dialect interface

```csharp
/// <summary>Serializes a StructuredPrompt into a specific model's expected conditioning text/format.</summary>
public interface IPromptDialect
{
    string Name { get; }
    /// <summary>Render to the model's conditioning string (JSON for Ideogram, prose for NL models, etc.).</summary>
    string Serialize(StructuredPrompt prompt);
    /// <summary>Optional per-region outputs for models that condition on (prompt, mask) regions.</summary>
    IReadOnlyList<RegionPrompt> SerializeRegions(StructuredPrompt prompt, int width, int height) => [];
}

/// <summary>A per-region prompt + pixel-space mask, for attention-coupled regional prompting.</summary>
public readonly record struct RegionPrompt(string Prompt, RectMask Mask, float Weight);
```

## Ideogram-4 dialect (the first, exact)

`Ideogram4Dialect.Serialize` produces the JSON Ideogram 4 was trained on. **The training distribution is sensitive to these details — get them exactly right** (from `docs/prompting.md`):

- **Top-level keys, in this order:** `high_level_description` (optional, strongly recommended), `style_description` (optional object), `compositional_deconstruction` (required object).
- **`style_description`** required sub-fields when present: `aesthetics`, `lighting`, `medium`, and **exactly one** of `photo` or `art_style`; optional `color_palette`.
- **`compositional_deconstruction`** = `{ "background": <string, required>, "elements": [ ... ] }`.
- **Element key order is fixed per type:**
  - object → `type:"obj"`, `bbox` (optional), `desc`, `color_palette` (optional)
  - text → `type:"text"`, `bbox` (optional), `text`, `desc`, `color_palette` (optional)
- **`bbox`** = `[y_min, x_min, y_max, x_max]`, integers in **normalized 0–1000** coordinates.
- **Color palettes**: uppercase `#RRGGBB` only (no shorthand, no lowercase). ≤16 overall, ≤5 per element. Example: `["#FF6B35","#F7C59F","#004E89","#1A659E","#2B2D42"]`.
- **Serialization flags (must match):** compact separators `(",", ":")` (no spaces), `ensure_ascii=False` (keep `らーめん`/accents literal — needed for multilingual text rendering).
- **Key order matters:** "trained on JSON with a consistent key order; maintaining it improves generation quality." Deviating "means sampling outside the training distribution."

**C# implementation note:** use **source-generated `System.Text.Json`** (`[JsonSerializable]`, no reflection — per CODE_STYLE) with a `JsonSerializerOptions { WriteIndented = false, Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping }` and an explicit property order. Because key order + the `obj`/`text` discriminator + uppercase-hex normalization are load-bearing, the cleanest path is a **hand-written serializer that writes via `Utf8JsonWriter` in the exact field order** rather than relying on attribute ordering (which is fragile). Normalize hex to uppercase and validate `#RRGGBB` at serialize time; throw `HartsyInferenceException` on a bad color or out-of-range bbox.

The dialect should also expose a **validator** (`Ideogram4Dialect.Validate(prompt)`) surfacing the common mistakes as actionable errors: lowercase/short hex, >16 or >5 palette entries, both `photo` and `art_style` set, missing `background`, bbox outside 0–1000 or `min ≥ max`.

## Other dialects (later, for "support multiple models")

- **`NaturalLanguageDialect`** — flatten a `StructuredPrompt` to prose for models that take plain text (SD1.5/SDXL/Flux/Lens). e.g. `"{Summary}. {Style.Medium}, {Style.Lighting}. Background: {Background}. {element descs joined}."` Lets a user build one structured prompt and target any model. Drops bbox/region info (or appends "X on the left, Y on the right" positional hints).
- **`RegionalAttentionDialect`** — for models that support regional prompting via attention coupling (per-region prompt + mask). `SerializeRegions` converts each element's `Bbox` (0–1000) into a pixel-space `RectMask` at the target resolution and emits a `RegionPrompt`. The denoise loop then biases cross-attention per region. **This is the bridge to "regional prompting for models that aren't Ideogram"** the user asked about — Ideogram gets bbox control natively via JSON; other models get it via attention masks built from the *same* `StructuredPrompt`. (Attention-coupling implementation is a separate, larger effort — this doc only fixes the data path; flag the kernel/loop work as future.)

## Magic-prompt expander (optional LLM)

```csharp
/// <summary>Expands a plain-text idea into a StructuredPrompt via an LLM. Optional — the builder works without one.</summary>
public interface IMagicPromptExpander
{
    Task<StructuredPrompt> ExpandAsync(string plainPrompt, CancellationToken ct = default);
}
```

- The **system prompts** that instruct the LLM to emit Ideogram's JSON live in `ideogram4/magic_prompt_system_prompts/` — port the relevant one as the default expander template.
- Concrete expanders (each behind the interface, no hard dependency in the core builder):
  - **`ClaudeMagicPromptExpander`** — calls the Claude API (the upstream repo offers `ClaudeOpusMagicPromptV1` / `ClaudeSonnetMagicPromptV1` via OpenRouter; we'd target the Anthropic API directly). See the `claude-api` skill for current model IDs/params before wiring.
  - **`LocalLlmMagicPromptExpander`** — use a local LLM via the native `HartsyInference.LLM` package (config-driven Qwen2/Qwen3/Llama/Mistral decoder), fully offline, no API key.
  - **`LocalGemmaMagicPromptExpander`** — mirrors what ComfyUI does (local Gemma-4). Lower priority.
- **Important caveat from upstream:** "The magic prompt shipped here is **not** the same magic prompt used in production at Ideogram.ai — results will differ." Set user expectations; the expander is a convenience, not a fidelity guarantee.
- **Safety note:** Ideogram's safety filter has a **higher false-positive rate for non-JSON-like prompts** — another reason the structured path is preferred.

## Package placement & files

`src/HartsyInference.Diffusion/Prompting/` (pure data + serialization; no backend/tensor deps):

| File | Contents |
|---|---|
| `StructuredPrompt.cs` | `StructuredPrompt` record |
| `StyleBlock.cs` | `StyleBlock` record |
| `PromptElement.cs` | `PromptElement` base + `ObjectElement` + `TextElement` |
| `BoundingBox.cs` | `BoundingBox` readonly record struct + validation |
| `StructuredPromptBuilder.cs` | fluent builder |
| `IPromptDialect.cs` | dialect interface + `RegionPrompt` / `RectMask` |
| `Dialects/Ideogram4Dialect.cs` | exact JSON serializer + validator |
| `Dialects/NaturalLanguageDialect.cs` | prose flattener |
| `Dialects/RegionalAttentionDialect.cs` | bbox→region-mask (data path only; attention coupling is future) |
| `MagicPrompt/IMagicPromptExpander.cs` | LLM expander interface |
| `MagicPrompt/Ideogram4SystemPrompt.cs` | ported default JSON-builder system prompt |

The `Ideogram4Pipeline` accepts **either** a raw conditioning string **or** a `StructuredPrompt` (it calls `Ideogram4Dialect.Serialize` internally), so callers aren't forced through the builder but get correctness for free if they use it.

## Testing

- **`StructuredPromptDialectTests.cs`** (`HartsyInference.Diffusion.Tests`):
  - Golden-file test: a known `StructuredPrompt` serializes byte-for-byte to an expected Ideogram JSON string (separators, key order, uppercase hex, literal unicode).
  - Round-trip: parse one of the `magic_prompt_system_prompts/` example JSONs into `StructuredPrompt` and re-serialize → identical.
  - Validator rejects: lowercase hex, `#abc` shorthand, >16/>5 palette, both photo+art_style, missing background, bbox out of `[0,1000]`, `min≥max`.
  - `NaturalLanguageDialect` produces non-empty prose with all element descriptions present.
  - `RegionalAttentionDialect` maps a 0–1000 bbox to the correct pixel rect at 1024×768.

## Open Questions

- **Exact field set in the official `magic_prompt_system_prompts/`** — port the precise schema text so our default expander matches Ideogram's training prompt. (Read the directory contents.)
- **Are `aesthetics`/`lighting`/`medium` strictly required when `style_description` is present**, or tolerated-if-missing? `prompting.md` says required-when-present; confirm against examples.
- **Coordinate origin** — confirm `[y_min, x_min, y_max, x_max]` (row-major, y first). `prompting.md` states this order; double-check against a rendered example before trusting it for region masks.
- **Regional-attention coupling mechanism** for non-Ideogram models is out of scope here (data path only). When we build it, decide: per-region cross-attention bias vs latent-couple (e.g. "Attention Couple" style). Separate research doc when that lands.

## Implementation Notes

- **Reuse, don't duplicate:** there is no existing prompt-construction utility in the repo (grep confirmed). This is genuinely new shared infrastructure — build it once under `Prompting/` and have every layout-aware pipeline consume it, per the AGENTS.md reuse rule.
- **No hot-path concerns** — this runs once per generation, before inference. Plain managed code, `System.Text.Json` source-gen, normal allocations are fine here (the zero-GC rule is for the denoise/attention path, not prompt prep).
- **Keep the LLM out of the core.** The builder + dialects must compile and run with zero LLM dependencies; expanders are opt-in behind the interface.
- **SwarmUI angle:** the SwarmUI-HartsyInference extension ([[swarmui_extension]] memory) can surface the builder as a structured-prompt UI (scene/style/elements with draggable bboxes) and call `Ideogram4Dialect.Serialize` — that's how "people having to LLM-prompt each region" becomes a real UI instead of hand-written JSON.
</content>
