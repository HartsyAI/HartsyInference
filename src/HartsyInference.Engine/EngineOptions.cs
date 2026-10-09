using HartsyInference.Core.Backends;
using HartsyInference.Core.MemoryManagement;

namespace HartsyInference.Engine;

/// <summary>Compute/model-lifecycle configuration for the inference engine, independent of any transport. The HTTP
/// server maps its own options onto this; the CLI constructs it directly.</summary>
public sealed class EngineOptions
{
    /// <summary>Optional generation diagnostics; null preserves the ordinary execution path.</summary>
    public Diagnostics.IInferenceDiagnostics? Diagnostics { get; set; }

    /// <summary>Creates the per-request <see cref="Services.ITextStreamFilter"/> that sees every decoded text delta (tool-call parsing lives in the Tools package, not here); null, or a factory returning null, streams plain text exactly as before.</summary>
    public Func<Requests.TextRequest, Services.ITextStreamFilter?>? TextStreamFilterFactory { get; set; }

    /// <summary>Model cache directory for HuggingFace downloads (null = default <c>~/.hartsyinference/models</c>).</summary>
    public string? ModelCacheDirectory { get; set; }

    /// <summary>Low-VRAM policy for this engine's backend; null = follow the <c>vram.lowVram</c> environment
    /// variable. Hosts with a per-backend setting (the SwarmUI extension) pass it here — the env var is process-wide
    /// last-writer-wins, which breaks one-backend-per-GPU setups with differing card sizes.</summary>
    /// <remarks>Superseded by <see cref="VramPolicy"/>, which carries every lever rather than the streaming one alone.
    /// Kept because it is the shape existing hosts already pass; when both are set, <see cref="VramPolicy"/> wins.</remarks>
    public LowVramMode? LowVram { get; set; }

    /// <summary>Full VRAM policy for this engine's backends. Null falls back to <see cref="LowVram"/>, then to the
    /// environment. Set this rather than <see cref="LowVram"/> to control levers beyond weight streaming.</summary>
    public VramPolicy? VramPolicy { get; set; }

    /// <summary>Multi-device placement for this engine (component devices, shard devices); null = single-device.
    /// Can also be changed later via <c>InferenceEngine.SetPlacement</c>.</summary>
    public PlacementConfig? Placement { get; set; }

    /// <summary>Tokens per KV page for each loaded chat (dense/MoE transformer) model's <c>PagedKvPool</c>.</summary>
    public int KvPageSize { get; set; } = 16;

    /// <summary>VRAM budget (bytes) for each loaded chat model's KV pool. <see cref="HartsyInference.LLM.Transformer.PagedKvPool.PageCountForBudget"/> converts this
    /// into a page COUNT sized to the model's actual KV dimensions (numLayers × numKvHeads × headDim, F32, K+V),
    /// allocated up front — a fixed page count is unsafe across model shapes. Once the budget is spent, a scheduled sequence that needs a page, at admission
    /// or mid-decode, fails with <see cref="HartsyInference.LLM.Transformer.KvPoolExhaustedException"/>, which the API answers with 500; a mid-decode failure
    /// fails every sequence in that round. Admission by pages, with 429 for a full queue, arrives with PR 18b. Default 512 MB is a conservative
    /// single-GPU-dev starting point.</summary>
    public long KvPoolBytesBudget { get; set; } = 512L * 1024 * 1024;
}
