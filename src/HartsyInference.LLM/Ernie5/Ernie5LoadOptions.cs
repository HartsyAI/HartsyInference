using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Ernie5;

/// <summary>How <see cref="Ernie5Model"/> holds its weights and how long its sequences may be.</summary>
public sealed record Ernie5LoadOptions
{
    /// <summary>Longest sequence (prompt plus generation); sizes the rope tables and caps every sequence state.</summary>
    public required int MaxSequenceTokens { get; init; }

    /// <summary>Builds every layer once at load and keeps it; on a GPU the load also preloads each layer's per-token weights while the device
    /// has room. Weights stay mapped views of the checkpoint, but every layer keeps the tensors it decoded to F32 at build (norms, the grouped
    /// output projection, the hyper-connection mappers), so host memory grows by all layers' copies at once. When false each layer is read,
    /// dequantized and dropped on every forward, and no weight stays on the device between forwards.</summary>
    public bool ResidentWeights { get; init; }

    /// <summary>Maps and loads the multi-token-prediction layers so <see cref="Ernie5Model.MtpForward"/> can draft tokens, and gives every
    /// sequence state their caches. When false their keys are skipped (<see cref="Ernie5WeightCoverage.SkippedMtp"/>). Without
    /// <see cref="ResidentWeights"/> every draft call reads and dequantizes the whole MTP layer again, its routed experts included.</summary>
    public bool Mtp { get; init; }

    /// <summary>Layer ranges per backend (a pipeline split across devices): each layer runs on its stage's backend, and the head read-out, LM
    /// head and MTP layer on the last stage's. The residual stream between layers is a host array, so a stage boundary needs no device copy.
    /// Null runs every layer on the backend the model is built with. Needs <see cref="ResidentWeights"/> when it has more than one stage.</summary>
    public LlmPlacement? Placement { get; init; }

    /// <summary>With <see cref="ResidentWeights"/> on GPU stages, also preloads every decoder layer's routed experts and block-scale grids, so no
    /// decoder-layer weight is read from the checkpoint during a forward. The MTP layer is not preloaded: with <see cref="Mtp"/> its weights still
    /// upload per call. The load is refused, naming the stage, when a stage's weights do not fit its device's budget
    /// (<see cref="Ernie5Model.PreloadBudget"/>).</summary>
    public bool PreloadExperts { get; init; }
}
