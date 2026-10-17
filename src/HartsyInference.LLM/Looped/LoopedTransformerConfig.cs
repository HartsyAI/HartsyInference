using HartsyInference.Core.Configuration;

namespace HartsyInference.LLM.Looped;

/// <summary>Shape of a looped (latent-recurrent) normalized-GPT decoder such as NVIDIA Nemotron Dory: an input block, a recurrent block
/// whose weights are applied <see cref="Loops"/> times with the input block's output re-injected before every pass, and an output block.
/// Every physical layer is one sublayer (<see cref="LoopedLayerKind"/>) joined to the residual by a hypersphere interpolation with a learned
/// per-channel step (nGPT), so there are no norm weights: every normalization is the parameter-free unit-norm.</summary>
/// <remarks>The learned scale vectors the checkpoint ships are stored in nGPT's "init divided by scaling" form; the loader bakes
/// <see cref="InitStd"/>, <see cref="AlphaInit"/> and <see cref="GateInit"/> into them once, so the runtime multiplies plain vectors.</remarks>
public sealed record LoopedTransformerConfig
{
    /// <summary>Loop count the trained checkpoints are documented for; a cache is sized for at least this many so a request can dial up without a reload.</summary>
    public const int DefaultMaxLoops = 4;

    /// <summary>Residual-stream width.</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Attention query heads.</summary>
    public required int NumHeads { get; init; }

    /// <summary>Key/value heads (GQA).</summary>
    public required int NumKvHeads { get; init; }

    /// <summary>Per-head dimension; RoPE rotates the whole head.</summary>
    public required int HeadDim { get; init; }

    /// <summary>SwiGLU inner width of the MLP layers.</summary>
    public required int IntermediateSize { get; init; }

    /// <summary>Vocabulary size (embedding rows and logit width).</summary>
    public required int VocabSize { get; init; }

    /// <summary>Longest position the model was trained for.</summary>
    public int MaxPositionEmbeddings { get; init; } = 8192;

    /// <summary>Physical layers of the input block, run once before the recurrence.</summary>
    public required int InputLayers { get; init; }

    /// <summary>Physical layers of the shared recurrent block, run <see cref="Loops"/> times.</summary>
    public required int RecurrentLayers { get; init; }

    /// <summary>Physical layers of the output block, run once after the recurrence.</summary>
    public required int OutputLayers { get; init; }

    /// <summary>Loop count the checkpoint declares; a request may choose another up to <see cref="EffectiveMaxLoops"/>.</summary>
    public required int Loops { get; init; }

    /// <summary>Most loops a sequence may run, which sizes the per-loop caches; 0 (default) means <see cref="Loops"/> or <see cref="DefaultMaxLoops"/>, whichever is larger.</summary>
    public int MaxLoops { get; init; }

    /// <summary>One entry per physical layer, input block first, then the recurrent block, then the output block.</summary>
    public required IReadOnlyList<LoopedLayerKind> LayerKinds { get; init; }

    /// <summary>RoPE base frequency per physical layer (Dory gives sliding and full attention layers different bases); unread for MLP layers.</summary>
    public required IReadOnlyList<float> RopeThetas { get; init; }

    /// <summary>Keys a sliding-attention query may see; required when any layer is <see cref="LoopedLayerKind.SlidingAttention"/>.</summary>
    public int SlidingWindow { get; init; }

    /// <summary>Key/value storage of the recurrent attention layers across loops.</summary>
    public LoopedKvCacheMode KvCacheMode { get; init; } = LoopedKvCacheMode.PerLoop;

    /// <summary>nGPT initialization scale (<c>init_method_std</c>); every learned scale vector is divided by it.</summary>
    public float InitStd { get; init; } = 0.02f;

    /// <summary>Initial value of the residual interpolation step (<c>ngpt_alpha_init</c>).</summary>
    public float AlphaInit { get; init; } = 0.05f;

    /// <summary>Initial value of the attention output gate's logit scale; 0 (default) means <c>sqrt(HiddenSize)</c>, the checkpoint's own fallback.</summary>
    public float GateInit { get; init; }

    /// <summary>Epsilon inside every unit-norm: <c>x / sqrt(sum(x²) + eps)</c>.</summary>
    public float NormEps { get; init; } = 1e-6f;

    /// <summary>The checkpoint's architecture name (GGUF <c>general.architecture</c>, e.g. <c>dory</c>); null for a config built in code.</summary>
    public string? Architecture { get; init; }

    /// <summary>For quantized weights: keep them compressed on the device and dequantize per projection (low VRAM) instead of caching an F16 copy.</summary>
    public bool LowVramQuant { get; init; } = EngineKnobs.LowvramQuant.Value;

    /// <summary>Layers with their own weights: input + recurrent + output.</summary>
    public int PhysicalLayers => InputLayers + RecurrentLayers + OutputLayers;

    /// <summary>Query projection width, <see cref="NumHeads"/> × <see cref="HeadDim"/>.</summary>
    public int QDim => NumHeads * HeadDim;

    /// <summary>Key/value projection width, <see cref="NumKvHeads"/> × <see cref="HeadDim"/>.</summary>
    public int KvDim => NumKvHeads * HeadDim;

    /// <summary>Query heads per key/value head.</summary>
    public int KvGroup => NumHeads / NumKvHeads;

    /// <summary><see cref="MaxLoops"/> when set, else the larger of <see cref="Loops"/> and <see cref="DefaultMaxLoops"/>.</summary>
    public int EffectiveMaxLoops => MaxLoops > 0 ? MaxLoops : Math.Max(Loops, DefaultMaxLoops);

    /// <summary><see cref="GateInit"/> when set, else <c>sqrt(HiddenSize)</c>.</summary>
    public float EffectiveGateInit => GateInit > 0f ? GateInit : MathF.Sqrt(HiddenSize);

    /// <summary>Whether physical layer <paramref name="physical"/> is an attention layer (and so owns a key/value cache slot).</summary>
    public bool IsAttention(int physical) => LayerKinds[physical] != LoopedLayerKind.Mlp;

    /// <summary>Whether physical layer <paramref name="physical"/> restricts attention to <see cref="SlidingWindow"/> keys.</summary>
    public bool IsSliding(int physical) => LayerKinds[physical] == LoopedLayerKind.SlidingAttention;

    /// <summary>Whether physical layer <paramref name="physical"/> belongs to the recurrent block.</summary>
    public bool IsRecurrent(int physical) => physical >= InputLayers && physical < InputLayers + RecurrentLayers;

    /// <summary>The attention layers among the first <paramref name="count"/> physical layers starting at <paramref name="start"/>.</summary>
    public int AttentionLayersIn(int start, int count)
    {
        int n = 0;
        for (int p = start; p < start + count; p++)
        {
            if (IsAttention(p)) n++;
        }
        return n;
    }

    /// <summary>Checks the shape for internal consistency; throws <see cref="ArgumentException"/> naming the first problem.</summary>
    public void Validate()
    {
        if (HiddenSize <= 0 || NumHeads <= 0 || NumKvHeads <= 0 || HeadDim <= 0 || IntermediateSize <= 0 || VocabSize <= 0)
            throw new ArgumentException("Looped transformer dimensions must be positive.");
        if (NumHeads % NumKvHeads != 0)
            throw new ArgumentException($"NumHeads ({NumHeads}) must be a multiple of NumKvHeads ({NumKvHeads}).");
        if (HeadDim % 2 != 0)
            throw new ArgumentException($"HeadDim ({HeadDim}) must be even for RoPE.");
        if (InputLayers < 0 || RecurrentLayers <= 0 || OutputLayers < 0)
            throw new ArgumentException("The recurrent block needs at least one layer and the input/output blocks cannot be negative.");
        if (Loops < 1)
            throw new ArgumentException($"Loops ({Loops}) must be at least 1.");
        if (MaxLoops < 0 || (MaxLoops > 0 && MaxLoops < Loops))
            throw new ArgumentException($"MaxLoops ({MaxLoops}) must be 0 or at least Loops ({Loops}).");
        if (LayerKinds.Count != PhysicalLayers)
            throw new ArgumentException($"LayerKinds has {LayerKinds.Count} entries for {PhysicalLayers} physical layers.");
        if (RopeThetas.Count != PhysicalLayers)
            throw new ArgumentException($"RopeThetas has {RopeThetas.Count} entries for {PhysicalLayers} physical layers.");
        bool anySliding = false;
        bool anyAttention = false;
        for (int p = 0; p < PhysicalLayers; p++)
        {
            if (!IsAttention(p)) continue;
            anyAttention = true;
            anySliding |= IsSliding(p);
            if (!(RopeThetas[p] > 0f) || !float.IsFinite(RopeThetas[p]))
                throw new ArgumentException($"Layer {p} needs a positive finite RoPE base; got {RopeThetas[p]}.");
        }
        if (!anyAttention)
            throw new ArgumentException("A looped transformer needs at least one attention layer.");
        if (anySliding && SlidingWindow <= 0)
            throw new ArgumentException("SlidingWindow must be positive when a layer uses sliding attention.");
        if (!(InitStd > 0f) || !(AlphaInit > 0f) || GateInit < 0f || !(NormEps > 0f))
            throw new ArgumentException("InitStd, AlphaInit and NormEps must be positive and GateInit non-negative.");
    }
}
