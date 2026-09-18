using HartsyInference.Core.Exceptions;

namespace HartsyInference.Core.Tensors;

/// <summary>LoRA deltas carried ON a weight instead of merged INTO it, so a block-quantized base keeps its packed bytes: every GEMM adds <c>Σ scale·(x·Downᵀ)·Upᵀ</c> to its own result.</summary>
/// <remarks><para>This is the only way to LoRA a classic-codec GGUF. Q4_0/Q5_0/Q5_1 have no quantizer
/// (<c>SupportsQuantize</c> covers Q4_K/Q5_K/Q6_K/Q8_0 only), and requantizing a merged result would degrade the base
/// as well as the LoRA — so ComfyUI-GGUF patches the dequantized weight per forward, and this is the same arithmetic
/// without the per-forward dequant. fp8 and int8_tensorwise bases keep the dequant-merge-requant round trip, which is
/// ComfyUI's approach for those.</para>
/// <para><b>Never set this on a shared base tensor.</b> Cached converted-weight dictionaries, resident models, shared
/// text-encoder/VAE components and the identity-keyed GPU cache all hold that object, so a no-LoRA request reusing it
/// would silently carry the previous request's LoRA. Use <see cref="Tensor.WithLowRankAdjunct"/>, which aliases the
/// same bytes under a fresh <see cref="Tensor"/> the LoRA stack owns and disposes.</para>
/// <para>Stacked LoRAs on one weight are separate <see cref="Terms"/>, applied in order; ranks are never
/// concatenated, because each term carries its own scale.</para></remarks>
public sealed class LowRankAdjunct
{
    private readonly Dictionary<(long Offset, long Count), LowRankAdjunct> _rowWindows = [];

    /// <summary>The additive terms, applied in order.</summary>
    public required IReadOnlyList<LowRankAdjunctTerm> Terms { get; init; }

    /// <summary>Rows of the delta — the target weight's output dimension.</summary>
    public long OutFeatures => Terms[0].OutFeatures;

    /// <summary>Columns of the delta — the target weight's input dimension.</summary>
    public long InFeatures => Terms[0].InFeatures;

    /// <summary>Throws unless every term matches a weight of <paramref name="outFeatures"/> × <paramref name="inFeatures"/>.</summary>
    /// <param name="weightKey">The weight's checkpoint key, named in the refusal.</param>
    public void Validate(long outFeatures, long inFeatures, string weightKey)
    {
        if (Terms.Count == 0)
        {
            throw new HartsyInferenceException($"LoRA adjunct for '{weightKey}' carries no terms.");
        }
        foreach (LowRankAdjunctTerm term in Terms)
        {
            term.Validate(outFeatures, inFeatures, weightKey);
        }
    }

    /// <summary>Returns this adjunct narrowed to a contiguous run of output rows — what a windowed GEMM (a fused projection consumed in parts) needs.</summary>
    /// <remarks>The result is MEMOIZED per window. Backends cache device copies by tensor identity, so handing a
    /// freshly-sliced <see cref="LowRankAdjunctTerm.Up"/> to every call of a per-step projection would re-upload it
    /// every step; the same window must come back as the same objects.</remarks>
    public LowRankAdjunct SliceRows(long rowOffset, long rowCount)
    {
        lock (_rowWindows)
        {
            if (_rowWindows.TryGetValue((rowOffset, rowCount), out LowRankAdjunct? cached))
            {
                return cached;
            }
            LowRankAdjunctTerm[] sliced = new LowRankAdjunctTerm[Terms.Count];
            for (int i = 0; i < Terms.Count; i++)
            {
                sliced[i] = Terms[i].SliceRows(rowOffset, rowCount);
            }
            LowRankAdjunct window = new() { Terms = sliced };
            _rowWindows[(rowOffset, rowCount)] = window;
            return window;
        }
    }

    /// <summary>Yields every matrix a GEMM will read, including the row windows produced so far — what a backend preloads to device and frees alongside the weight itself.</summary>
    public IEnumerable<Tensor> EnumerateOperands()
    {
        foreach (LowRankAdjunctTerm term in Terms)
        {
            yield return term.Down;
            if (term.Up is not null)
            {
                yield return term.Up;
            }
        }
        LowRankAdjunct[] windows;
        lock (_rowWindows)
        {
            windows = [.. _rowWindows.Values];
        }
        foreach (LowRankAdjunct window in windows)
        {
            foreach (Tensor operand in window.EnumerateOperands())
            {
                yield return operand;
            }
        }
    }

    /// <summary>Yields <paramref name="weights"/> followed by every adjunct matrix any of them carries — the expansion a backend's preload/free pass applies so a LoRA'd weight's factors are resident for the same window the weight is.</summary>
    /// <remarks>Without this the factors upload per call, which is a real cost on the eager path and outright breaks
    /// CUDA-graph capture: a host-to-device copy of a non-resident tensor inside a capture is not a replayable node.</remarks>
    public static IEnumerable<Tensor> ExpandWeights(IEnumerable<Tensor> weights)
    {
        foreach (Tensor weight in weights)
        {
            yield return weight;
            if (weight.LowRankAdjunct is not null)
            {
                foreach (Tensor operand in weight.LowRankAdjunct.EnumerateOperands())
                {
                    yield return operand;
                }
            }
        }
    }
}
