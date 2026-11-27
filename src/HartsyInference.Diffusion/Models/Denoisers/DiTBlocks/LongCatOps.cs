using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Norm;
using HartsyInference.Core.Tensors;
using HartsyInference.Core.Layers.Attention;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Per-frame adaLN operations of the LongCat DiT: every latent frame has its own timestep embedding row, so each modulation vector is
/// shared by the <c>tokensPerFrame</c> consecutive token rows of its frame. The tables stay packed as the projection wrote them; no per-token
/// copies are made.</summary>
internal static class LongCatOps
{
    /// <summary><c>Linear(SiLU(timeEmbed))</c>: the packed modulation <c>[frames, width]</c>. Runs through <see cref="IBackend.Linear"/> so a runtime
    /// LoRA adjunct on the projection applies.</summary>
    public static Tensor AdaLnProjection(IBackend backend, Tensor timeEmbed, Tensor weight, Tensor? bias)
    {
        using Tensor activated = new(timeEmbed.Shape, DType.F32);
        backend.Silu(activated, timeEmbed);
        Tensor packed = new(new TensorShape(timeEmbed.Shape[0], weight.Shape[0]), DType.F32);
        backend.Linear(packed, activated, weight, bias);
        return packed;
    }

    /// <summary><c>LayerNorm(x) · (1 + scale) + shift</c> with scale and shift read from <paramref name="packed"/> at the given column offsets,
    /// one row per frame.</summary>
    /// <param name="roundOutput">Rounds the output to the BF16 grid: the reference's modulate_fp32 casts back to the input's dtype.</param>
    public static Tensor NormModulate(IBackend backend, Tensor x, Tensor packed, int scaleOffset, int shiftOffset, int tokensPerFrame,
        bool roundOutput = false)
    {
        Tensor output = new(x.Shape, DType.F32);
        ModulationSpec modulation = ModulationSpec.Packed(packed, packed.Shape[1], scaleOffset, shiftOffset, tokensPerFrame, ScaleMode.OnePlus);
        backend.NormModulate(NormModulateCall.Rows(output, x, NormKind.LayerNorm, LongCatConfig.NormEps).WithModulation(in modulation));
        return RoundOwned(backend, output, roundOutput);
    }

    /// <summary>The rounding as a projection hook (<see cref="AttentionLayer.RoundProjection"/>): always rounds.</summary>
    public static readonly Func<IBackend, Tensor, Tensor> RoundFn = static (backend, values) => RoundOwned(backend, values, on: true);

    /// <summary>The caption projection (PixArtAlpha: Linear, GELU, Linear) with BF16 rounding after every op, as the reference's y_embedder runs in
    /// BF16 (per-stage check, B6). The encoder is rounded on entry. A new tensor, owned by the caller.</summary>
    public static Tensor TextProjectionRounded(IBackend backend, Tensor encoder, int dim, Tensor w1, Tensor? b1, Tensor w2, Tensor? b2)
    {
        int lastDim = (int)encoder.Shape[encoder.Shape.Rank - 1];
        int l = (int)(encoder.Shape.ElementCount / lastDim);
        using Tensor input = RoundToBf16(backend, encoder);
        Tensor h1Raw = new(new TensorShape(l, dim), DType.F32);
        backend.Linear(h1Raw, input, w1, b1);
        using Tensor h1 = RoundOwned(backend, h1Raw, on: true);
        Tensor actRaw = new(h1.Shape, DType.F32);
        backend.Gelu(actRaw, h1);
        using Tensor act = RoundOwned(backend, actRaw, on: true);
        Tensor outRaw = new(new TensorShape(l, dim), DType.F32);
        backend.Linear(outRaw, act, w2, b2);
        return RoundOwned(backend, outRaw, on: true);
    }

    /// <summary>The tensor rounded to the BF16 grid when <paramref name="on"/>: the input is disposed and a new tensor returned. Otherwise the
    /// input itself is returned, so the caller's <c>using</c> disposes it either way.</summary>
    public static Tensor RoundOwned(IBackend backend, Tensor owned, bool on)
    {
        if (!on)
            return owned;
        Tensor rounded = RoundToBf16(backend, owned);
        owned.Dispose();
        return rounded;
    }

    /// <summary>Rounds an F32 tensor to BF16 (nearest-even) and back, so the values sit on the BF16 grid the reference's x holds. A new tensor.</summary>
    public static Tensor RoundToBf16(IBackend backend, Tensor values)
    {
        using Tensor bf16 = new(values.Shape, DType.BF16);
        backend.CastToBf16(bf16, values);
        Tensor rounded = new(values.Shape, DType.F32);
        backend.CastToF32(rounded, bf16);
        return rounded;
    }

    /// <summary><c>residual + gate · value</c> with the gate of each token's frame, read from <paramref name="packed"/> at
    /// <paramref name="gateOffset"/>.</summary>
    public static Tensor GatedResidual(IBackend backend, Tensor residual, Tensor value, Tensor packed, int gateOffset, int width,
        Tensor frameRowIndex)
    {
        using Tensor gates = new(new TensorShape(packed.Shape[0], width), DType.F32);
        backend.SliceLastDim(gates, packed, gateOffset);
        Tensor output = new(residual.Shape, DType.F32);
        backend.GatedResidualRowIndexed(output, residual, value, gates, frameRowIndex);
        return output;
    }

    /// <summary>I32 <c>[frames · tokensPerFrame]</c> mapping each token row to its frame's modulation row.</summary>
    public static unsafe Tensor FrameRowIndex(int frames, int tokensPerFrame)
    {
        Tensor index = new(new TensorShape((long)frames * tokensPerFrame), DType.I32);
        int* p = (int*)index.DataPointer;
        for (int f = 0; f < frames; f++)
            for (int i = 0; i < tokensPerFrame; i++)
                p[f * tokensPerFrame + i] = f;
        return index;
    }

    /// <summary>An F32 view of a checkpoint vector. A widened copy is added to <paramref name="owned"/> so the owner disposes it.</summary>
    public static Tensor WidenF32(IReadOnlyDictionary<string, Tensor> weights, string key, List<Tensor> owned) =>
        CheckpointTensors.RequireF32(weights, "LongCat", key, owned);

    /// <summary>A device-friendly copy of <paramref name="source"/>.</summary>
    public static Tensor Copy(IBackend backend, Tensor source)
    {
        Tensor copy = new(source.Shape, source.DType);
        backend.CopyInto(copy, source);
        return copy;
    }

    /// <summary>Row-wise sum of two equal tensors.</summary>
    public static Tensor Add(IBackend backend, Tensor a, Tensor b)
    {
        Tensor output = new(a.Shape, DType.F32);
        backend.Add(output, a, b);
        return output;
    }
}
