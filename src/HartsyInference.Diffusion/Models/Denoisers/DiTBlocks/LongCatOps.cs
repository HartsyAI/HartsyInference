using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Norm;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Per-frame adaLN operations of the LongCat DiT: every latent frame has its own timestep embedding row, so each modulation vector is
/// shared by the <c>tokensPerFrame</c> consecutive token rows of its frame. The tables stay packed as the projection wrote them; no per-token
/// copies are made. Every function that returns a new tensor releases it when a backend call throws.</summary>
internal static class LongCatOps
{
    /// <summary><c>Linear(SiLU(timeEmbed))</c>: the packed modulation <c>[frames, width]</c>. Runs through <see cref="IBackend.Linear"/> so a runtime
    /// LoRA adjunct on the projection applies.</summary>
    public static Tensor AdaLnProjection(IBackend backend, Tensor timeEmbed, Tensor weight, Tensor? bias)
    {
        using Tensor activated = new(timeEmbed.Shape, DType.F32);
        backend.Silu(activated, timeEmbed);
        return Fill(new TensorShape(timeEmbed.Shape[0], weight.Shape[0]), DType.F32, packed => backend.Linear(packed, activated, weight, bias));
    }

    /// <summary><c>LayerNorm(x) · (1 + scale) + shift</c> with scale and shift read from <paramref name="packed"/> at the given column offsets,
    /// one row per frame.</summary>
    /// <param name="roundOutput">Rounds the output to the BF16 grid: the reference's modulate_fp32 casts back to the input's dtype.</param>
    public static Tensor NormModulate(IBackend backend, Tensor x, Tensor packed, int scaleOffset, int shiftOffset, int tokensPerFrame,
        bool roundOutput = false)
    {
        ModulationSpec modulation = ModulationSpec.Packed(packed, packed.Shape[1], scaleOffset, shiftOffset, tokensPerFrame, ScaleMode.OnePlus);
        Tensor output = new(x.Shape, DType.F32);
        try
        {
            backend.NormModulate(NormModulateCall.Rows(output, x, NormKind.LayerNorm, LongCatConfig.NormEps).WithModulation(in modulation));
        }
        catch
        {
            output.Dispose();
            throw;
        }
        return RoundOwned(backend, output, roundOutput);
    }

    /// <summary>The caption projection (PixArtAlpha: Linear, GELU, Linear) with BF16 rounding after every op, as the reference's y_embedder runs in
    /// BF16 (per-stage check, B6). The encoder is rounded on entry. A new tensor, owned by the caller.</summary>
    public static Tensor TextProjectionRounded(IBackend backend, Tensor encoder, int dim, Tensor w1, Tensor? b1, Tensor w2, Tensor? b2)
    {
        int lastDim = (int)encoder.Shape[encoder.Shape.Rank - 1];
        int l = (int)(encoder.Shape.ElementCount / lastDim);
        using Tensor input = RoundToBf16(backend, encoder);
        using Tensor h1 = RoundOwned(backend, Fill(new TensorShape(l, dim), DType.F32, t => backend.Linear(t, input, w1, b1)), on: true);
        using Tensor act = RoundOwned(backend, Fill(h1.Shape, DType.F32, t => backend.Gelu(t, h1)), on: true);
        return RoundOwned(backend, Fill(new TensorShape(l, dim), DType.F32, t => backend.Linear(t, act, w2, b2)), on: true);
    }

    /// <summary>The tensor rounded to the BF16 grid when <paramref name="on"/>. Takes ownership of <paramref name="owned"/>: it is disposed on every
    /// path, and a new tensor is returned. Otherwise the input itself is returned, so the caller's <c>using</c> disposes it either way.</summary>
    public static Tensor RoundOwned(IBackend backend, Tensor owned, bool on)
    {
        if (!on)
            return owned;
        try
        {
            return RoundToBf16(backend, owned);
        }
        finally
        {
            owned.Dispose();
        }
    }

    /// <summary>Rounds an F32 tensor to BF16 (nearest-even) and back, so the values sit on the BF16 grid the reference's x holds. A new tensor.</summary>
    public static Tensor RoundToBf16(IBackend backend, Tensor values)
    {
        using Tensor bf16 = new(values.Shape, DType.BF16);
        backend.CastToBf16(bf16, values);
        return Fill(values.Shape, DType.F32, t => backend.CastToF32(t, bf16));
    }

    /// <summary><c>residual + gate · value</c> with the gate of each token's frame, read from <paramref name="packed"/> at
    /// <paramref name="gateOffset"/>.</summary>
    public static Tensor GatedResidual(IBackend backend, Tensor residual, Tensor value, Tensor packed, int gateOffset, int width,
        Tensor frameRowIndex)
    {
        using Tensor gates = new(new TensorShape(packed.Shape[0], width), DType.F32);
        backend.SliceLastDim(gates, packed, gateOffset);
        return Fill(residual.Shape, DType.F32, o => backend.GatedResidualRowIndexed(o, residual, value, gates, frameRowIndex));
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
    public static Tensor Copy(IBackend backend, Tensor source) =>
        Fill(source.Shape, source.DType, t => backend.CopyInto(t, source));

    /// <summary>Row-wise sum of two equal tensors.</summary>
    public static Tensor Add(IBackend backend, Tensor a, Tensor b) =>
        Fill(a.Shape, DType.F32, o => backend.Add(o, a, b));

    /// <summary>A new tensor filled by <paramref name="fill"/>. Released when the fill throws, so a caller receives a complete tensor or nothing.</summary>
    private static Tensor Fill(TensorShape shape, DType dtype, Action<Tensor> fill)
    {
        Tensor output = new(shape, dtype);
        try
        {
            fill(output);
        }
        catch
        {
            output.Dispose();
            throw;
        }
        return output;
    }
}
