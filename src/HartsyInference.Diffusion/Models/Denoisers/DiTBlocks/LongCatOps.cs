using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Norm;
using HartsyInference.Core.Tensors;

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
    public static Tensor NormModulate(IBackend backend, Tensor x, Tensor packed, int scaleOffset, int shiftOffset, int tokensPerFrame)
    {
        Tensor output = new(x.Shape, DType.F32);
        ModulationSpec modulation = ModulationSpec.Packed(packed, packed.Shape[1], scaleOffset, shiftOffset, tokensPerFrame, ScaleMode.OnePlus);
        backend.NormModulate(NormModulateCall.Rows(output, x, NormKind.LayerNorm, LongCatConfig.NormEps).WithModulation(in modulation));
        return output;
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

    /// <summary>A zero-filled F32 tensor.</summary>
    public static unsafe Tensor Zeros(TensorShape shape)
    {
        Tensor zeros = new(shape, DType.F32);
        new Span<float>((float*)zeros.DataPointer, checked((int)shape.ElementCount)).Clear();
        return zeros;
    }

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
