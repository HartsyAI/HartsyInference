using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Prompting;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The CondScale seam at Qwen-Image 2.1's exact geometry: weights describe the FULL templated sequence
/// (25 ids for "a red apple") while the conditioning has had the 14-token system turn sliced off, so the
/// right-alignment offset is negative and weight <c>i</c> must land on sequence position <c>i</c>.</summary>
public sealed class QwenImage21WeightSeamTests
{
    [Fact]
    public void ANegativeOffsetPutsThePromptWeightsOnThePromptRows()
    {
        const int total = 25, drop = 14, dim = 8;
        int rows = total - drop;
        float[] weights = new float[total];
        Array.Fill(weights, 1f);
        weights[17] = weights[18] = weights[19] = 1.5f;

        using CpuBackend backend = new CpuBackend();
        Tensor cond = new Tensor(new TensorShape(1, rows, dim), DType.F32);
        cond.AsSpan<float>().Fill(2.0f);

        Tensor? scaled = CondTokenWeights.ScaleRightAligned(backend, cond, weights);
        Assert.NotNull(scaled);
        ReadOnlySpan<float> s = scaled!.AsReadOnlySpan<float>();
        for (int r = 0; r < rows; r++)
        {
            float expected = r is 3 or 4 or 5 ? 3.0f : 2.0f;   // sequence 17,18,19 -> rows 3,4,5
            Assert.Equal(expected, s[r * dim], 4);
        }
        scaled.Dispose();
        cond.Dispose();
    }

    /// <summary>And why that scaling cannot change the image on THIS family. The first thing the DiT does to the
    /// conditioning is <c>txt_in.text_norm</c>, a per-row RMSNorm — and RMSNorm is scale-invariant per row, so a
    /// per-row multiply is cancelled exactly. SwarmUI scales the same tensor, so its CondScale is inert on
    /// Qwen-Image 2.1 in ComfyUI too, and reproducing the no-op IS the parity behaviour (the Kandinsky5 case).
    /// Pinned as a test because "the weighting does nothing" is otherwise indistinguishable from a wiring bug.</summary>
    [Fact]
    public void APerRowScaleIsCancelledByTheTextProjectionsRmsNorm()
    {
        const int rows = 4, dim = 16;
        using CpuBackend backend = new CpuBackend();
        Tensor weight = new Tensor(new TensorShape(dim), DType.F32);
        weight.AsSpan<float>().Fill(1.0f);

        Tensor plain = new Tensor(new TensorShape(1, rows, dim), DType.F32);
        Random rng = new Random(4);
        Span<float> p = plain.AsSpan<float>();
        for (int i = 0; i < p.Length; i++) p[i] = (float)(rng.NextDouble() - 0.5) * 4f;

        Tensor scaled = new Tensor(new TensorShape(1, rows, dim), DType.F32);
        Span<float> q = scaled.AsSpan<float>();
        plain.AsReadOnlySpan<float>().CopyTo(q);
        for (int c = 0; c < dim; c++) q[2 * dim + c] *= 1.5f;   // emphasize row 2

        Tensor a = new Tensor(plain.Shape, DType.F32);
        backend.RmsNorm(a, plain, weight, 1e-6f);
        Tensor b = new Tensor(scaled.Shape, DType.F32);
        backend.RmsNorm(b, scaled, weight, 1e-6f);

        ReadOnlySpan<float> sa = a.AsReadOnlySpan<float>(), sb = b.AsReadOnlySpan<float>();
        for (int i = 0; i < sa.Length; i++)
        {
            Assert.Equal(sa[i], sb[i], 4);
        }
        plain.Dispose(); scaled.Dispose(); a.Dispose(); b.Dispose(); weight.Dispose();
    }
}
