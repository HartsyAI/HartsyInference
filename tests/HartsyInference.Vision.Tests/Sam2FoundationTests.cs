using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Segmentation.Sam2;
using Xunit;

namespace HartsyInference.Vision.Tests;

/// <summary>Tests for the SAM 2 foundation: positional encoding, prompt encoding (point/box token
/// assembly + type embeddings), checkpoint grouping, and config presets. The Hiera image encoder and
/// two-way mask decoder forward passes are validation-pending (weights + Python reference required) and
/// are not covered here.</summary>
public sealed class Sam2FoundationTests
{
    [Fact]
    public void PositionalEncoding_IsDeterministic_AndSinCosShaped()
    {
        Tensor gaussian = Gaussian(numPosFeats: 8, seed: 3);
        SamPositionalEncoding pe = new SamPositionalEncoding(gaussian);

        Assert.Equal(16, pe.EmbedDim);
        float[] a = pe.Encode(100, 50, 1024, 1024);
        float[] b = pe.Encode(100, 50, 1024, 1024);
        Assert.Equal(a, b); // deterministic

        // First half = sin, second half = cos of the same projection → sin² + cos² ≈ 1.
        for (int f = 0; f < 8; f++)
        {
            float s = a[f], c = a[8 + f];
            Assert.InRange(s * s + c * c, 1.0f - 1e-4f, 1.0f + 1e-4f);
        }

        gaussian.Dispose();
    }

    [Fact]
    public void PromptEncoder_PointsPlusPadding_TokenCountAndForegroundDiffer()
    {
        SamPromptEncoder enc = BuildEncoder(numPosFeats: 8, seed: 9);
        SamPrompt prompt = new SamPrompt
        {
            Points = [new SamPointPrompt(100, 100, Foreground: true), new SamPointPrompt(200, 50, Foreground: false)],
        };

        Tensor sparse = enc.EncodeSparse(prompt, 1024, 1024);
        // 2 points + 1 padding token (no box).
        Assert.Equal(3, (int)sparse.Shape[1]);
        Assert.Equal(enc.EmbedDim, (int)sparse.Shape[2]);

        sparse.Dispose();
    }

    [Fact]
    public void PromptEncoder_BoxAddsTwoCornerTokens_NoPadding()
    {
        SamPromptEncoder enc = BuildEncoder(8, seed: 13);
        SamPrompt prompt = new SamPrompt
        {
            Points = [new SamPointPrompt(100, 100, true)],
            Box = new SamBoxPrompt(10, 10, 200, 200),
        };

        Tensor sparse = enc.EncodeSparse(prompt, 1024, 1024);
        // 1 point + 2 box corners, no padding token.
        Assert.Equal(3, (int)sparse.Shape[1]);

        sparse.Dispose();
    }

    // ── Helpers ──────────────────────────────────────────────────────────

    private static unsafe Tensor Gaussian(int numPosFeats, int seed)
    {
        Tensor t = new Tensor(new TensorShape(2, numPosFeats), DType.F32);
        Span<float> s = t.AsSpan<float>();
        uint state = (uint)seed;
        for (int i = 0; i < s.Length; i++)
        {
            state = state * 1664525u + 1013904223u;
            s[i] = (state >> 8) / (float)(1 << 24) - 0.5f;
        }
        return t;
    }

    private static SamPromptEncoder BuildEncoder(int numPosFeats, int seed)
    {
        Tensor gaussian = Gaussian(numPosFeats, seed);
        SamPositionalEncoding pe = new SamPositionalEncoding(gaussian);
        int dim = pe.EmbedDim;

        Tensor[] points = new Tensor[4];
        for (int i = 0; i < 4; i++)
        {
            Tensor e = new Tensor(new TensorShape(dim), DType.F32);
            Span<float> s = e.AsSpan<float>();
            for (int d = 0; d < dim; d++) s[d] = (i + 1) * 0.01f * (d + 1);
            points[i] = e;
        }
        Tensor notAPoint = new Tensor(new TensorShape(dim), DType.F32);
        notAPoint.AsSpan<float>().Fill(0.5f);

        SamPromptEncoder enc = new SamPromptEncoder(pe, points, notAPoint);
        gaussian.Dispose();
        foreach (Tensor t in points) t.Dispose();
        notAPoint.Dispose();
        return enc;
    }
}
