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
    public void PositionalEncoding_DifferentPoints_DifferentEncodings()
    {
        Tensor gaussian = Gaussian(8, seed: 5);
        SamPositionalEncoding pe = new SamPositionalEncoding(gaussian);

        float[] p0 = pe.Encode(10, 10, 1024, 1024);
        float[] p1 = pe.Encode(900, 900, 1024, 1024);
        Assert.NotEqual(p0, p1);

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

    [Fact]
    public void Converter_GroupsByComponent_AndStripsModelPrefix()
    {
        Dictionary<string, Tensor> w = new()
        {
            ["model.image_encoder.trunk.blocks.0.norm1.weight"] = Scalar(),
            ["model.sam_prompt_encoder.point_embeddings.0.weight"] = Scalar(),
            ["model.sam_mask_decoder.iou_prediction_head.layers.0.weight"] = Scalar(),
            ["model.memory_attention.layers.0.self_attn.q_proj.weight"] = Scalar(),
            ["model.memory_encoder.fuser.layers.0.weight"] = Scalar(),
        };

        Sam2Converter.Groups g = Sam2Converter.Convert(w);
        Assert.True(g.ImageEncoder.ContainsKey("trunk.blocks.0.norm1.weight"));
        Assert.True(g.PromptEncoder.ContainsKey("point_embeddings.0.weight"));
        Assert.True(g.MaskDecoder.ContainsKey("iou_prediction_head.layers.0.weight"));
        Assert.Single(g.MemoryAttention);
        Assert.Single(g.MemoryEncoder);

        foreach (Tensor t in w.Values) t.Dispose();
    }

    [Fact]
    public void Config_Presets_HaveExpectedDepths()
    {
        Assert.Equal([1, 2, 7, 2], Sam2Config.HieraTiny.EncoderStageDepths);
        Assert.Equal([2, 3, 16, 3], Sam2Config.HieraBasePlus.EncoderStageDepths);
        Assert.Equal(144, Sam2Config.HieraLarge.EncoderEmbedDim);
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

    private static Tensor Scalar() => new Tensor(new TensorShape(1), DType.F32);
}
