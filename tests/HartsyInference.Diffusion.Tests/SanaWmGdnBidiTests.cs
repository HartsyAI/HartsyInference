using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.SanaWm;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>SANA-WM bidirectional GDN main branch against the float64 Python transcription of the upstream recurrence
/// (<c>Fixtures/generate_sana_wm_s1_fixtures.py</c>), stored as F32 fixtures.</summary>
public sealed class SanaWmGdnBidiTests
{
    private const float FixtureTolerance = 1e-5f;

    [Theory]
    [InlineData("sana_wm_s1_gdn_bidi_tiny.safetensors")]
    [InlineData("sana_wm_s1_gdn_bidi_single_frame.safetensors")]
    public void Forward_MatchesPythonReference_WithinAbsoluteTolerance(string fixture)
    {
        using SafeTensorsLoader loader = LoadFixture(fixture);
        using IBackend backend = new CpuBackend();
        (SanaWmGdnWeights weights, int heads, int headDim) = ReadWeights(loader);
        using SanaWmGdnBidi layer = new(backend, weights, heads, headDim);
        Tensor x = loader.GetTensor("x");
        using Tensor output = Run(layer, loader, x);
        float[] expected = loader.GetTensor("expected").AsReadOnlySpan<float>().ToArray();

        float maxError = MaxAbsDiff(output.AsReadOnlySpan<float>(), expected);

        Assert.True(maxError <= FixtureTolerance, $"{fixture}: max abs error {maxError:E3} exceeds {FixtureTolerance:E0}.");
    }

    [Fact]
    public void Forward_ReturnsBatchTokenChannelShape()
    {
        using SafeTensorsLoader loader = LoadFixture("sana_wm_s1_gdn_bidi_tiny.safetensors");
        using IBackend backend = new CpuBackend();
        (SanaWmGdnWeights weights, int heads, int headDim) = ReadWeights(loader);
        using SanaWmGdnBidi layer = new(backend, weights, heads, headDim);
        using Tensor output = Run(layer, loader, loader.GetTensor("x"));

        Assert.Equal(DType.F32, output.DType);
        Assert.Equal(3, output.Shape.Rank);
        Assert.Equal(2, output.Shape[0]);
        Assert.Equal(24, output.Shape[1]);
        Assert.Equal(16, output.Shape[2]);
    }

    [Fact]
    public void Forward_ZeroBeta_LeavesStateAtZeroAndOutputZero()
    {
        using SafeTensorsLoader loader = LoadFixture("sana_wm_s1_gdn_bidi_tiny.safetensors");
        using IBackend backend = new CpuBackend();
        (SanaWmGdnWeights weights, int heads, int headDim) = ReadWeights(loader);
        using Tensor zeroWeight = Filled(backend, weights.BetaWeight.Shape, 0f);
        using Tensor closedBias = Filled(backend, weights.BetaBias.Shape, -1e4f);
        SanaWmGdnWeights closed = weights with { BetaWeight = zeroWeight, BetaBias = closedBias };
        using SanaWmGdnBidi layer = new(backend, closed, heads, headDim);
        using Tensor output = Run(layer, loader, loader.GetTensor("x"));

        foreach (float value in output.AsReadOnlySpan<float>())
        {
            Assert.Equal(0f, value);
        }
    }

    [Fact]
    public void Forward_PerturbingEitherEndFrame_ChangesTheOtherEnd()
    {
        using SafeTensorsLoader loader = LoadFixture("sana_wm_s1_gdn_bidi_tiny.safetensors");
        using IBackend backend = new CpuBackend();
        (SanaWmGdnWeights weights, int heads, int headDim) = ReadWeights(loader);
        using SanaWmGdnBidi layer = new(backend, weights, heads, headDim);
        Tensor x = loader.GetTensor("x");
        using Tensor baseline = Run(layer, loader, x);
        float[] baseFlat = baseline.AsReadOnlySpan<float>().ToArray();

        // Frame 3 is the last of four frames (6 tokens each). Changing it must reach frame 0 through the reverse scan.
        using Tensor lastPerturbed = PerturbFrame(backend, x, 3, 0.5f);
        using Tensor lastChanged = Run(layer, loader, lastPerturbed);
        Assert.True(FrameDiff(baseFlat, lastChanged.AsReadOnlySpan<float>(), 0) > 1e-3f, "reverse scan did not reach frame 0");

        // Changing frame 0 must reach the last frame through the forward scan.
        using Tensor firstPerturbed = PerturbFrame(backend, x, 0, 0.5f);
        using Tensor firstChanged = Run(layer, loader, firstPerturbed);
        Assert.True(FrameDiff(baseFlat, firstChanged.AsReadOnlySpan<float>(), 3) > 1e-3f, "forward scan did not reach frame 3");
    }

    [Fact]
    public void Forward_Bf16Inputs_StayWithinBf16ToleranceOfF32()
    {
        using SafeTensorsLoader loader = LoadFixture("sana_wm_s1_gdn_bidi_tiny.safetensors");
        using IBackend backend = new CpuBackend();
        (SanaWmGdnWeights weights, int heads, int headDim) = ReadWeights(loader);
        using SanaWmGdnBidi reference = new(backend, weights, heads, headDim);
        using Tensor expected = Run(reference, loader, loader.GetTensor("x"));
        float[] expectedFlat = expected.AsReadOnlySpan<float>().ToArray();

        using Tensor xBf16 = loader.GetTensor("x").CastTo(DType.BF16);
        SanaWmGdnWeights bf16 = new()
        {
            Qkv = weights.Qkv.CastTo(DType.BF16),
            ConvK = weights.ConvK.CastTo(DType.BF16),
            BetaWeight = weights.BetaWeight.CastTo(DType.BF16),
            BetaBias = weights.BetaBias.CastTo(DType.BF16),
            GateWeight = weights.GateWeight.CastTo(DType.BF16),
            GateBias = weights.GateBias.CastTo(DType.BF16),
            ALog = weights.ALog.CastTo(DType.BF16),
            DtBias = weights.DtBias.CastTo(DType.BF16),
            QNorm = weights.QNorm.CastTo(DType.BF16),
            KNorm = weights.KNorm.CastTo(DType.BF16),
        };
        using SanaWmGdnBidi layer = new(backend, bf16, heads, headDim);
        using Tensor actual = Run(layer, loader, xBf16);

        float scale = expectedFlat.Max(MathF.Abs);
        float maxError = MaxAbsDiff(actual.AsReadOnlySpan<float>(), expectedFlat);
        Assert.True(maxError <= 0.05f * scale, $"BF16 max abs error {maxError:E3} exceeds 5% of {scale:E3}.");
    }

    [Fact]
    public void Constructor_RejectsWrongWeightShape()
    {
        using SafeTensorsLoader loader = LoadFixture("sana_wm_s1_gdn_bidi_tiny.safetensors");
        using IBackend backend = new CpuBackend();
        (SanaWmGdnWeights weights, int heads, int headDim) = ReadWeights(loader);

        Assert.Throws<ArgumentException>(() => new SanaWmGdnBidi(backend, weights, heads + 1, headDim));
    }

    [Fact]
    public void Forward_RejectsTokenCountThatDisagreesWithGrid()
    {
        using SafeTensorsLoader loader = LoadFixture("sana_wm_s1_gdn_bidi_tiny.safetensors");
        using IBackend backend = new CpuBackend();
        (SanaWmGdnWeights weights, int heads, int headDim) = ReadWeights(loader);
        using SanaWmGdnBidi layer = new(backend, weights, heads, headDim);
        Tensor x = loader.GetTensor("x");

        Assert.Throws<ArgumentException>(() => layer.Forward(x, 4, 2, 2, loader.GetTensor("rope_cos"), loader.GetTensor("rope_sin")));
    }

    [Fact]
    public void Dispose_IsIdempotent_AndForwardAfterDisposeThrows()
    {
        using SafeTensorsLoader loader = LoadFixture("sana_wm_s1_gdn_bidi_tiny.safetensors");
        using IBackend backend = new CpuBackend();
        (SanaWmGdnWeights weights, int heads, int headDim) = ReadWeights(loader);
        SanaWmGdnBidi layer = new(backend, weights, heads, headDim);
        layer.Dispose();
        layer.Dispose();

        Assert.Throws<ObjectDisposedException>(() => layer.Forward(loader.GetTensor("x"), 4, 2, 3, loader.GetTensor("rope_cos"), loader.GetTensor("rope_sin")));
    }

    private static SafeTensorsLoader LoadFixture(string name)
    {
        SafeTensorsLoader loader = new();
        loader.Load(Path.Combine(AppContext.BaseDirectory, "Fixtures", name));
        return loader;
    }

    private static (SanaWmGdnWeights Weights, int Heads, int HeadDim) ReadWeights(SafeTensorsLoader loader)
    {
        IReadOnlyDictionary<string, string> meta = loader.Metadata!;
        SanaWmGdnWeights weights = new()
        {
            Qkv = loader.GetTensor("qkv_weight"),
            ConvK = loader.GetTensor("conv_k_weight"),
            BetaWeight = loader.GetTensor("beta_proj_weight"),
            BetaBias = loader.GetTensor("beta_proj_bias"),
            GateWeight = loader.GetTensor("gate_proj_weight"),
            GateBias = loader.GetTensor("gate_proj_bias"),
            ALog = loader.GetTensor("a_log"),
            DtBias = loader.GetTensor("dt_bias"),
            QNorm = loader.GetTensor("q_norm_weight"),
            KNorm = loader.GetTensor("k_norm_weight"),
        };
        return (weights, int.Parse(meta["heads"]), int.Parse(meta["head_dim"]));
    }

    private static Tensor Run(SanaWmGdnBidi layer, SafeTensorsLoader loader, Tensor x)
    {
        IReadOnlyDictionary<string, string> meta = loader.Metadata!;
        return layer.Forward(
            x,
            int.Parse(meta["frames"]),
            int.Parse(meta["latent_height"]),
            int.Parse(meta["latent_width"]),
            loader.GetTensor("rope_cos"),
            loader.GetTensor("rope_sin"));
    }

    private static Tensor Filled(IBackend backend, Core.Tensors.TensorShape shape, float value)
    {
        Tensor tensor = new(shape, DType.F32);
        backend.Fill(tensor, value);
        return tensor;
    }

    // Adds a constant to every element of one frame of a [B, N, C] input, returning a new owned tensor.
    private static Tensor PerturbFrame(IBackend backend, Tensor x, int frame, float delta)
    {
        int batch = (int)x.Shape[0];
        int tokens = (int)x.Shape[1];
        int channels = (int)x.Shape[2];
        int framesTotal = 4;
        int tokensPerFrame = tokens / framesTotal;
        Tensor perturbed = new(x.Shape, DType.F32);
        float[] values = x.AsReadOnlySpan<float>().ToArray();
        for (int b = 0; b < batch; b++)
        {
            for (int t = frame * tokensPerFrame; t < (frame + 1) * tokensPerFrame; t++)
            {
                for (int c = 0; c < channels; c++)
                {
                    values[((long)b * tokens + t) * channels + c] += delta;
                }
            }
        }

        values.AsSpan().CopyTo(perturbed.AsSpan<float>());
        return perturbed;
    }

    private static float MaxAbsDiff(ReadOnlySpan<float> actual, ReadOnlySpan<float> expected)
    {
        Assert.Equal(expected.Length, actual.Length);
        float max = 0f;
        for (int i = 0; i < expected.Length; i++)
        {
            max = MathF.Max(max, MathF.Abs(actual[i] - expected[i]));
        }

        return max;
    }

    // Largest change in one frame of a [B=2, N=24, C=16] output, frame size 6 tokens.
    private static float FrameDiff(float[] baseline, ReadOnlySpan<float> changed, int frame)
    {
        const int tokensPerFrame = 6;
        const int channels = 16;
        float max = 0f;
        for (int b = 0; b < 2; b++)
        {
            for (int t = frame * tokensPerFrame; t < (frame + 1) * tokensPerFrame; t++)
            {
                for (int c = 0; c < channels; c++)
                {
                    long i = ((long)b * 24 + t) * channels + c;
                    max = MathF.Max(max, MathF.Abs(changed[(int)i] - baseline[i]));
                }
            }
        }

        return max;
    }
}
