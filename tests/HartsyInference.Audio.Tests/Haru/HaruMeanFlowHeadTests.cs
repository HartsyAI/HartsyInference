using System.Text.Json;
using HartsyInference.Audio.Layers;
using HartsyInference.Audio.Models.Haru;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests.Haru;

/// <summary>Haru's MeanFlow head against the upstream PyTorch head (<c>tools/haru/haru_reference.py</c>, CPU F32) on a tiny random
/// checkpoint, fed the same hidden states and noise: normalized latents, then un-normalized as the session does.</summary>
public sealed unsafe class HaruMeanFlowHeadTests(ITestOutputHelper output)
{
    private static readonly string Dir = Path.Combine(AppContext.BaseDirectory, "Fixtures", "Haru");

    private static float[] Read(Tensor t) => new ReadOnlySpan<float>((void*)t.DataPointer, (int)t.ElementCount).ToArray();

    private static Tensor Make(float[] values, params long[] dims)
    {
        Tensor t = new(new TensorShape(dims), DType.F32);
        values.AsSpan().CopyTo(new Span<float>((void*)t.DataPointer, values.Length));
        return t;
    }

    private static float RelativeError(float[] got, float[] want)
    {
        Assert.Equal(want.Length, got.Length);
        float worst = 0;
        float scale = 1e-12f;
        for (int i = 0; i < got.Length; i++)
        {
            worst = MathF.Max(worst, MathF.Abs(got[i] - want[i]));
            scale = MathF.Max(scale, MathF.Abs(want[i]));
        }
        return worst / scale;
    }

    private sealed class Fixture : IDisposable
    {
        internal readonly SafeTensorsLoader Weights = new();
        internal readonly SafeTensorsLoader Expected = new();
        internal readonly HaruHeadConfig Config;
        internal readonly Dictionary<string, Tensor> Tensors;

        internal Fixture()
        {
            Weights.Load(Path.Combine(Dir, "haru_tiny.safetensors"));
            Expected.Load(Path.Combine(Dir, "haru_tiny_expected.safetensors"));
            using JsonDocument json = JsonDocument.Parse(File.ReadAllText(Path.Combine(Dir, "config.json")));
            Config = HaruHeadConfig.FromJson(json.RootElement.GetProperty("head"));
            Tensors = Weights.GetAllTensors();
        }

        public void Dispose()
        {
            Weights.Dispose();
            Expected.Dispose();
        }
    }

    [Fact]
    public void Sample_MatchesUpstreamHead()
    {
        using HaruReferenceModel model = new(Path.Combine(Dir, "haru_tiny.safetensors"), Path.Combine(Dir, "config.json"), capacity: 64);
        float rel = model.CompareHead(Path.Combine(Dir, "haru_tiny_expected.safetensors"));
        output.WriteLine($"head latents: rel={rel:E3}");
        Assert.True(rel < 2e-5f, $"relative error {rel:E3}");
    }

    [Theory]
    [InlineData(1, 1.0f)]
    [InlineData(1, 2.0f)]
    [InlineData(2, 1.5f)]
    [InlineData(3, 1.25f)]
    public void Sample_AtOtherStepCountsAndGuidance_MatchesUpstream(int steps, float guidance)
    {
        using Fixture f = new();
        using SafeTensorsLoader want = new();
        want.Load(Path.Combine(Dir, "haru_head_steps_expected.safetensors"));
        using HaruMeanFlowHead head = new(f.Config);
        head.LoadWeights(f.Tensors);
        using CpuBackend backend = new();
        float[] hidden = Read(f.Expected.GetTensor("ref.hidden"));
        float[] noise = Read(f.Expected.GetTensor("in.noise"));
        int n = hidden.Length / f.Config.HiddenSize;
        using Tensor h = Make(hidden, 1, n, f.Config.HiddenSize);
        using Tensor z = Make(noise, 1, n, f.Config.LatentDim);
        using Tensor got = head.Sample(backend, h, z, guidance, steps);
        float rel = RelativeError(Read(got), Read(want.GetTensor(FormattableString.Invariant($"ref.steps{steps}.guidance{guidance:0.0#}"))));
        output.WriteLine($"steps {steps}, guidance {guidance}: rel={rel:E3}");
        Assert.True(rel < 2e-5f, $"relative error {rel:E3}");
    }

    [Fact]
    public void LoadWeights_RefusesASecondLoad()
    {
        using Fixture f = new();
        using HaruMeanFlowHead head = new(f.Config);
        head.LoadWeights(f.Tensors);
        Assert.Throws<InvalidOperationException>(() => head.LoadWeights(f.Tensors));
    }

    [Fact]
    public void Sample_WithBf16Weights_StaysCloseToTheF32Head()
    {
        using Fixture f = new();
        Dictionary<string, Tensor> half = [];
        foreach ((string key, Tensor tensor) in f.Tensors)
        {
            half[key] = key.StartsWith("head.", StringComparison.Ordinal) ? tensor.CastTo(DType.BF16) : tensor;
        }
        using HaruMeanFlowHead full = new(f.Config);
        full.LoadWeights(f.Tensors);
        using HaruMeanFlowHead reduced = new(f.Config);
        reduced.LoadWeights(half);
        using CpuBackend backend = new();
        float[] hidden = Read(f.Expected.GetTensor("ref.hidden"));
        float[] noise = Read(f.Expected.GetTensor("in.noise"));
        int n = hidden.Length / f.Config.HiddenSize;
        using Tensor h = Make(hidden, 1, n, f.Config.HiddenSize);
        using Tensor z = Make(noise, 1, n, f.Config.LatentDim);
        using Tensor want = full.Sample(backend, h, z);
        using Tensor got = reduced.Sample(backend, h, z);
        float rel = RelativeError(Read(got), Read(want));
        output.WriteLine($"bf16 vs f32 head: rel={rel:E3}");
        Assert.True(rel < 5e-2f, $"relative error {rel:E3}");
    }

    [Fact]
    public void Sample_RefusesShapesThatDoNotMatchTheConfig()
    {
        using Fixture f = new();
        using HaruMeanFlowHead head = new(f.Config);
        head.LoadWeights(f.Tensors);
        using CpuBackend backend = new();
        using Tensor hidden = Make(new float[2 * f.Config.HiddenSize], 1, 2, f.Config.HiddenSize);
        using Tensor wrongNoise = Make(new float[3 * f.Config.LatentDim], 1, 3, f.Config.LatentDim);
        Assert.Throws<ArgumentException>(() => head.Sample(backend, hidden, wrongNoise));
        using Tensor noise = Make(new float[2 * f.Config.LatentDim], 1, 2, f.Config.LatentDim);
        Assert.Throws<ArgumentOutOfRangeException>(() => head.Sample(backend, hidden, noise, guidance: 0f));
        using Tensor flat = Make(new float[f.Config.HiddenSize], f.Config.HiddenSize);
        Assert.Throws<ArgumentException>(() => head.Sample(backend, flat, noise));
    }

    [Fact]
    public void ConfigFrom_ReadsTheReleasedHeadBlock()
    {
        using JsonDocument doc = JsonDocument.Parse("""
            {"hidden_size":2560,"latent_dim":64,"num_layers":4,"ffn_ratio":3.0,"norm_eps":1e-05,
             "interval_embed":7.0,"num_steps":1,"guidance_scale":1.5,"guidance_range":[1.0,2.0]}
            """);
        HaruHeadConfig config = HaruHeadConfig.FromJson(doc.RootElement);
        Assert.Equal(2560, config.HiddenSize);
        Assert.Equal(7680, config.FfnSize);
        Assert.Equal(7f, config.IntervalEmbed);
        Assert.Equal(1.5f, config.GuidanceScale);
        Assert.Equal(1, config.NumSteps);
    }

    [Fact]
    public unsafe void ScalarEmbedFeatures_AreCosinesThenSinesOfTheGeometricFrequencies()
    {
        float[] features = new float[2 * 8];
        fixed (float* p = features)
        {
            ScalarEmbed.FillFeatures(p, [0f, 1.5f], 8);
        }
        Assert.Equal([1f, 1f, 1f, 1f, 0f, 0f, 0f, 0f], features[..8]);
        for (int k = 0; k < 4; k++)
        {
            float arg = 1.5f * MathF.Exp(-MathF.Log(10_000f) * k / 4);
            Assert.Equal(MathF.Cos(arg), features[8 + k], 1e-6f);
            Assert.Equal(MathF.Sin(arg), features[8 + 4 + k], 1e-6f);
        }
    }
}
