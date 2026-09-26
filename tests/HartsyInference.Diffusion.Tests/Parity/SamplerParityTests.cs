using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Sampling;
using HartsyInference.Diffusion.Schedulers;
using Xunit;

namespace HartsyInference.Diffusion.Tests.Parity;

/// <summary>Replays ComfyUI's own sampler trajectories (fixture from <c>tests/python-reference/dump_k_samplers.py</c>)
/// on a synthetic nonlinear denoiser with the reference run's exact noise draws, for epsilon and flow models, from
/// the first step and from an img2img start.</summary>
public sealed class SamplerParityTests
{
    private const double Tolerance = 5e-5;
    private static readonly Lazy<JsonDocument> Fixture = new(() => JsonDocument.Parse(File.ReadAllText(
        Path.Combine(AppContext.BaseDirectory, "Fixtures", "Samplers", "k_sampler_parity.json"))));

    /// <summary>Every case in the fixture.</summary>
    public static TheoryData<string, string, int> Cases()
    {
        TheoryData<string, string, int> data = new();
        foreach (JsonElement c in Fixture.Value.RootElement.GetProperty("cases").EnumerateArray())
        {
            data.Add(c.GetProperty("sampler").GetString()!, c.GetProperty("model").GetString()!, c.GetProperty("start").GetInt32());
        }
        return data;
    }

    /// <summary>The final latent and the sequence of model-query sigmas match ComfyUI.</summary>
    [Theory]
    [MemberData(nameof(Cases))]
    public void MatchesComfyUi(string sampler, string model, int start)
    {
        JsonElement root = Fixture.Value.RootElement;
        JsonElement c = FindCase(root, sampler, model, start);
        TensorShape shape = ReadShape(root.GetProperty("shape"));
        float[] pattern = ReadFloats(root.GetProperty("pattern"));
        float guidance = root.GetProperty("guidance").GetSingle();
        float[] sigmas = ReadFloats(c.GetProperty("sigmas"));
        Dictionary<string, double> percents = [];
        foreach (JsonProperty p in c.GetProperty("percents").EnumerateObject())
        {
            percents[p.Name] = p.Value.GetDouble();
        }
        List<(float Sigma, float SigmaNext, float[] Data)> draws = [];
        foreach (JsonElement d in c.GetProperty("noise").EnumerateArray())
        {
            draws.Add((d.GetProperty("sigma").GetSingle(), d.GetProperty("sigmaNext").GetSingle(), ReadFloats(d.GetProperty("data"))));
        }

        IBackend backend = new CpuBackend();
        SyntheticPredictor predictor = new(model == "flow" ? PredictionType.FlowVelocity : PredictionType.Epsilon, pattern, guidance);
        QueuedNoise noise = new(shape, draws);
        SamplerOptions options = new()
        {
            PercentToSigma = p => percents.TryGetValue(PercentKey(p), out double v) ? v
                : throw new InvalidOperationException($"No reference percent_to_sigma for {p}."),
            NoiseFactory = (_, _) => noise,
        };
        ISampler s = SamplerRegistry.Create(sampler, sigmas, 7, options);
        using Tensor z = FromHost(shape, ReadFloats(c.GetProperty("x0")));
        s.Reset(shape);
        for (int i = start; i < sigmas.Length - 1; i++)
        {
            s.Step(backend, z, predictor, i);
        }

        float[] expected = ReadFloats(c.GetProperty("final"));
        float[] actual = ToHost(z);
        double scale = 1.0;
        double worst = 0.0;
        for (int k = 0; k < expected.Length; k++)
        {
            scale = Math.Max(scale, Math.Abs(expected[k]));
            worst = Math.Max(worst, Math.Abs(expected[k] - actual[k]));
        }
        Assert.True(worst / scale <= Tolerance, $"{sampler}/{model}/start{start}: max |Δ| {worst:G4} relative {worst / scale:G4}.");

        float[] queries = ReadFloats(c.GetProperty("queries"));
        Assert.Equal(queries.Length, predictor.Queries.Count);
        // dpm_adaptive's step controller amplifies float32-vs-double differences in its error norm into step placement.
        double queryTolerance = sampler == "dpm_adaptive" ? 1e-3 : 1e-4;
        for (int k = 0; k < queries.Length; k++)
        {
            Assert.True(Math.Abs(queries[k] - predictor.Queries[k]) <= queryTolerance * Math.Max(1.0, Math.Abs(queries[k])),
                $"{sampler}/{model}/start{start}: model query {k} at sigma {predictor.Queries[k]}, reference {queries[k]}.");
        }
        Assert.Equal(draws.Count, noise.Used);
    }

    private static string PercentKey(double p) => p switch
    {
        1e-4 => "0.0001",
        0.2 => "0.2",
        0.8 => "0.8",
        _ => p.ToString("R", System.Globalization.CultureInfo.InvariantCulture),
    };

    private static JsonElement FindCase(JsonElement root, string sampler, string model, int start)
    {
        foreach (JsonElement c in root.GetProperty("cases").EnumerateArray())
        {
            if (c.GetProperty("sampler").GetString() == sampler && c.GetProperty("model").GetString() == model
                && c.GetProperty("start").GetInt32() == start)
            {
                return c;
            }
        }
        throw new InvalidOperationException($"No fixture case {sampler}/{model}/{start}.");
    }

    private static TensorShape ReadShape(JsonElement e)
    {
        long[] dims = e.EnumerateArray().Select(v => v.GetInt64()).ToArray();
        return new TensorShape(dims);
    }

    private static float[] ReadFloats(JsonElement e) => e.EnumerateArray().Select(v => v.GetSingle()).ToArray();

    private static unsafe Tensor FromHost(TensorShape shape, float[] data)
    {
        Tensor t = new Tensor(shape, DType.F32);
        new ReadOnlySpan<float>(data).CopyTo(new Span<float>((float*)t.DataPointer, data.Length));
        return t;
    }

    private static unsafe float[] ToHost(Tensor t) => new ReadOnlySpan<float>((float*)t.DataPointer, (int)t.Shape.ElementCount).ToArray();

    /// <summary>The fixture's denoiser: cond and uncond x0 estimates returned as eps/flow predictions.</summary>
    private sealed class SyntheticPredictor(PredictionType type, float[] pattern, float guidance) : IDenoisePredictor
    {
        public List<float> Queries { get; } = [];

        public PredictionType Prediction => type;

        public unsafe DenoisePrediction Predict(Tensor x, float sigma, int stepIndex)
        {
            Queries.Add(sigma);
            Tensor cond = new Tensor(x.Shape, DType.F32);
            Tensor uncond = new Tensor(x.Shape, DType.F32);
            float* px = (float*)x.DataPointer;
            float* pc = (float*)cond.DataPointer;
            float* pu = (float*)uncond.DataPointer;
            float s = sigma;
            for (int k = 0; k < pattern.Length; k++)
            {
                float v = px[k];
                float dc = (v / (1f + (s * s))) + (0.3f * s / (1f + s) * MathF.Tanh(v)) + (0.1f * pattern[k]);
                float du = dc - (0.2f * s / (1f + s) * MathF.Tanh(0.5f * v)) + (0.05f * pattern[k]);
                pc[k] = (v - dc) / s;
                pu[k] = (v - du) / s;
            }
            return new DenoisePrediction(cond, uncond, guidance);
        }
    }

    /// <summary>Returns the reference run's draws in order, checking each is asked for the same interval.</summary>
    private sealed class QueuedNoise(TensorShape shape, List<(float Sigma, float SigmaNext, float[] Data)> draws) : INoiseSource
    {
        public int Used { get; private set; }

        public Tensor Sample(int stepIndex, int subDraw, float sigma, float sigmaNext)
        {
            Assert.True(Used < draws.Count, $"Sampler drew more noise than the reference ({draws.Count}).");
            (float refSigma, float refNext, float[] data) = draws[Used++];
            Assert.True(Math.Abs(refSigma - sigma) <= 1e-4 * Math.Max(1f, refSigma) && Math.Abs(refNext - sigmaNext) <= 1e-4 * Math.Max(1f, refNext),
                $"Noise draw {Used - 1} requested for [{sigma}, {sigmaNext}], reference [{refSigma}, {refNext}].");
            return FromHost(shape, data);
        }

        public void Dispose()
        {
        }
    }
}
