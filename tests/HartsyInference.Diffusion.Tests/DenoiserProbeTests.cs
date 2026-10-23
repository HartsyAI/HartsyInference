using System.Globalization;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Sampling;
using HartsyInference.Diffusion.Schedulers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The teacher-forced denoiser probe on a synthetic denoiser: off, it changes nothing; its record round-trips bit for bit; a
/// replay against itself reads exactly zero and leaves the arm's own trajectory alone; a perturbed denoiser reads a nonzero
/// distance that repeats exactly; and a replay that cannot be teacher-forced is refused.</summary>
public sealed unsafe class DenoiserProbeTests : IDisposable
{
    private static readonly TensorShape Shape = new(1, 4, 6, 6);
    private readonly string _dir = Path.Combine(Path.GetTempPath(), $"denoiser-probe-{Guid.NewGuid():N}");

    public void Dispose()
    {
        if (Directory.Exists(_dir)) Directory.Delete(_dir, recursive: true);
    }

    /// <summary>A CFG pair that depends on the input and the step: cond <c>0.5·x + 0.1·sin(i + s)</c>, uncond <c>0.25·x</c>, plus
    /// <c>perturbation·x</c> on cond.</summary>
    private sealed class PairDenoiser(float perturbation = 0f) : IDenoisePredictor
    {
        public int Evaluations { get; private set; }

        public PredictionType Prediction => PredictionType.FlowVelocity;

        public DenoisePrediction Predict(Tensor x, float sigma, int stepIndex)
        {
            Evaluations++;
            Tensor cond = new(x.Shape, DType.F32), uncond = new(x.Shape, DType.F32);
            float* src = (float*)x.DataPointer, c = (float*)cond.DataPointer, u = (float*)uncond.DataPointer;
            for (long i = 0; i < x.ElementCount; i++)
            {
                c[i] = (0.5f + perturbation) * src[i] + 0.1f * MathF.Sin(i + stepIndex);
                u[i] = 0.25f * src[i];
            }
            return new DenoisePrediction(cond, uncond, 3.0f);
        }
    }

    private static float[] Sigmas(int steps, float scale = 1f)
    {
        float[] sigmas = new float[steps + 1];
        for (int i = 0; i < steps; i++) sigmas[i] = scale * (1.0f - (float)i / steps);
        return sigmas;
    }

    private static Tensor Noise(int seed)
    {
        Tensor t = new(Shape, DType.F32);
        Random rng = new(seed);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private IDisposable Probe(string mode, string? steps = null, string? outDir = null, bool graphs = false)
    {
        KnobProfile profile = KnobProfile.Create("denoiser-probe-test").With(EngineKnobs.DenoiserProbe, mode)
            .With(EngineKnobs.DenoiserProbeDir, _dir).With(EngineKnobs.DitGraph, graphs);
        if (steps is not null) profile = profile.With(EngineKnobs.DenoiserProbeSteps, steps);
        if (outDir is not null) profile = profile.With(EngineKnobs.DenoiserProbeOut, outDir);
        return profile.Push();
    }

    /// <summary>Runs the whole loop and returns the final latent's bits.</summary>
    private static uint[] Run(IDenoisePredictor denoiser, float[] sigmas, string samplerName = "euler")
    {
        using IBackend backend = new CpuBackend();
        ISampler sampler = SamplerRegistry.Create(samplerName, sigmas, seed: 3);
        using Tensor z = Noise(9);
        sampler.Reset(Shape);
        for (int i = 0; i < sigmas.Length - 1; i++) sampler.Step(backend, z, denoiser, i);
        return z.AsReadOnlySpan<uint>().ToArray();
    }

    private static List<string[]> Rows(string outDir) =>
        [.. File.ReadAllLines(Path.Combine(outDir, "probe.tsv")).Skip(1).Select(line => line.Split('\t'))];

    [Fact]
    public void Off_TheRegistryHandsBackTheBareSampler_AndConditioningIsUntouched()
    {
        Assert.False(DenoiserProbe.Active);
        Assert.Null(DenoiserProbe.Open(8));
        Assert.IsType<EulerSampler>(SamplerRegistry.Create("euler", Sigmas(4), seed: 1));
        using Tensor text = Noise(5);
        uint[] before = text.AsReadOnlySpan<uint>().ToArray();
        DenoiserProbe.Condition("text", text);
        Assert.Equal(before, text.AsReadOnlySpan<uint>().ToArray());
        Assert.False(Directory.Exists(_dir));
    }

    [Fact]
    public void WhileProbing_StepGraphsStayOff()
    {
        using (KnobProfile.Create("graphs").With(EngineKnobs.DitGraph, true).Push())
            Assert.True(DitStepGraph.Enabled && DitStepGraph.EnabledDefaultOn);
        using (Probe("record", graphs: true))
            Assert.False(DitStepGraph.Enabled || DitStepGraph.EnabledDefaultOn);
    }

    [Fact]
    public void TheRecord_RoundTripsBitForBit_InEveryActivationDType()
    {
        DType[] dtypes = [DType.F32, DType.F16, DType.BF16];
        Tensor[] inputs = [.. dtypes.Select((dt, i) => Noise(20 + i).CastTo(dt))];
        Tensor?[] outputs = [Noise(30).CastTo(DType.BF16), Noise(31).CastTo(DType.F16)];
        const float sigma = 0.123456789f;
        try
        {
            using (Probe("record"))
            {
                DenoiserProbe probe = DenoiserProbe.Open(3)!;
                Assert.True(probe.Takes(1));
                probe.RecordInputs(1, sigma, inputs);
                probe.RecordOutputs(1, 2.5f, ["cond", "uncond"], outputs);
            }
            string outDir = Path.Combine(_dir, "self");
            using (Probe("replay", outDir: outDir))
            {
                DenoiserProbe probe = DenoiserProbe.Open(3)!;
                Assert.True(probe.Takes(1));
                Tensor[] recorded = probe.ReplayInputs(1, sigma, inputs);
                for (int i = 0; i < inputs.Length; i++)
                {
                    Assert.Equal(inputs[i].DType, recorded[i].DType);
                    Assert.Equal(inputs[i].AsReadOnlySpan<byte>().ToArray(), recorded[i].AsReadOnlySpan<byte>().ToArray());
                }
                probe.ReplayOutputs(1, [DenoiserProbe.Copy(outputs[0]!), DenoiserProbe.Copy(outputs[1]!)]);
            }
            List<string[]> rows = Rows(outDir);
            Assert.Equal(["cond", "uncond", "cfg"], rows.Select(r => r[2]));
            Assert.All(rows, r => Assert.Equal(0.0, double.Parse(r[3], CultureInfo.InvariantCulture)));
            Assert.All(rows, r => Assert.Equal(sigma, float.Parse(r[1], CultureInfo.InvariantCulture)));
        }
        finally
        {
            foreach (Tensor t in inputs) t.Dispose();
            foreach (Tensor? t in outputs) t?.Dispose();
        }
    }

    [Fact]
    public void ALoopReplayedOnItself_ReadsExactlyZero_AndKeepsItsOwnTrajectory()
    {
        float[] sigmas = Sigmas(8);
        uint[] plain = Run(new PairDenoiser(), sigmas);
        uint[] recorded;
        using (Probe("record"))
            recorded = Run(new PairDenoiser(), sigmas);
        string outDir = Path.Combine(_dir, "self");
        PairDenoiser replaying = new();
        uint[] replayed;
        using (Probe("replay", outDir: outDir))
            replayed = Run(replaying, sigmas);

        Assert.Equal(plain, recorded);
        Assert.Equal(plain, replayed);
        Assert.Equal(8 + 3, replaying.Evaluations);   // one extra forward per probed step
        List<string[]> rows = Rows(outDir);
        Assert.Equal([0, 0, 0, 4, 4, 4, 7, 7, 7], rows.Select(r => int.Parse(r[0], CultureInfo.InvariantCulture)));
        Assert.All(rows, r => Assert.Equal(0.0, double.Parse(r[3], CultureInfo.InvariantCulture)));
        Assert.All(rows, r => Assert.Equal(1.0, double.Parse(r[4], CultureInfo.InvariantCulture), 12));
    }

    [Fact]
    public void ASecondOrderSampler_IsProbedOnItsFirstEvaluationOnly_AndKeepsItsTrajectory()
    {
        float[] sigmas = Sigmas(8);
        PairDenoiser plainDenoiser = new();
        uint[] plain = Run(plainDenoiser, sigmas, "heun");
        using (Probe("record"))
            Run(new PairDenoiser(), sigmas, "heun");
        string outDir = Path.Combine(_dir, "heun");
        PairDenoiser replaying = new();
        uint[] replayed;
        using (Probe("replay", outDir: outDir))
            replayed = Run(replaying, sigmas, "heun");

        Assert.Equal(plain, replayed);
        Assert.Equal(plainDenoiser.Evaluations + 3, replaying.Evaluations);
        List<string[]> rows = Rows(outDir);
        Assert.Equal(9, rows.Count);
        Assert.All(rows, r => Assert.Equal(0.0, double.Parse(r[3], CultureInfo.InvariantCulture)));
    }

    [Fact]
    public void APerturbedDenoiser_ReadsANonzeroDistance_ThatRepeatsExactly()
    {
        float[] sigmas = Sigmas(8);
        using (Probe("record"))
            Run(new PairDenoiser(), sigmas);
        string first = Path.Combine(_dir, "first"), second = Path.Combine(_dir, "second");
        using (Probe("replay", outDir: first))
            Run(new PairDenoiser(1e-3f), sigmas);
        using (Probe("replay", outDir: second))
            Run(new PairDenoiser(1e-3f), sigmas);

        List<string[]> rows = Rows(first);
        Assert.All(rows.Where(r => r[2] != "uncond"), r => Assert.True(double.Parse(r[3], CultureInfo.InvariantCulture) > 1e-4,
            $"step {r[0]} {r[2]} read {r[3]}"));
        Assert.All(rows.Where(r => r[2] == "uncond"), r => Assert.Equal(0.0, double.Parse(r[3], CultureInfo.InvariantCulture)));
        Assert.Equal(File.ReadAllText(Path.Combine(first, "probe.tsv")), File.ReadAllText(Path.Combine(second, "probe.tsv")));
    }

    [Fact]
    public void AReplayThatCannotBeTeacherForced_IsRefused()
    {
        using (Probe("record"))
            Run(new PairDenoiser(), Sigmas(8));
        using (Probe("replay"))
            Assert.Throws<HartsyInferenceException>(() => Run(new PairDenoiser(), Sigmas(8, scale: 0.5f)));
        using (Probe("replay"))
        {
            DenoiserProbe probe = DenoiserProbe.Open(8)!;
            Assert.True(probe.Takes(0));
            using Tensor wrong = new(new TensorShape(1, 4, 6, 7), DType.F32);
            Assert.Throws<HartsyInferenceException>(() => probe.ReplayInputs(0, Sigmas(8)[0], [wrong]));
        }
        using (Probe("replay", steps: "3"))
            Assert.Throws<HartsyInferenceException>(() => Run(new PairDenoiser(), Sigmas(8)));
    }

    [Fact]
    public void Conditioning_IsRecorded_AndReplayedInPlace()
    {
        using Tensor recorded = Noise(40), other = Noise(41), wrongShape = new(new TensorShape(1, 4, 6, 7), DType.F32);
        using (Probe("record"))
            DenoiserProbe.Condition("text", recorded);
        using (Probe("replay"))
        {
            DenoiserProbe.Condition("text", other);
            Assert.Equal(recorded.AsReadOnlySpan<uint>().ToArray(), other.AsReadOnlySpan<uint>().ToArray());
            Assert.Throws<HartsyInferenceException>(() => DenoiserProbe.Condition("text", wrongShape));
            Assert.Throws<HartsyInferenceException>(() => DenoiserProbe.Condition("pooled", other));
        }
    }

    [Fact]
    public void PerLayerConditioning_IsRecorded_AndReplayedInPlace()
    {
        using Tensor a = Noise(50), b = Noise(51), c = Noise(52), d = Noise(53);
        using (Probe("record"))
            DenoiserProbe.Condition("layers", (Tensor[])[a, b]);
        using (Probe("replay"))
        {
            DenoiserProbe.Condition("layers", (Tensor[])[c, d]);
            Assert.Equal(a.AsReadOnlySpan<uint>().ToArray(), c.AsReadOnlySpan<uint>().ToArray());
            Assert.Equal(b.AsReadOnlySpan<uint>().ToArray(), d.AsReadOnlySpan<uint>().ToArray());
        }
    }

    [Fact]
    public void Refuse_ThrowsOnlyWhenTheProbeIsActive()
    {
        DenoiserProbe.Refuse(true, "ControlNet");
        using (Probe("record"))
        {
            DenoiserProbe.Refuse(false, "ControlNet");
            Assert.Throws<HartsyInferenceException>(() => DenoiserProbe.Refuse(true, "ControlNet"));
        }
    }

    /// <summary>Every pipeline that drives a sampler taps its conditioning, so a probe arm whose knob reaches a text encoder still
    /// compares the denoiser alone. A presence check only: it cannot tell whether every stream the forward reads is tapped. Lance
    /// embeds its own token ids inside the denoiser and has no conditioning tensor.</summary>
    [Fact]
    public void EveryPipelineThatDrivesASampler_TapsItsConditioning()
    {
        string src = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src");
        List<string> missing = [];
        foreach (string file in Directory.EnumerateFiles(src, "*Pipeline.cs", SearchOption.AllDirectories))
        {
            string text = File.ReadAllText(file);
            if (text.Contains("sampler.Step(", StringComparison.Ordinal) && !text.Contains("DenoiserProbe.Condition(", StringComparison.Ordinal)
                && !Path.GetFileName(file).StartsWith("LanceVideo", StringComparison.Ordinal))
                missing.Add(Path.GetFileName(file));
        }
        Assert.True(missing.Count == 0, "No conditioning tap in: " + string.Join(", ", missing));
    }
}
