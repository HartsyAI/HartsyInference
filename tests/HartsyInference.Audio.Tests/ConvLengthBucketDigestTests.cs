using System.Diagnostics;
using System.Globalization;
using System.Text;
using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Configuration;
using HartsyInference.Cuda;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>The length-bucketed cuDNN engine choice (<c>numerics.audioConvLengthBuckets</c>) on a second conv-heavy audio
/// model: Piper, whose VITS duration predictor, flow and HiFi-GAN vocoder (transposed-conv upsamplers, dilated residual
/// convs) all run as 1D convs. Seeded, so its audio is deterministic. Opt-in with <c>HARTSY_CONV_BUCKET_DIGEST=1</c>; the
/// device must be a 3060.
///
/// <para>Three arms, each on a fresh backend so no plan or engine choice carries over: buckets off (today's heuristic per
/// length), buckets on, and buckets on with the sentences in reverse order. Every sentence is a length the backend has not
/// seen, so the per-sentence time is the first-synthesis time. Per sentence it reports each arm's digest and time, and the
/// on-against-off comparison (byte identity, else log-spectral correlation and max-abs; a duration predictor can also move
/// the length). It asserts what the design guarantees: the two bucket arms are byte-identical, whatever order the lengths
/// arrive in. Tables go to the test log and, with <c>HARTSY_CONV_BUCKET_OUT</c>, are appended to that file.</para></summary>
[Trait("Category", "GpuIntegration")]
[Trait("Category", "RealWeights")]
public sealed class ConvLengthBucketDigestTests
{
    private const string GateEnvVar = "HARTSY_CONV_BUCKET_DIGEST";
    private const string OrdinalEnvVar = "HARTSY_CONV_BUCKET_CUDA_ORDINAL";
    private const string OutEnvVar = "HARTSY_CONV_BUCKET_OUT";
    private const string PiperVoice = "en_US-ryan-medium";
    private const int Seed = 1234;

    /// <summary>Six sentences of different lengths, so their convs land in several length buckets.</summary>
    private static readonly string[] Sentences =
    [
        "Okay.",
        "Please hold while I check.",
        "Thanks for calling, I can see your appointment is booked for Tuesday afternoon at three.",
        "Your package left our warehouse this morning and should arrive at your door by Friday.",
        "I have updated the delivery address on your order, the driver will call you when they are ten minutes away.",
        "The forecast today calls for scattered clouds with a high of seventy two degrees and a light breeze from the northwest.",
    ];

    private readonly ITestOutputHelper _out;

    public ConvLengthBucketDigestTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void Piper_BucketedEngineChoice_DigestsAgainstPerLengthHeuristic()
    {
        if (Environment.GetEnvironmentVariable(GateEnvVar) != "1")
        {
            _out.WriteLine($"SKIPPED: set {GateEnvVar}=1 to run the length-bucket digest A/B.");
            return;
        }
        string onnx = Path.Combine(AudioModelCache.GetRepoDirectory("rhasspy/piper-voices", "tts"),
            "en", "en_US", "ryan", "medium", PiperVoice + ".onnx");
        if (!RealWeightGate.Require(_out.WriteLine, onnx, onnx + ".json"))
        {
            return;
        }
        using PiperPipeline piper = PiperPipeline.LoadFromFiles(onnx, onnx + ".json");

        Arm off = RunArm(piper, buckets: false, reverse: false);
        Arm on = RunArm(piper, buckets: true, reverse: false);
        Arm onReverse = RunArm(piper, buckets: true, reverse: true);

        StringBuilder table = new StringBuilder();
        table.AppendLine($"### cuDNN conv length buckets — Piper {PiperVoice}, seed {Seed}, each arm on a fresh 3060 backend");
        table.AppendLine();
        table.AppendLine("| Arm | conv plans (from bucket) | bucket references | heuristic ms | total synth ms |");
        table.AppendLine("|---|---|---:|---:|---:|");
        foreach (Arm arm in new[] { off, on, onReverse })
        {
            table.AppendLine($"| {arm.Name} | {arm.Stats.PlanBuilds} ({arm.Stats.BucketPlanBuilds}) | {arm.Stats.ReferenceBuilds} "
                + $"| {arm.Stats.HeuristicMs:F1} | {arm.Ms.Sum():F1} |");
        }
        table.AppendLine();
        table.AppendLine("| # | samples | off ms | on ms | on, reverse ms | sha off | sha on | sha on, reverse | on vs off |");
        table.AppendLine("|---:|---:|---:|---:|---:|---|---|---|---|");
        int identical = 0;
        double lowestLogSpec = 1.0;
        for (int i = 0; i < Sentences.Length; i++)
        {
            float[] a = off.Waves[i], b = on.Waves[i];
            string versus;
            if (a.Length != b.Length)
            {
                versus = $"length {a.Length} → {b.Length}";
            }
            else if (a.AsSpan().SequenceEqual(b))
            {
                versus = "identical";
                identical++;
            }
            else
            {
                double logSpec = AudioParityMetrics.LogSpectralCorrelation(a, b);
                (double maxAbs, double _) = AudioParityMetrics.Compare(a, b);
                lowestLogSpec = Math.Min(lowestLogSpec, logSpec);
                versus = $"log-spec {logSpec:F6}, max-abs {maxAbs:E2}";
            }
            table.AppendLine($"| {i + 1} | {b.Length} | {off.Ms[i]:F1} | {on.Ms[i]:F1} | {onReverse.Ms[i]:F1} "
                + $"| `{Digest(a)}` | `{Digest(b)}` | `{Digest(onReverse.Waves[i])}` | {versus} |");
        }
        table.AppendLine();
        table.AppendLine($"Buckets on against off: {identical} of {Sentences.Length} sentences byte-identical"
            + (identical == Sentences.Length ? "." : $"; lowest log-spectral correlation among same-length others {lowestLogSpec:F6}."));
        GpuBenchSupport.Emit(_out.WriteLine, table, OutEnvVar);

        for (int i = 0; i < Sentences.Length; i++)
        {
            Assert.True(on.Waves[i].AsSpan().SequenceEqual(onReverse.Waves[i]),
                $"sentence {i + 1}: the bucketed audio depends on the order lengths arrived in");
        }
    }

    /// <summary>Speaks every sentence once on a fresh backend with buckets on or off, after one warm-up synthesis.</summary>
    private Arm RunArm(PiperPipeline piper, bool buckets, bool reverse)
    {
        string name = $"buckets {(buckets ? "on" : "off")}{(reverse ? ", reverse order" : "")}";
        KnobStore.Set(EngineKnobs.AudioConvLengthBuckets, buckets);
        try
        {
            using CudaBackend backend = GpuBenchSupport.Open3060(_out.WriteLine, OrdinalEnvVar);
            piper.SynthesizeText(backend, "Warm up.", seed: Seed);
            CudnnConvPlanStats before = backend.CudnnConvPlanStats;
            float[][] waves = new float[Sentences.Length][];
            double[] ms = new double[Sentences.Length];
            for (int position = 0; position < Sentences.Length; position++)
            {
                int index = reverse ? Sentences.Length - 1 - position : position;
                long start = Stopwatch.GetTimestamp();
                waves[index] = piper.SynthesizeText(backend, Sentences[index], seed: Seed);
                ms[index] = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            }
            CudnnConvPlanStats after = backend.CudnnConvPlanStats;
            CudnnConvPlanStats delta = new(after.Executions - before.Executions, after.PlanBuilds - before.PlanBuilds,
                after.BucketPlanBuilds - before.BucketPlanBuilds, after.BucketFallbacks - before.BucketFallbacks,
                after.ReferenceBuilds - before.ReferenceBuilds, after.CachedPlans, after.BuildMs - before.BuildMs,
                after.GraphMs - before.GraphMs, after.HeuristicMs - before.HeuristicMs, after.FinalizeMs - before.FinalizeMs,
                after.ConfigsTried - before.ConfigsTried, after.RuntimeCompiledBuilds - before.RuntimeCompiledBuilds);
            _out.WriteLine($"{name}: {delta}");
            return new Arm(name, waves, ms, delta);
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.AudioConvLengthBuckets);
        }
    }

    private static string Digest(float[] wave) => AudioParityMetrics.Sha256Hex(wave)[..12];

    private sealed record Arm(string Name, float[][] Waves, double[] Ms, CudnnConvPlanStats Stats);
}
