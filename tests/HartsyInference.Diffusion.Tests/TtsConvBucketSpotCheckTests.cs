using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;
using HartsyInference.Audio.Io;
using HartsyInference.Core.Configuration;
using HartsyInference.Cuda;
using HartsyInference.Engine;
using HartsyInference.Engine.Audio;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Writes one arm of the cuDNN length-bucket spot check for one TTS model: the model's sentences synthesized on a
/// fresh 3060 backend with the conv engine chosen per length bucket (<c>on</c>), per exact length (<c>off</c>, today's
/// heuristic), or with no TF32 anywhere (<c>f32</c>: direct F32 conv kernels, full-precision GEMMs; an exact-math reference
/// for models whose output does not go through sampling). Same text, seed and reference audio in every arm, so the arms
/// differ only in conv numerics. Each sentence is synthesized twice: the first time pays the per-length setup, the
/// second is the steady state with every plan cached, and must give the same bytes. Opt-in with <c>HARTSY_TTS_SPOT=1</c>.
///
/// <para><c>HARTSY_TTS_SPOT_MODEL</c> is a speech catalog id (<c>piper</c>, <c>cosyvoice</c>, <c>kyutaitts</c>,
/// <c>csm</c>, <c>orpheus</c>, <c>dia</c>), <c>HARTSY_TTS_SPOT_VARIANT</c> an optional variant,
/// <c>HARTSY_TTS_SPOT_ARM</c> one of on / off / f32, and <c>HARTSY_TTS_SPOT_OUT_DIR</c> where each sentence lands as
/// <c>&lt;model&gt;/&lt;arm&gt;_&lt;nn&gt;.f32</c> (and <c>.wav</c>) with a line in <c>&lt;model&gt;/arms.csv</c>. The audio
/// is compared on the CPU by <c>TtsConvBucketSpotCheckCompareTests</c> in the audio test project.</para></summary>
public sealed class TtsConvBucketSpotCheckTests
{
    private const string GateEnvVar = "HARTSY_TTS_SPOT";
    private const string ModelEnvVar = "HARTSY_TTS_SPOT_MODEL";
    private const string VariantEnvVar = "HARTSY_TTS_SPOT_VARIANT";
    private const string ArmEnvVar = "HARTSY_TTS_SPOT_ARM";
    private const string OutDirEnvVar = "HARTSY_TTS_SPOT_OUT_DIR";
    private const string OrdinalEnvVar = "HARTSY_TTS_SPOT_CUDA_ORDINAL";
    private const int Seed = 1234;
    private const string JfkTranscript =
        "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.";

    /// <summary>The Piper bucket digest's six sentences, so its sentence 3 can be looked at again.</summary>
    private static readonly string[] PiperSentences =
    [
        "Okay.",
        "Please hold while I check.",
        "Thanks for calling, I can see your appointment is booked for Tuesday afternoon at three.",
        "Your package left our warehouse this morning and should arrive at your door by Friday.",
        "I have updated the delivery address on your order, the driver will call you when they are ten minutes away.",
        "The forecast today calls for scattered clouds with a high of seventy two degrees and a light breeze from the northwest.",
    ];

    /// <summary>Two sentences for the sampling models: a short one and a 15-word one.</summary>
    private static readonly string[] Sentences =
    [
        "Please hold while I check.",
        "Thanks for calling, I can see your appointment is booked for Tuesday afternoon at three.",
    ];

    private readonly ITestOutputHelper _out;

    public TtsConvBucketSpotCheckTests(ITestOutputHelper output) => _out = output;

    [Fact]
    [Trait("Category", "GpuIntegration")]
    [Trait("Category", "RealWeights")]
    public async Task SynthesizeOneArm()
    {
        if (Environment.GetEnvironmentVariable(GateEnvVar) != "1")
        {
            _out.WriteLine($"SKIPPED: set {GateEnvVar}=1 to write a length-bucket spot-check arm.");
            return;
        }
        string model = Required(ModelEnvVar);
        string arm = Required(ArmEnvVar);
        string outDir = Path.Combine(Required(OutDirEnvVar), model);
        string variant = Environment.GetEnvironmentVariable(VariantEnvVar) ?? "";
        Assert.Contains(arm, (string[])["on", "off", "f32"]);
        Directory.CreateDirectory(outDir);

        string[] texts = model == "piper" ? PiperSentences : Sentences;
        float[]? reference = null;
        if (model == "cosyvoice")
        {
            string jfk = Path.Combine(RepoPaths.RepoRoot(), "tests", "python-reference", "silerovad_reference", "jfk.wav");
            Assert.True(File.Exists(jfk), $"missing {jfk}");
            WavFile.DecodedAudio decoded = WavFile.Read(jfk);
            reference = Resampler.Create(decoded.SampleRate, 24_000).Resample(decoded.ToMono());
        }

        SetArm(arm);
        try
        {
            using CudaBackend backend = Open3060();
            TtsModelDescriptor descriptor = TtsCatalog.Resolve(model);
            Stopwatch load = Stopwatch.StartNew();
            using ITtsRunner runner = await descriptor.LoadAsync(new TtsLoadContext { Backend = backend }, variant, CancellationToken.None);
            _out.WriteLine($"{model}{(variant.Length > 0 ? ":" + variant : "")} loaded in {load.Elapsed.TotalSeconds:F1}s, arm {arm}");
            runner.Synthesize(backend, Job("Warm up.", reference));

            StringBuilder csv = new StringBuilder();
            for (int i = 0; i < texts.Length; i++)
            {
                CudnnConvPlanStats before = backend.CudnnConvPlanStats;
                long start = Stopwatch.GetTimestamp();
                float[] wave = runner.Synthesize(backend, Job(texts[i], reference));
                double ms = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
                CudnnConvPlanStats after = backend.CudnnConvPlanStats;
                long again = Stopwatch.GetTimestamp();
                float[] repeat = runner.Synthesize(backend, Job(texts[i], reference));
                double repeatMs = Stopwatch.GetElapsedTime(again).TotalMilliseconds;
                Assert.True(repeat.AsSpan().SequenceEqual(wave), $"{model} {arm} sentence {i + 1}: a repeat gave different audio");
                string name = $"{arm}_{i + 1:D2}";
                File.WriteAllBytes(Path.Combine(outDir, name + ".f32"), MemoryMarshal.AsBytes<float>(wave).ToArray());
                WavFile.WriteMono16(Path.Combine(outDir, name + ".wav"), wave, runner.SampleRate);
                string line = string.Join(',', arm, (i + 1).ToString(CultureInfo.InvariantCulture),
                    runner.SampleRate.ToString(CultureInfo.InvariantCulture), wave.Length.ToString(CultureInfo.InvariantCulture),
                    ms.ToString("F1", CultureInfo.InvariantCulture), repeatMs.ToString("F1", CultureInfo.InvariantCulture),
                    (after.PlanBuilds - before.PlanBuilds).ToString(CultureInfo.InvariantCulture),
                    (after.BucketPlanBuilds - before.BucketPlanBuilds).ToString(CultureInfo.InvariantCulture),
                    (after.ReferenceBuilds - before.ReferenceBuilds).ToString(CultureInfo.InvariantCulture),
                    '"' + texts[i].Replace("\"", "'") + '"');
                csv.AppendLine(line);
                _out.WriteLine(line);
            }
            File.AppendAllText(Path.Combine(outDir, "arms.csv"), csv.ToString());
            _out.WriteLine($"cuDNN conv: {backend.CudnnConvPlanStats}");
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.AudioConvLengthBuckets);
            KnobStore.Clear(EngineKnobs.AudioConvCudnn);
            KnobStore.Clear(EngineKnobs.HighPrecisionGemm);
            KnobStore.Clear(EngineKnobs.NoTf32);
        }
    }

    private static TtsJob Job(string text, float[]? reference) => reference is null
        ? new TtsJob { Text = text, Seed = Seed }
        : new TtsJob { Text = text, Seed = Seed, ReferenceMono24k = reference, RefText = JfkTranscript };

    /// <summary>on / off pick the conv engine per length bucket or per exact length; f32 turns every TF32 path off (direct
    /// F32 conv kernels instead of cuDNN, full-precision GEMMs).</summary>
    private static void SetArm(string arm)
    {
        KnobStore.Set(EngineKnobs.AudioConvLengthBuckets, arm == "on");
        if (arm == "f32")
        {
            KnobStore.Set(EngineKnobs.AudioConvCudnn, false);
            KnobStore.Set(EngineKnobs.HighPrecisionGemm, true);
            KnobStore.Set(EngineKnobs.NoTf32, true);
        }
    }

    private CudaBackend Open3060()
    {
        string? ordinalText = Environment.GetEnvironmentVariable(OrdinalEnvVar);
        int ordinal = string.IsNullOrEmpty(ordinalText) ? 0 : int.Parse(ordinalText, CultureInfo.InvariantCulture);
        Assert.True(CudaContext.IsAvailable(), $"CUDA unavailable: {CudaContext.LastUnavailableReason}");
        string? ptx = BackendGate.KernelDir("Ptx", "HartsyInference.Cuda");
        Assert.False(ptx is null, "no compiled PTX directory beside the tests or in the repo");
        CudaBackend backend = new CudaBackend(ordinal, ptx);
        _out.WriteLine($"CUDA ordinal {ordinal}: {backend.Capabilities.DeviceName}");
        if (!backend.Capabilities.DeviceName.Contains("3060", StringComparison.Ordinal))
        {
            backend.Dispose();
            Assert.Fail($"ordinal {ordinal} is not a 3060; set {OrdinalEnvVar}");
        }
        return backend;
    }

    private static string Required(string name) =>
        Environment.GetEnvironmentVariable(name) is { Length: > 0 } value ? value : throw new InvalidOperationException($"{name} is not set");
}
