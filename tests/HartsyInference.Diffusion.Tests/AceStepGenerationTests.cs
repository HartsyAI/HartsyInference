using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Music;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tests.Common;
using HartsyInference.Tokenizers;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real-checkpoint ACE-Step v1 tests, two tiers. <c>LoadSmoke</c> is cheap (mmap-borrowed BF16, no big
/// allocations) and validates every loader key against <c>ACE-Step/ACE-Step-v1-3.5B</c>; it skips cleanly when
/// <c>ACE_STEP_DIR</c> is absent. <c>Generate</c> runs the full pipeline on CPU — it casts the DiT to F32
/// (~13 GB RSS peak) and takes minutes, so it additionally requires <c>HARTSYINFERENCE_RUN_ACE_E2E=1</c>; run it from
/// a bare terminal, not inside an IDE test runner:
/// <code>HARTSYINFERENCE_RUN_ACE_E2E=1 dotnet test tests/HartsyInference.Diffusion.Tests -f net10.0 --filter AceStepGenerationTests.Generate</code>
/// Output lands in <c>Output/ace_step_*.wav</c>. Numeric parity vs the Python reference is still
/// validation-pending — this proves the plumbing end-to-end on real weights.</summary>
public unsafe class AceStepGenerationTests
{
    private readonly ITestOutputHelper _output;
    public AceStepGenerationTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void LoadSmoke_AllComponentsResolveRealKeys()
    {
        string root = TestPaths.AceStep.RootDir;
        if (!Directory.Exists(root)) { _output.WriteLine($"SKIPPED: ACE-Step checkpoint not found at {root} (set ACE_STEP_DIR)."); return; }

        AceStepLyricTokenizer tokenizer = new(TestPaths.AceStep.LyricVocabPath, TestPaths.AceStep.LyricMergesPath);
        Assert.Equal(6693, tokenizer.VocabSize);
        int[] ids = tokenizer.TokenizeLyrics("[verse]\nhello world");
        Assert.Equal(AceStepLyricTokenizer.StartToken, ids[0]);
        Assert.Contains(6683, ids);   // [verse] structure token

        (Dictionary<string, Tensor> dcaeW, SafeTensorsLoader dcaeLoader) = AceStepCheckpointConverter.LoadDcae(TestPaths.AceStep.DcaePath);
        using (dcaeLoader)
        {
            MusicDcaeDecoder dcae = new();
            dcae.LoadWeights(dcaeW);
        }

        (Dictionary<string, Tensor> vocW, SafeTensorsLoader vocLoader) = AceStepCheckpointConverter.LoadVocoder(TestPaths.AceStep.VocoderPath);
        using (vocLoader)
        {
            AdaMosHiFiGanV1 vocoder = new();
            vocoder.LoadWeights(vocW);
        }

        (Dictionary<string, Tensor> ditW, SafeTensorsLoader ditLoader) = AceStepCheckpointConverter.LoadTransformer(TestPaths.AceStep.TransformerPath);
        using (ditLoader)
        {
            using AceStepDit dit = new(AceStepConfig.V1);
            dit.LoadWeights(ditW);
            long paramCount = dit.EnumerateWeights().Sum(t => t.Shape.ElementCount);
            _output.WriteLine($"DiT loaded: {paramCount / 1e9:F2}B params");
            Assert.True(paramCount > 2.0e9);
        }
    }

    [Fact]
    public void Generate_ShortInstrumental_WritesWav()
    {
        string root = TestPaths.AceStep.RootDir;
        if (!Directory.Exists(root)) { _output.WriteLine($"SKIPPED: ACE-Step checkpoint not found at {root}."); return; }
        if (Environment.GetEnvironmentVariable("HARTSYINFERENCE_RUN_ACE_E2E") != "1")
        { _output.WriteLine("SKIPPED: set HARTSYINFERENCE_RUN_ACE_E2E=1 (casts the 3.5B DiT to F32, ~13 GB RSS)."); return; }

        CpuBackend backend = new();

        // UMT5-base style prompt features.
        T5Tokenizer t5Tokenizer = T5Tokenizer.CreateUmt5(maxLength: 256);
        T5TextEncoder umt5 = new(T5TextEncoderConfig.Umt5Base);
        SafeTensorsLoader umt5Loader = new();
        umt5Loader.Load(Path.Combine(TestPaths.AceStep.Umt5BaseDir, "model.safetensors"));
        umt5.LoadWeights(umt5Loader.GetAllTensors());
        int[] promptIds = t5Tokenizer.Encode("lo-fi hip hop, mellow piano, calm, 90 BPM");
        Tensor textBatch = umt5.Encode(backend, [promptIds]);
        Tensor textEmbeds = CfgHelper.SliceBatchElement(textBatch, 0, promptIds.Length, 768);
        textBatch.Dispose();
        umt5Loader.Dispose();
        _output.WriteLine($"UMT5: [{textEmbeds.Shape[0]}, {textEmbeds.Shape[1]}]");

        AceStepLyricTokenizer lyricTokenizer = new(TestPaths.AceStep.LyricVocabPath, TestPaths.AceStep.LyricMergesPath);
        int[] lyricIds = lyricTokenizer.TokenizeLyrics("[inst]");

        (Dictionary<string, Tensor> ditW, SafeTensorsLoader ditLoader) = AceStepCheckpointConverter.LoadTransformer(TestPaths.AceStep.TransformerPath, castToF32: true);
        AceStepDit dit = new(AceStepConfig.V1);
        dit.LoadWeights(ditW);
        ditLoader.Dispose();

        (Dictionary<string, Tensor> dcaeW, SafeTensorsLoader dcaeLoader) = AceStepCheckpointConverter.LoadDcae(TestPaths.AceStep.DcaePath, castToF32: true);
        MusicDcaeDecoder dcae = new();
        dcae.LoadWeights(dcaeW);
        dcaeLoader.Dispose();

        (Dictionary<string, Tensor> vocW, SafeTensorsLoader vocLoader) = AceStepCheckpointConverter.LoadVocoder(TestPaths.AceStep.VocoderPath, castToF32: true);
        AdaMosHiFiGanV1 vocoder = new();
        vocoder.LoadWeights(vocW);
        vocLoader.Dispose();

        AceStepPipeline pipeline = new(backend, dit, dcae, vocoder, AceStepConfig.V1);
        (float[] left, float[] right, int rate, int seed) = pipeline.Generate(
            textEmbeds, lyricIds, durationSeconds: 10, steps: 27, guidance: 7f,
            guidanceMode: AceStepPipeline.GuidanceMode.Cfg, seed: 0,
            onProgress: p => _output.WriteLine($"step {p.Step}/{p.TotalSteps} ({p.ElapsedMs / 1000:F0}s)"));

        Directory.CreateDirectory(TestPaths.OutputDir);
        string wavPath = Path.Combine(TestPaths.OutputDir, $"ace_step_seed{seed}.wav");
        WriteWavStereo(wavPath, left, right, rate);
        _output.WriteLine($"Wrote {wavPath} ({left.Length / (double)rate:F1}s)");
        Assert.True(left.Length > rate * 5);
    }

    private static void WriteWavStereo(string path, float[] left, float[] right, int sampleRate)
    {
        int samples = left.Length;
        using BinaryWriter wr = new(File.Create(path));
        int dataBytes = samples * 2 * 2;
        wr.Write("RIFF"u8); wr.Write(36 + dataBytes); wr.Write("WAVE"u8);
        wr.Write("fmt "u8); wr.Write(16); wr.Write((short)1); wr.Write((short)2);
        wr.Write(sampleRate); wr.Write(sampleRate * 4); wr.Write((short)4); wr.Write((short)16);
        wr.Write("data"u8); wr.Write(dataBytes);
        for (int i = 0; i < samples; i++)
        {
            wr.Write((short)Math.Clamp(left[i] * 32767f, short.MinValue, short.MaxValue));
            wr.Write((short)Math.Clamp(right[i] * 32767f, short.MinValue, short.MaxValue));
        }
    }
}
