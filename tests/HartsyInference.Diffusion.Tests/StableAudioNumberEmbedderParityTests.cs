using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real-weight parity for <see cref="StableAudioNumberEmbedder"/> (the <c>seconds_total</c> timing
/// conditioner) vs the real <c>stable_audio_tools.models.conditioners.NumberConditioner</c> math (inlined,
/// not imported, in <c>gen_timing_reference.py</c> to avoid that package's heavy unrelated deps — see its
/// docstring) loaded with the real checkpoint. Skip-guarded when the checkpoint/fixtures are absent.</summary>
[Trait("Category", "Integration")]
public unsafe class StableAudioNumberEmbedderParityTests
{
    private readonly ITestOutputHelper _output;
    public StableAudioNumberEmbedderParityTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Embed_MatchesNumberConditionerReference_OnRealWeights()
    {
        string ckptPath = System.IO.Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".cache", "hartsyinference", "models", "stable-audio-open-small", "conditioner",
            "diffusion_pytorch_model.safetensors");
        string fixtureDir = System.IO.Path.Combine(RepoRoot.Path, "tests", "python-reference",
            "stable_audio_open_small_parity");

        if (!File.Exists(ckptPath) || !File.Exists(System.IO.Path.Combine(fixtureDir, "timing_embed_ref.bin")))
        {
            _output.WriteLine("Checkpoint or reference fixtures not present — skipping.");
            return;
        }

        const float Value = 11.89f;
        StableAudioDitConfig cfg = StableAudioDitConfig.OpenSmall;

        using SafeTensorsLoader loader = new();
        loader.Load(ckptPath);
        Dictionary<string, Tensor> weights = loader.GetAllTensors();

        StableAudioNumberEmbedder embedder = new(minVal: 0f, maxVal: (float)cfg.TimingMaxSeconds);
        embedder.LoadWeights(weights, "conditioners.seconds_total");
        using CpuBackend backend = new();

        Tensor embed = embedder.Embed(backend, Value);
        Assert.Equal(new TensorShape(1, 1, cfg.CondTokenDim), embed.Shape);

        Tensor reference = ReadBin(System.IO.Path.Combine(fixtureDir, "timing_embed_ref.bin"), new TensorShape(1, 1, cfg.CondTokenDim));

        (float cosine, float maxAbsDiff) = Compare(embed, reference);
        embed.Dispose();
        reference.Dispose();

        _output.WriteLine($"cosine={cosine:F6} maxAbsDiff={maxAbsDiff:E4}");
        Assert.True(cosine > 0.999f, $"cosine similarity {cosine} too low vs stable_audio_tools NumberConditioner reference.");
    }

    private static Tensor ReadBin(string path, TensorShape shape)
    {
        byte[] bytes = File.ReadAllBytes(path);
        Tensor t = new(shape, DType.F32);
        fixed (byte* src = bytes) Buffer.MemoryCopy(src, (void*)t.DataPointer, bytes.Length, bytes.Length);
        return t;
    }

    private static (float Cosine, float MaxAbsDiff) Compare(Tensor a, Tensor b)
    {
        long n = a.Shape.ElementCount;
        float* pa = (float*)a.DataPointer;
        float* pb = (float*)b.DataPointer;
        double dot = 0, na = 0, nb = 0;
        float maxAbsDiff = 0;
        for (long i = 0; i < n; i++)
        {
            dot += (double)pa[i] * pb[i];
            na += (double)pa[i] * pa[i];
            nb += (double)pb[i] * pb[i];
            maxAbsDiff = MathF.Max(maxAbsDiff, MathF.Abs(pa[i] - pb[i]));
        }
        float cosine = (float)(dot / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-12));
        return (cosine, maxAbsDiff);
    }
}
