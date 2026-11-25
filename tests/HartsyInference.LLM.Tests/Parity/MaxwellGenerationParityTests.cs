using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Transformer;
using HartsyInference.LLM.Maxwell;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary>The Text path's generation adapter (<see cref="MaxwellGenerationModel"/>) on the tiny reference checkpoint: a 30-token prefill through
/// the pipeline's prefill and projection, then ten single-token batched decode steps, each compared with the release's own logits.</summary>
public sealed class MaxwellGenerationParityTests(ITestOutputHelper output)
{
    private static readonly string Directory = Path.Combine(AppContext.BaseDirectory, "Fixtures", "Maxwell");

    private static float[] Load(string name) => NpyReader.ReadFloat32(Path.Combine(Directory, name + ".npy")).Data;

    private static double Relative(ReadOnlySpan<float> actual, ReadOnlySpan<float> reference)
    {
        Assert.Equal(reference.Length, actual.Length);
        double diff = 0, scale = 1e-6;
        for (int i = 0; i < reference.Length; i++)
        {
            diff = Math.Max(diff, Math.Abs(actual[i] - reference[i]));
            scale = Math.Max(scale, Math.Abs(reference[i]));
        }
        return diff / scale;
    }

    [Fact]
    public void PrefillAndBatchedDecode_MatchTheReferenceLogits()
    {
        using MaxwellModel model = MaxwellModel.Load(Directory, new CpuBackend(), new MaxwellLoadOptions { MaxSequenceTokens = 64, ResidentWeights = true });
        using MaxwellGenerationModel generation = new(model, new CpuBackend(), 64);
        int[] ids = NpyReader.ReadInt32(Path.Combine(Directory, "input_ids.npy")).Data;
        const int Prefill = 30;
        int vocab = model.Config.VocabSize;
        ISequenceState state = generation.CreateSequenceState(new SequenceStateOptions(64));

        using Tensor hidden = generation.Prefill(new PrefillChunk(ids.AsMemory(0, Prefill), 0, LastRowOnly: false), state);
        using Tensor prefillLogits = generation.ProjectLogits(hidden, Prefill);
        double worst = Relative(prefillLogits.AsReadOnlySpan<float>(), Load("prefill_logits"));

        float[] decodeReference = Load("decode_logits");
        for (int step = 0; step < ids.Length - Prefill; step++)
        {
            using Tensor decoded = generation.DecodeBatch([ids[Prefill + step]], [state]);
            using Tensor logits = generation.ProjectLogits(decoded, 1);
            worst = Math.Max(worst, Relative(logits.AsReadOnlySpan<float>(), decodeReference.AsSpan(step * vocab, vocab)));
        }

        output.WriteLine($"prefill {Prefill} + {ids.Length - Prefill} batched decode steps: worst relative max diff {worst:E2}");
        Assert.True(worst < 1e-4, $"generation adapter differs from the reference by {worst:E2}");
    }

    [Fact]
    public void SequenceStatesBeyondTheLoadedLimit_AreRefused()
    {
        using MaxwellModel model = MaxwellModel.Load(Directory, new CpuBackend(), new MaxwellLoadOptions { MaxSequenceTokens = 64, ResidentWeights = true });
        using MaxwellGenerationModel generation = new(model, new CpuBackend(), 64);

        Assert.Throws<ArgumentOutOfRangeException>(() => generation.CreateSequenceState(new SequenceStateOptions(65)));
    }

    [Fact]
    public void EstimateSequenceBytes_CountsKeysValuesAndIndexerKeysPerLayer()
    {
        using MaxwellModel model = MaxwellModel.Load(Directory, new CpuBackend(), new MaxwellLoadOptions { MaxSequenceTokens = 64, ResidentWeights = true });
        using MaxwellGenerationModel generation = new(model, new CpuBackend(), 64);
        MaxwellConfig c = model.Config;

        long rowFloats = 2L * c.NumKeyValueHeads * c.HeadDim + c.Sparse.IndexerHeadDim;
        // 20 tokens may double to 40 rows (within the 64-token limit); 100 tokens cannot exceed the 100 rows they need.
        Assert.Equal(40L * c.NumHiddenLayers * rowFloats * sizeof(float), generation.EstimateSequenceBytes(20));
        Assert.Equal(100L * c.NumHiddenLayers * rowFloats * sizeof(float), generation.EstimateSequenceBytes(100));
    }
}
