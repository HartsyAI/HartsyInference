using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Maxwell;
using HartsyInference.LLM.Transformer;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary>The tiny Maxwell reference checkpoint on the CUDA backend, against the release's logits: the dry run before a real-weight GPU job.
/// Run explicitly (GpuIntegration); it needs a CUDA device and the PTX kernels.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class MaxwellCudaTinyTests(ITestOutputHelper output)
{
    private static readonly string Directory = Path.Combine(AppContext.BaseDirectory, "Fixtures", "Maxwell");

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
    public void Cuda_PrefillAndDecode_MatchTheReference()
    {
        using CudaBackend backend = new(0, CudaPtxDirectory.Resolve());
        using MaxwellModel model = MaxwellModel.Load(Directory, backend, new MaxwellLoadOptions { MaxSequenceTokens = 64, ResidentWeights = true });
        using MaxwellGenerationModel generation = new(model, backend, 64);
        int[] ids = NpyReader.ReadInt32(Path.Combine(Directory, "input_ids.npy")).Data;
        float[] prefillReference = NpyReader.ReadFloat32(Path.Combine(Directory, "prefill_logits.npy")).Data;
        float[] decodeReference = NpyReader.ReadFloat32(Path.Combine(Directory, "decode_logits.npy")).Data;
        const int Prefill = 30;
        int vocab = model.Config.VocabSize;
        ISequenceState state = generation.CreateSequenceState(new SequenceStateOptions(64));

        using Tensor hidden = generation.Prefill(new PrefillChunk(ids.AsMemory(0, Prefill), 0, LastRowOnly: false), state);
        using Tensor logits = generation.ProjectLogits(hidden, Prefill);
        double worst = Relative(logits.AsReadOnlySpan<float>(), prefillReference);
        for (int step = 0; step < ids.Length - Prefill; step++)
        {
            using Tensor decoded = generation.DecodeBatch([ids[Prefill + step]], [state]);
            using Tensor stepLogits = generation.ProjectLogits(decoded, 1);
            worst = Math.Max(worst, Relative(stepLogits.AsReadOnlySpan<float>(), decodeReference.AsSpan(step * vocab, vocab)));
        }

        output.WriteLine($"CUDA tiny checkpoint: prefill {Prefill} + {ids.Length - Prefill} decode steps, worst relative max diff {worst:E2}");
        Assert.True(worst < 2e-3, $"CUDA differs from the reference by {worst:E2}");
    }
}
