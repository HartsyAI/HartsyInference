using HartsyInference.Audio.Models.Codecs.Mimi;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using Xunit;
using static HartsyInference.Tests.Common.CpuSchedules;

namespace HartsyInference.Audio.Tests;

/// <summary>Mimi's RVQ encode (Kyutai STT and CSM) searches every codebook entry for each frame, with the frames fanned
/// out through <see cref="CpuParallel"/>. The codes are the same whether the frames ran over every core, under a
/// <c>numerics.cpuThreads</c> cap of 1, or inside an <see cref="CpuParallel.InlineScope"/>. Synthetic weights, sized so
/// each codebook's search really fans out.</summary>
public sealed class MimiSplitRvqScheduleTests
{
    private const string Semantic = "semantic_residual_vector_quantizer";
    private const string Acoustic = "acoustic_residual_vector_quantizer";

    [Fact]
    public void Encode_GivesTheSameCodes_UnderEverySchedule()
    {
        const int numSemantic = 1, numTotal = 4, vocab = 128, dim = 16, latentDim = 32, frames = 96;
        Random rng = new(83);
        Dictionary<string, Tensor> weights = new();
        MimiSplitRvq? rvq = null;
        try
        {
            for (int i = 0; i < numTotal; i++)
            {
                string prefix = i < numSemantic ? $"quantizer.{Semantic}.layers.{i}" : $"quantizer.{Acoustic}.layers.{i - numSemantic}";
                weights[$"{prefix}.codebook.embed_sum"] = Filled(rng, 0.0, new TensorShape(vocab, dim));
                weights[$"{prefix}.codebook.cluster_usage"] = Filled(rng, 1.5, new TensorShape(vocab));
            }
            foreach (string quantizer in new[] { Semantic, Acoustic })
            {
                weights[$"quantizer.{quantizer}.output_proj.weight"] = Filled(rng, 0.0, new TensorShape(latentDim, dim, 1));
                weights[$"quantizer.{quantizer}.input_proj.weight"] = Filled(rng, 0.0, new TensorShape(dim, latentDim, 1));
            }
            MimiSplitRvq model = new("quantizer", numSemantic, numTotal, dim, latentDim);
            rvq = model;
            model.LoadWeights(weights);
            using Tensor latent = Filled(rng, 0.0, new TensorShape(1, latentDim, frames));
            using CpuBackend backend = new();

            (int[] parallel, int[] capped, int[] inline) = UnderEverySchedule(() => Codes(model.Encode(backend, latent, 1, frames)));

            Assert.Equal(parallel, capped);
            Assert.Equal(parallel, inline);
            Assert.Equal(numTotal * frames, parallel.Length);
            // A search that picked one entry for everything would agree under any schedule without showing anything.
            Assert.True(parallel.Distinct().Count() > numTotal, "the codebook search picked too few distinct entries to tell schedules apart");
        }
        finally
        {
            IEnumerable<Tensor> loaded = rvq?.EnumerateWeights() ?? [];
            foreach (Tensor tensor in loaded.Concat(weights.Values).Distinct()) tensor.Dispose();
        }
    }

    /// <summary>Uniform values in <c>[offset - 1, offset + 1)</c>.</summary>
    private static Tensor Filled(Random rng, double offset, TensorShape shape)
    {
        Tensor tensor = new(shape, DType.F32);
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++) values[i] = (float)(offset + rng.NextDouble() * 2.0 - 1.0);
        return tensor;
    }

    private static int[] Codes(Tensor codes)
    {
        using (codes)
        {
            return codes.AsReadOnlySpan<int>().ToArray();
        }
    }
}
