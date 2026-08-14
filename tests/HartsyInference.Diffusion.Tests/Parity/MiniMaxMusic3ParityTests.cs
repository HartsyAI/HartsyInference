using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Music;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real-weight parity for MiniMax Music 3's condition encoder and flow-matching transformer against the
/// diffusers reference dumped by <c>tests/python-reference/dump_minimax_music3_reference.py</c>. Skips without
/// <c>HARTSY_MINIMAX_MUSIC3_PATH</c> and the dump.</summary>
[Trait("Category", "Integration")]
public sealed unsafe class MiniMaxMusic3ParityTests(ITestOutputHelper output)
{
    private const double ConditionTolerance = 1e-5;
    private const double BlockTolerance = 1e-4;
    private const double DitTolerance = 1e-3;

    [Fact]
    public void ConditionEncoder_MatchesReference()
    {
        if (!TryOpen("condition_encoder", out MiniMaxMusic3Reference? reference, out Dictionary<string, Tensor>? weights, out List<SafeTensorsLoader> loaders)
            || !reference!.Has("cond_out"))
        {
            return;
        }
        using Disposables open = new Disposables(loaders);
        using MiniMaxMusic3ConditionEncoder encoder = new MiniMaxMusic3ConditionEncoder();
        encoder.LoadWeights(weights!);

        int[] inputShape = reference.Shape("cond_in");
        float[] frameHiddens = reference.Read("cond_in");
        using Tensor condition = encoder.Encode(new CpuBackend(), frameHiddens, frameOffset: 0, frameCount: inputShape[1]);

        int[] expectedShape = reference.Shape("cond_out");
        Assert.Equal(expectedShape[1], (int)condition.Shape[1]);
        Assert.Equal(expectedShape[1], MiniMaxMusic3ConditionEncoder.LatentLength(inputShape[1]));
        Report("ConditionEncoder", condition.AsReadOnlySpan<float>(), reference.Read("cond_out"), ConditionTolerance);
    }

    [Fact]
    public void DitBlockZero_MatchesReference()
    {
        if (!TryOpen("transformer", out MiniMaxMusic3Reference? reference, out Dictionary<string, Tensor>? weights, out List<SafeTensorsLoader> loaders)
            || !reference!.Has("dit_block0_out"))
        {
            return;
        }
        using Disposables open = new Disposables(loaders);
        MiniMaxMusic3DitConfig config = MiniMaxMusic3DitConfig.Default with { NumLayers = 1 };
        using MiniMaxMusic3Dit dit = new MiniMaxMusic3Dit(config);
        dit.LoadWeights(weights!);

        int[] shape = reference.Shape("dit_block0_in");
        float[] blockInput = reference.Read("dit_block0_in");
        using Tensor hidden = new Tensor(new TensorShape(shape[1], shape[2]), DType.F32);
        blockInput.CopyTo(new Span<float>((float*)hidden.DataPointer, blockInput.Length));

        using Tensor actual = dit.ForwardBlocks(new CpuBackend(), hidden);
        Report("DitBlock0", actual.AsReadOnlySpan<float>(), reference.Read("dit_block0_out"), BlockTolerance);
    }

    [Fact]
    [Trait("Category", "Slow")]
    public void Dit_MatchesReference()
    {
        if (!TryOpen("transformer", out MiniMaxMusic3Reference? reference, out Dictionary<string, Tensor>? weights, out List<SafeTensorsLoader> loaders)
            || !reference!.Has("dit_out_t0_cond"))
        {
            return;
        }
        using Disposables open = new Disposables(loaders);
        using MiniMaxMusic3Dit dit = new MiniMaxMusic3Dit();
        dit.LoadWeights(weights!);

        int[] latentShape = reference.Shape("dit_latents");
        float[] latentValues = reference.Read("dit_latents");
        using Tensor latents = new Tensor(new TensorShape(latentShape[0], latentShape[1], latentShape[2]), DType.F32);
        latentValues.CopyTo(new Span<float>((float*)latents.DataPointer, latentValues.Length));

        int[] conditionShape = reference.Shape("cond_out");
        float[] conditionValues = reference.Read("cond_out");
        using Tensor condition = new Tensor(new TensorShape(conditionShape[0], conditionShape[1], conditionShape[2]), DType.F32);
        conditionValues.CopyTo(new Span<float>((float*)condition.DataPointer, conditionValues.Length));
        using Tensor zeros = new Tensor(condition.Shape, DType.F32);

        CpuBackend backend = new CpuBackend();
        float timestep = (float)reference.Meta("dit_timesteps")[0].GetDouble();
        using Tensor conditional = dit.Forward(backend, latents, timestep, condition);
        Report("Dit(t0,cond)", conditional.AsReadOnlySpan<float>(), reference.Read("dit_out_t0_cond"), DitTolerance);

        // The unconditional branch conditions on zeros, so a broken zero path would still look plausible alone.
        using Tensor unconditional = dit.Forward(backend, latents, timestep, zeros);
        Report("Dit(t0,uncond)", unconditional.AsReadOnlySpan<float>(), reference.Read("dit_out_t0_uncond"), DitTolerance);

        float lateTimestep = (float)reference.Meta("dit_timesteps")[1].GetDouble();
        using Tensor later = dit.Forward(backend, latents, lateTimestep, condition);
        Report("Dit(t1,cond)", later.AsReadOnlySpan<float>(), reference.Read("dit_out_t1_cond"), DitTolerance);
    }

    private void Report(string label, ReadOnlySpan<float> actual, float[] expected, double tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        (double meanAbs, double maxAbs, double correlation) = MiniMaxMusic3Reference.Compare(actual, expected);
        output.WriteLine($"[{label}] n={actual.Length} meanAbs={meanAbs:E3} maxAbs={maxAbs:E3} corr={correlation:F8}");
        Assert.True(meanAbs < tolerance,
            $"{label} diverges from the diffusers reference: meanAbs={meanAbs:E3}, maxAbs={maxAbs:E3}, corr={correlation:F8}");
    }

    /// <summary>Opens a checkpoint subfolder, merging its shards when the weights are split.</summary>
    private static bool TryOpen(string subfolder, out MiniMaxMusic3Reference? reference,
        out Dictionary<string, Tensor>? weights, out List<SafeTensorsLoader> loaders)
    {
        reference = MiniMaxMusic3Reference.TryLoad();
        weights = null;
        loaders = [];
        string? checkpoint = Environment.GetEnvironmentVariable("HARTSY_MINIMAX_MUSIC3_PATH");
        if (checkpoint is null || reference is null)
        {
            return false;
        }
        string directory = Path.Combine(checkpoint, subfolder);
        if (!Directory.Exists(directory))
        {
            return false;
        }
        string[] shards = Directory.GetFiles(directory, "*.safetensors");
        if (shards.Length == 0)
        {
            return false;
        }
        Array.Sort(shards, StringComparer.Ordinal);
        weights = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        foreach (string shard in shards)
        {
            SafeTensorsLoader loader = new SafeTensorsLoader();
            loader.Load(shard);
            loaders.Add(loader);
            foreach (KeyValuePair<string, Tensor> entry in loader.GetAllTensors())
            {
                weights[entry.Key] = entry.Value;
            }
        }
        return true;
    }

    /// <summary>Disposes a shard set as one <c>using</c>.</summary>
    private sealed class Disposables(List<SafeTensorsLoader> loaders) : IDisposable
    {
        public void Dispose()
        {
            foreach (SafeTensorsLoader loader in loaders)
            {
                loader.Dispose();
            }
        }
    }
}
