using HartsyInference.Audio.Models.LongCat;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Tests.Shared;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Block 0 of a released INT8 LongCat-Video-Avatar DiT on CUDA, with and without <c>numerics.longcatInt8WeightOnly</c>, against the CPU
/// backend on exact F32 level times scale: self-attention, text cross-attention, audio step and FFN. The block's inputs are what the CPU forward's
/// own front end makes of seeded latents, text and audio at timestep 1000 (4,600 tokens, one conditioning frame), so the modulation is the
/// release's own. No LoRA. Runs only when HARTSY_LONGCAT_INT8_INDEX names the release's index, beside its shards and <c>config.json</c>, and a CUDA
/// device opens. The CPU forward takes one to two minutes.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class LongCatInt8WeightOnlyRealBlockTests(ITestOutputHelper output)
{
    private const string EnvName = "HARTSY_LONGCAT_INT8_INDEX";
    private const int LatentFrames = 5, LatentHeight = 46, LatentWidth = 80, TextRows = 512, ValidTokens = 4;
    private const int PixelFrames = (LatentFrames - 1) * 4 + 1;

    /// <summary>The relative RMS each stage may differ from the CPU by under the setting. Written before this test's first run, from an earlier
    /// measurement of the same route on this block with other inputs and a LoRA attached, whose largest stage figure was 1.5e-2.</summary>
    private const double WeightOnlyBound = 2e-2;

    private static readonly string[] Stages = ["self_attn", "after_self", "cross update", "audio update", "ffn", "out"];

    [RealBlockFact]
    public void ReleasedBlockZeroUnderTheSettingMatchesTheCpuWithExactWeights()
    {
        string directory = Path.GetDirectoryName(Path.GetFullPath(Environment.GetEnvironmentVariable(EnvName)!))!;
        using LongCatCheckpoint release = LongCatCheckpoint.Open(directory);
        LongCatConfig config = release.Config with { Depth = 1 };
        int hw = LatentHeight / 2 * (LatentWidth / 2);

        Dictionary<string, float[]> expected = [];
        float[] audioTokens;
        TensorShape audioShape;
        Dictionary<string, Tensor> exact = ExactWeights(FirstBlock(release));
        try
        {
            using CpuBackend cpu = new();
            using LongCatTransformer dit = new(config);
            dit.LoadWeights(exact);
            dit.Probe = (name, t) => expected[name] = t.AsReadOnlySpan<float>().ToArray();
            LongCatAudioLayout layout = LongCatAudioLayout.Whisper;
            using Tensor latent = LongCatTestData.Latent(config, LatentFrames, LatentHeight, LatentWidth, seed: 7);
            using Tensor text = LongCatTestData.Random(new TensorShape(TextRows, config.CaptionChannels), 8, scale: 0.4f);
            using Tensor windows = LongCatTestData.Random(new TensorShape(PixelFrames, layout.WindowValues), 9);
            using Tensor audio = dit.ProjectAudio(cpu, windows, LatentFrames);
            audioTokens = audio.AsReadOnlySpan<float>().ToArray();
            audioShape = audio.Shape;
            LongCatConditioning conditioning = new(NumCondLatents: 1);
            using Tensor velocity = dit.Forward(cpu, latent, text, conditioning.FrameTimesteps(1000f, LatentFrames), in conditioning, ValidTokens, audio);
        }
        finally
        {
            LongCatTestData.DisposeAll(exact);
        }
        Dictionary<string, float[]> cpuStages = new[] { "self_attn", "after_self", "after_cross", "after_audio", "ffn", "out" }
            .ToDictionary(static s => s, s => expected[$"block0.{s}"]);
        AddUpdates(cpuStages);

        using CudaBackend cuda = GpuDevices.OpenCuda();
        double[] resident = Lane(setting: false);
        double[] weightOnly = Lane(setting: true);

        Assert.All(weightOnly, static e => Assert.True(e <= WeightOnlyBound, $"a stage differs by {e:E3} under the setting, above {WeightOnlyBound:E0}"));
        // The setting has to be what brought the block inside the bound.
        Assert.True(resident.Max() > WeightOnlyBound, $"without the setting the largest stage difference is {resident.Max():E3}, inside the same bound");

        double[] Lane(bool setting)
        {
            // A checkpoint of its own per lane: the setting marks the tensors it is given.
            using LongCatCheckpoint copy = LongCatCheckpoint.Open(directory);
            Dictionary<string, Tensor> weights = FirstBlock(copy);
            if (setting)
            {
                using IDisposable on = KnobProfile.Create("longcat-int8-weight-only").With(EngineKnobs.LongcatInt8WeightOnly, true).Push();
                Assert.True(LongCatRecipeSupport.ApplyInt8WeightOnly(weights) > 0);
            }
            Dictionary<string, float[]> stages = Block(cuda, config, weights, expected["x_embedder"], expected["y_embedder"], expected["t_embedder"],
                audioTokens, audioShape, hw);
            return [.. Stages.Select(stage =>
            {
                double error = LongCatTestData.RelRms(stages[stage], cpuStages[stage]);
                output.WriteLine(FormattableString.Invariant(
                    $"{cuda.Capabilities.Name}, setting {(setting ? "on" : "off")}: {stage,-13} relRMS {error:E3}"));
                return error;
            })];
        }
    }

    private static Dictionary<string, Tensor> FirstBlock(LongCatCheckpoint checkpoint) => checkpoint.Weights
        .Where(static p => !p.Key.StartsWith("blocks.", StringComparison.Ordinal) || p.Key.StartsWith("blocks.0.", StringComparison.Ordinal))
        .ToDictionary(static p => p.Key, static p => p.Value, StringComparer.Ordinal);

    /// <summary>F32 copies: each INT8 weight as level times row scale, every other tensor as it is.</summary>
    private static Dictionary<string, Tensor> ExactWeights(Dictionary<string, Tensor> source)
    {
        Dictionary<string, Tensor> result = new(StringComparer.Ordinal);
        foreach ((string key, Tensor tensor) in source)
        {
            if (tensor.QuantInfo?.RowScale is not { } rowScale)
            {
                result[key] = tensor.CastTo(DType.F32);
                continue;
            }
            Tensor dense = new(tensor.Shape, DType.F32);
            ReadOnlySpan<byte> levels = tensor.AsReadOnlySpan<byte>();
            ReadOnlySpan<float> scales = rowScale.AsReadOnlySpan<float>();
            Span<float> values = dense.AsSpan<float>();
            int columns = (int)tensor.Shape[1];
            for (int i = 0; i < values.Length; i++)
                values[i] = unchecked((sbyte)levels[i]) * scales[i / columns];
            result[key] = dense;
        }
        return result;
    }

    private static Dictionary<string, float[]> Block(IBackend backend, LongCatConfig config, Dictionary<string, Tensor> weights, float[] hidden,
        float[] text, float[] time, float[] audio, TensorShape audioShape, int hw)
    {
        List<Tensor> owned = [];
        LongCatBlock block = new(config);
        Dictionary<string, float[]> stages = [];
        try
        {
            block.LoadWeights(weights, "blocks.0", owned);
            using Tensor h = FromArray(new TensorShape(LatentFrames * hw, config.HiddenSize), hidden);
            using Tensor y = FromArray(new TensorShape(TextRows, config.HiddenSize), text);
            using Tensor t = FromArray(new TensorShape(LatentFrames, config.AdalnTembedDim), time);
            using Tensor a = FromArray(audioShape, audio);
            WanRope rope = new(config.HeadDim, LongCatConfig.RopeTheta, 0);
            (Tensor cos, Tensor sin) = rope.BuildCosSin(LatentFrames, LatentHeight / 2, LatentWidth / 2);
            LongCatConditioning conditioning = new(NumCondLatents: 1);
            using (cos)
            using (sin)
            using (Tensor index = LongCatOps.FrameRowIndex(LatentFrames, hw))
            {
                LongCatBlockInputs inputs = new(LatentFrames, hw, 1, cos, sin, conditioning.AttentionPlan(hw, LatentFrames), index);
                using Tensor result = block.Forward(backend, h, y, t, inputs,
                    (name, tensor) => stages[name[(name.IndexOf('.') + 1)..]] = tensor.AsReadOnlySpan<float>().ToArray(), "block0", a);
            }
            AddUpdates(stages);
            return stages;
        }
        finally
        {
            backend.FreeWeights(block.EnumerateWeights());
            block.Dispose();
            foreach (Tensor tensor in owned)
                tensor.Dispose();
        }
    }

    /// <summary>The cross-attention and audio updates alone: the stream around them is far larger and would hide their difference.</summary>
    private static void AddUpdates(Dictionary<string, float[]> stages)
    {
        stages["cross update"] = Difference(stages["after_cross"], stages["after_self"]);
        stages["audio update"] = Difference(stages["after_audio"], stages["after_cross"]);
    }

    private static float[] Difference(float[] after, float[] before) => [.. after.Select((value, i) => value - before[i])];

    private static Tensor FromArray(TensorShape shape, float[] values)
    {
        Tensor tensor = new(shape, DType.F32);
        values.AsSpan().CopyTo(tensor.AsSpan<float>());
        return tensor;
    }

    /// <summary>Runs only when the released index is named and a CUDA device opens. Otherwise the test is reported as skipped, never
    /// passed.</summary>
    private sealed class RealBlockFactAttribute : FactAttribute
    {
        public RealBlockFactAttribute()
        {
            string? index = Environment.GetEnvironmentVariable(EnvName);
            if (string.IsNullOrWhiteSpace(index) || !File.Exists(index))
                Skip = $"SKIPPED: {EnvName} is unset or not an existing file.";
            else if (GpuDevices.CudaFailure is { } failure)
                Skip = $"SKIPPED: no usable CUDA device ({failure}).";
        }
    }
}
