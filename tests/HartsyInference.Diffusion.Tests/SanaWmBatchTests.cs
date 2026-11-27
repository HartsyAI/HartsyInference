using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.SanaWm;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>A CFG batch of two through the SANA-WM block and DiT equals each item run alone: the per-item attention, the
/// row-indexed per-frame gates, the chunked text attention and the DiT's sequential split all slice and write their rows
/// correctly. Tiny synthetic two-block layout on the CPU; the items differ in input, timesteps, caption mask and camera.</summary>
public sealed class SanaWmBatchTests : IDisposable
{
    private const int Frames = 2;
    private const int Height = 2;
    private const int Width = 2;
    private const int Tokens = Frames * Height * Width;
    private const int TextTokens = 4;
    private const int CameraChannels = 20;
    private const float Tolerance = 1e-5f;
    private static readonly float[][] Timesteps = [[21f, 500f], [300f, 900f]];
    private static readonly float[][] Masks = [[1f, 1f, 1f, 1f], [1f, 1f, 1f, 0f]];
    private readonly string _path = Path.Combine(Path.GetTempPath(), $"sanawm-batch-{Guid.NewGuid():N}.safetensors");

    public void Dispose()
    {
        if (File.Exists(_path))
            File.Delete(_path);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void Block_BatchOfTwo_MatchesEachItemAlone(int index)
    {
        IBackend backend = new CpuBackend();
        SanaWmSyntheticLayout.WriteFile(_path);
        using SanaWmCheckpoint checkpoint = SanaWmCheckpointConverter.Load(_path);
        using SanaWmBlock block = new(backend, checkpoint.Config, index, checkpoint.Weights);

        using Tensor joint = RunBlock(backend, checkpoint, block, [0, 1]);
        for (int item = 0; item < 2; item++)
        {
            using Tensor alone = RunBlock(backend, checkpoint, block, [item]);
            AssertItem(joint, alone, item);
        }
    }

    [Fact]
    public void Dit_BatchOfTwo_MatchesEachItemAlone()
    {
        IBackend backend = new CpuBackend();
        SanaWmSyntheticLayout.WriteFile(_path);
        using SanaWmCheckpoint checkpoint = SanaWmCheckpointConverter.Load(_path);
        using SanaWmDit dit = new(backend, checkpoint);

        using Tensor joint = RunDit(dit, checkpoint.Config, [0, 1]);
        for (int item = 0; item < 2; item++)
        {
            using Tensor alone = RunDit(dit, checkpoint.Config, [item]);
            AssertItem(joint, alone, item);
        }
    }

    private static Tensor RunBlock(IBackend backend, SanaWmCheckpoint checkpoint, SanaWmBlock block, int[] items)
    {
        SanaWmConfig config = checkpoint.Config;
        int channels = config.HiddenSize;
        List<Tensor> scratch = [];
        try
        {
            Tensor x = Patterned(scratch, items, [Tokens, channels], 0.01f);
            Tensor memory = Patterned(scratch, items, [TextTokens, channels], 0.02f);
            Tensor memoryBias = SanaWmConditioning.MaskBias(backend, Concat(items, Masks), items.Length, TextTokens, scratch);
            SanaWmTimeWeights time = new(checkpoint.Weights["t_embedder.mlp.0.weight"], checkpoint.Weights["t_embedder.mlp.0.bias"],
                checkpoint.Weights["t_embedder.mlp.2.weight"], checkpoint.Weights["t_embedder.mlp.2.bias"],
                checkpoint.Weights["t_block.1.weight"], checkpoint.Weights["t_block.1.bias"]);
            Tensor blockTime = SanaWmConditioning.Time(backend, time, config.TimestepWidth, Concat(items, Timesteps), scratch).BlockTime;
            Tensor chunk = Patterned(scratch, items, [config.PluckerChannels, Frames, Height, Width], 0.05f);
            Tensor weight = checkpoint.Weights["plucker_embedder.proj.weight"].Reshape(new TensorShape([channels, config.PluckerChannels]));
            scratch.Add(weight);
            Tensor plucker = SanaWmConditioning.Plucker(backend, weight, checkpoint.Weights["plucker_embedder.proj.bias"], chunk, scratch);

            using SanaWmForwardContext context = SanaWmForwardContext.Create(backend, config, items.Length, Frames, Height, Width,
                blockTime, memory, memoryBias, plucker, Cameras(items));
            return block.Forward(x, context);
        }
        finally
        {
            SanaWmTensors.DisposeAll(scratch);
        }
    }

    private static Tensor RunDit(SanaWmDit dit, SanaWmConfig config, int[] items)
    {
        List<Tensor> scratch = [];
        try
        {
            Tensor x = Patterned(scratch, items, [config.InChannels, Frames, Height, Width], 0.03f);
            Tensor text = Patterned(scratch, items, [1, TextTokens, config.TextWidth], 0.02f);
            Tensor mask = Hosted(scratch, Concat(items, Masks), [items.Length, TextTokens]);
            Tensor chunk = Patterned(scratch, items, [config.PluckerChannels, Frames, Height, Width], 0.05f);
            return dit.Forward(x, Concat(items, Timesteps), text, mask, Cameras(items), chunk);
        }
        finally
        {
            SanaWmTensors.DisposeAll(scratch);
        }
    }

    private static void AssertItem(Tensor joint, Tensor alone, int item)
    {
        ReadOnlySpan<float> all = joint.AsReadOnlySpan<float>();
        ReadOnlySpan<float> single = alone.AsReadOnlySpan<float>();
        Assert.Equal(all.Length, 2 * single.Length);
        ReadOnlySpan<float> half = all.Slice(item * single.Length, single.Length);
        float maxError = 0f;
        for (int i = 0; i < single.Length; i++)
        {
            Assert.True(float.IsFinite(single[i]), "The single-item output holds a non-finite value.");
            maxError = MathF.Max(maxError, MathF.Abs(half[i] - single[i]));
        }
        Assert.True(maxError <= Tolerance, $"Item {item}: batch-2 and batch-1 outputs differ by {maxError:E3}.");
    }

    // Per item, an F32 pattern over itemDims that depends on the item, stacked along a new leading axis.
    private static Tensor Patterned(List<Tensor> scratch, int[] items, long[] itemDims, float scale)
    {
        long[] dims = [items.Length, .. itemDims];
        Tensor tensor = new(new TensorShape(dims), DType.F32);
        scratch.Add(tensor);
        Span<float> values = tensor.AsSpan<float>();
        int length = values.Length / items.Length;
        for (int k = 0; k < items.Length; k++)
        {
            for (int i = 0; i < length; i++)
                values[k * length + i] = (((i + 37 * items[k]) % 97) - 48) * scale;
        }
        return tensor;
    }

    private static Tensor Hosted(List<Tensor> scratch, float[] values, long[] dims)
    {
        Tensor tensor = new(new TensorShape(dims), DType.F32);
        values.CopyTo(tensor.AsSpan<float>());
        scratch.Add(tensor);
        return tensor;
    }

    private static float[] Concat(int[] items, float[][] perItem) => [.. items.SelectMany(item => perItem[item])];

    // Identity rotation per frame with a translation that differs by item, then fx, fy, cx, cy in latent pixels.
    private static float[] Cameras(int[] items)
    {
        float[] conditions = new float[items.Length * Frames * CameraChannels];
        for (int k = 0; k < items.Length; k++)
        {
            for (int f = 0; f < Frames; f++)
            {
                int at = (k * Frames + f) * CameraChannels;
                float[] pose = [1, 0, 0, 0.1f * f * (items[k] + 1), 0, 1, 0, -0.05f * items[k], 0, 0, 1, 0.02f * f, 0, 0, 0, 1];
                pose.CopyTo(conditions, at);
                conditions[at + 16] = 2f;
                conditions[at + 17] = 2f;
                conditions[at + 18] = 1f;
                conditions[at + 19] = 1f;
            }
        }
        return conditions;
    }
}
