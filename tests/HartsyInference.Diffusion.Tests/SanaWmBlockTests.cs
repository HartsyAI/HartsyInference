using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.SanaWm;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Tiny-config CPU checks of the assembled SANA-WM block and DiT on the synthetic two-block layout (block 0 GDN,
/// block 1 softmax): shapes, finiteness, the zeroed plucker term, single-frame grids and timestep truncation.</summary>
public sealed class SanaWmBlockTests : IDisposable
{
    private const int Frames = 2;
    private const int Height = 2;
    private const int Width = 2;
    private const int Tokens = Frames * Height * Width;
    private const int TextTokens = 4;
    private const int Channels = 224;
    private const int PluckerChannels = 12;
    private const int CameraChannels = 20;
    private readonly string _path = Path.Combine(Path.GetTempPath(), $"sanawm-s5-{Guid.NewGuid():N}.safetensors");

    public void Dispose()
    {
        if (File.Exists(_path))
            File.Delete(_path);
    }

    [Fact]
    public void Block_GdnAndSoftmax_ReturnTokenShapeAndFiniteValues()
    {
        using IBackend backend = new CpuBackend();
        using SanaWmCheckpoint checkpoint = LoadLayout();
        foreach (int index in new[] { 0, 1 })
        {
            using SanaWmBlock block = new(backend, checkpoint.Config, index, checkpoint.Weights);
            using Tensor output = RunBlock(backend, checkpoint, block, Frames, Height, Width, withPlucker: true);
            Assert.Equal(DType.F32, output.DType);
            Assert.Equal(new long[] { 1, Tokens, Channels }, Dims(output.Shape));
            AssertFinite(output);
        }
    }

    [Fact]
    public void Block_ZeroedPluckerProjection_IgnoresThePluckerTerm()
    {
        using IBackend backend = new CpuBackend();
        using SanaWmCheckpoint checkpoint = LoadLayout();
        ZeroPluckerProjection(checkpoint, 0);
        using SanaWmBlock block = new(backend, checkpoint.Config, 0, checkpoint.Weights);
        using Tensor withTerm = RunBlock(backend, checkpoint, block, Frames, Height, Width, withPlucker: true);
        using Tensor withoutTerm = RunBlock(backend, checkpoint, block, Frames, Height, Width, withPlucker: false);
        Assert.Equal(withoutTerm.AsReadOnlySpan<float>().ToArray(), withTerm.AsReadOnlySpan<float>().ToArray());
    }

    [Fact]
    public void Block_SingleFrameGrid_Runs()
    {
        using IBackend backend = new CpuBackend();
        using SanaWmCheckpoint checkpoint = LoadLayout();
        foreach (int index in new[] { 0, 1 })
        {
            using SanaWmBlock block = new(backend, checkpoint.Config, index, checkpoint.Weights);
            using Tensor output = RunBlock(backend, checkpoint, block, 1, Height, Width, withPlucker: true);
            Assert.Equal(new long[] { 1, Height * Width, Channels }, Dims(output.Shape));
            AssertFinite(output);
        }
    }

    [Fact]
    public void Dit_TinyConfig_ReturnsVelocityInLatentLayout()
    {
        using IBackend backend = new CpuBackend();
        using SanaWmCheckpoint checkpoint = LoadLayout();
        using SanaWmDit dit = new(backend, checkpoint);
        using Tensor velocity = RunDit(backend, dit, checkpoint.Config, Frames, [21f, 500f]);
        Assert.Equal(new long[] { 1, checkpoint.Config.OutChannels, Frames, Height, Width }, Dims(velocity.Shape));
        AssertFinite(velocity);
    }

    [Fact]
    public void Dit_SingleFrameGrid_Runs()
    {
        using IBackend backend = new CpuBackend();
        using SanaWmCheckpoint checkpoint = LoadLayout();
        using SanaWmDit dit = new(backend, checkpoint);
        using Tensor velocity = RunDit(backend, dit, checkpoint.Config, 1, [750f]);
        Assert.Equal(new long[] { 1, checkpoint.Config.OutChannels, 1, Height, Width }, Dims(velocity.Shape));
        AssertFinite(velocity);
    }

    [Fact]
    public void Dit_TimestepsTruncateToIntegers()
    {
        using IBackend backend = new CpuBackend();
        using SanaWmCheckpoint checkpoint = LoadLayout();
        using SanaWmDit dit = new(backend, checkpoint);
        using Tensor whole = RunDit(backend, dit, checkpoint.Config, Frames, [500f, 250f]);
        using Tensor fractional = RunDit(backend, dit, checkpoint.Config, Frames, [500.9f, 250.4f]);
        Assert.Equal(whole.AsReadOnlySpan<float>().ToArray(), fractional.AsReadOnlySpan<float>().ToArray());
    }

    private SanaWmCheckpoint LoadLayout()
    {
        SanaWmSyntheticLayout.WriteFile(_path);
        return SanaWmCheckpointConverter.Load(_path);
    }

    private static void ZeroPluckerProjection(SanaWmCheckpoint checkpoint, int index)
    {
        // The converted weight is BF16 and the bias F32; zero bits are +0.0 in both.
        checkpoint.Weights[$"blocks.{index}.plucker_proj.weight"].AsSpan<ushort>().Fill(0);
        checkpoint.Weights[$"blocks.{index}.plucker_proj.bias"].AsSpan<float>().Fill(0f);
    }

    private static Tensor RunBlock(IBackend backend, SanaWmCheckpoint checkpoint, SanaWmBlock block, int frames, int height,
        int width, bool withPlucker)
    {
        int tokens = frames * height * width;
        List<Tensor> scratch = [];
        List<Tensor> owned = [];
        try
        {
            Tensor x = Pattern(scratch, [1, tokens, Channels], 0.01f);
            Tensor memory = Pattern(scratch, [1, TextTokens, Channels], 0.02f);
            Tensor memoryBias = SanaWmConditioning.MaskBias(backend, Ones(scratch, [1, TextTokens]), 1, TextTokens, scratch);
            float[] stamps = frames == 1 ? [750f] : [21f, 500f];
            Tensor timeModulation = TimeModulation(backend, checkpoint, stamps, scratch, owned);
            Tensor? plucker = null;
            if (withPlucker)
            {
                Tensor chunk = Pattern(scratch, [1, PluckerChannels, frames, height, width], 0.05f);
                Tensor weight = checkpoint.Weights["plucker_embedder.proj.weight"].Reshape(new TensorShape([Channels, PluckerChannels]));
                owned.Add(weight);
                plucker = SanaWmConditioning.Plucker(backend, weight, checkpoint.Weights["plucker_embedder.proj.bias"], chunk, scratch);
            }

            using SanaWmForwardContext context = SanaWmForwardContext.Create(backend, checkpoint.Config, 1, frames, height, width,
                timeModulation, memory, memoryBias, plucker, CameraConditions(frames));
            return block.Forward(x, context);
        }
        finally
        {
            SanaWmTensors.DisposeAll(scratch);
            SanaWmTensors.DisposeAll(owned);
        }
    }

    private static Tensor RunDit(IBackend backend, SanaWmDit dit, SanaWmConfig config, int frames, float[] timesteps)
    {
        List<Tensor> scratch = [];
        try
        {
            Tensor x = Pattern(scratch, [1, config.InChannels, frames, Height, Width], 0.03f);
            Tensor text = Pattern(scratch, [1, 1, TextTokens, config.TextWidth], 0.02f);
            Tensor mask = Ones(scratch, [1, TextTokens]);
            Tensor chunk = Pattern(scratch, [1, config.PluckerChannels, frames, Height, Width], 0.05f);
            return dit.Forward(x, timesteps, text, mask, CameraConditions(frames), chunk);
        }
        finally
        {
            SanaWmTensors.DisposeAll(scratch);
        }
    }

    private static Tensor TimeModulation(IBackend backend, SanaWmCheckpoint checkpoint, float[] timesteps, List<Tensor> scratch,
        List<Tensor> owned)
    {
        SanaWmTimeWeights weights = new(checkpoint.Weights["t_embedder.mlp.0.weight"], checkpoint.Weights["t_embedder.mlp.0.bias"],
            checkpoint.Weights["t_embedder.mlp.2.weight"], checkpoint.Weights["t_embedder.mlp.2.bias"],
            checkpoint.Weights["t_block.1.weight"], checkpoint.Weights["t_block.1.bias"]);
        return SanaWmConditioning.Time(backend, weights, checkpoint.Config.TimestepWidth, timesteps, scratch).BlockTime;
    }

    // Identity camera-to-world per frame with a small translation, then fx, fy, cx, cy in latent pixels.
    private static float[] CameraConditions(int frames)
    {
        float[] conditions = new float[frames * CameraChannels];
        for (int f = 0; f < frames; f++)
        {
            float[] pose = [1, 0, 0, 0.1f * f, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
            pose.CopyTo(conditions, f * CameraChannels);
            conditions[f * CameraChannels + 16] = 2f;
            conditions[f * CameraChannels + 17] = 2f;
            conditions[f * CameraChannels + 18] = 1f;
            conditions[f * CameraChannels + 19] = 1f;
        }

        return conditions;
    }

    private static long[] Dims(TensorShape shape)
    {
        long[] dims = new long[shape.Rank];
        for (int i = 0; i < dims.Length; i++)
            dims[i] = shape[i];
        return dims;
    }

    private static Tensor Pattern(List<Tensor> scratch, ReadOnlySpan<long> dims, float scale)
    {
        Tensor tensor = new(new TensorShape(dims), DType.F32, DeviceKind.Cpu);
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++)
            values[i] = ((i % 97) - 48) * scale;
        scratch.Add(tensor);
        return tensor;
    }

    private static Tensor Ones(List<Tensor> scratch, ReadOnlySpan<long> dims)
    {
        Tensor tensor = new(new TensorShape(dims), DType.F32, DeviceKind.Cpu);
        tensor.AsSpan<float>().Fill(1f);
        scratch.Add(tensor);
        return tensor;
    }

    private static void AssertFinite(Tensor tensor)
    {
        foreach (float value in tensor.AsReadOnlySpan<float>())
            Assert.True(float.IsFinite(value), "Output holds a non-finite value.");
    }
}
