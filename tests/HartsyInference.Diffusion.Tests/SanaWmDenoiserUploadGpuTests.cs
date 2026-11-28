using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers.SanaWm;
using HartsyInference.Diffusion.Streaming;
using HartsyInference.Tests.Shared;
using HartsyInference.World.Camera;
using HartsyInference.World.Pipelines.SanaWm;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The two SANA-WM DiT denoisers on CUDA, on the synthetic two-block layout: each uploads its own inputs, so its
/// velocity must equal the DiT's on the same host tensors. GPU lane:
/// <c>Category=GpuIntegration&amp;FullyQualifiedName~SanaWmDenoiserUpload</c>.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class SanaWmDenoiserUploadGpuTests : IDisposable
{
    private const int Frames = 2, Height = 2, Width = 2, TextTokens = 4;
    private const float Tolerance = 1e-5f;
    private static readonly float[] ClipTimesteps = [21f, 500f, 300f, 900f];
    private static readonly float[] ChunkTimesteps = [0f, 727f];
    private readonly string _path = Path.Combine(Path.GetTempPath(), $"sanawm-upload-{Guid.NewGuid():N}.safetensors");

    public void Dispose()
    {
        if (File.Exists(_path))
            File.Delete(_path);
    }

    [CudaDeviceFact]
    public void ClipDenoiser_UploadsItsInputs_AndMatchesTheDit()
    {
        using CudaBackend backend = GpuDevices.OpenCuda();
        using SanaWmCheckpoint checkpoint = Checkpoint();
        SanaWmConfig config = checkpoint.Config;
        using SanaWmDit dit = new(backend, checkpoint);
        using SanaWmTextConditioning text = Text(config);
        using SanaWmCameraConditioning camera = Camera();
        using Tensor latents = Pattern([2, config.InChannels, Frames, Height, Width], 0.03f);
        using SanaWmDitDenoiser denoiser = new(backend, dit, text, camera);
        SanaWmGuidanceBranch[] branches = [SanaWmGuidanceBranch.Conditional, SanaWmGuidanceBranch.Unconditional];

        using Tensor first = denoiser.Predict(latents, ClipTimesteps, branches);
        // The second call reads the cached caption and camera uploads.
        using Tensor second = denoiser.Predict(latents, ClipTimesteps, branches);

        using Tensor captions = Join(text.Condition, text.Negative);
        using Tensor masks = Join(text.ConditionMask, text.NegativeMask);
        using Tensor plucker = Join(camera.ChunkPlucker, camera.ChunkPlucker);
        float[] conditions = [.. camera.Conditions, .. camera.Conditions];
        using Tensor expected = dit.Forward(latents, ClipTimesteps, captions, masks, conditions, plucker);

        AssertClose(expected, first, "first call");
        AssertClose(expected, second, "second call");
    }

    [CudaDeviceFact]
    public void ChunkDenoiser_UploadsItsInputs_AndMatchesTheDit()
    {
        using CudaBackend backend = GpuDevices.OpenCuda();
        using SanaWmCheckpoint checkpoint = Checkpoint();
        SanaWmConfig config = checkpoint.Config;
        using SanaWmDit dit = new(backend, checkpoint);
        using SanaWmTextConditioning text = Text(config);
        using SanaWmCameraConditioning camera = Camera();
        using Tensor latents = Pattern([1, config.InChannels, Frames, Height, Width], 0.03f);
        TokenWindowPolicy window = TokenWindowPolicy.RecentChunks(2, sink: true);
        using SanaWmDitChunkDenoiser denoiser = new(backend, dit, config, text, window);
        using SanaWmStreamCache cache = new(backend, config, Height, Width, window);

        foreach (SanaWmGuidanceBranch branch in (SanaWmGuidanceBranch[])[SanaWmGuidanceBranch.Conditional, SanaWmGuidanceBranch.Unconditional])
        {
            using Tensor actual = denoiser.Predict(latents, ChunkTimesteps, branch, camera.Conditions, camera.ChunkPlucker, commit: false);
            using Tensor expected = dit.ForwardChunk(latents, ChunkTimesteps, text.Features(branch), text.Mask(branch), camera.Conditions,
                camera.ChunkPlucker, cache, commit: false);
            AssertClose(expected, actual, $"{branch} branch");
        }
    }

    private SanaWmCheckpoint Checkpoint()
    {
        SanaWmSyntheticLayout.WriteFile(_path,
            config: SanaWmSyntheticLayout.Config with { PluckerChannels = SanaWmCameraConditioning.PluckerChannels });
        return SanaWmCheckpointConverter.Load(_path);
    }

    private static SanaWmTextConditioning Text(SanaWmConfig config)
    {
        long[] features = [1, 1, TextTokens, config.TextWidth];
        Tensor mask = Pattern([1, TextTokens], 0f, 1f);
        Tensor negativeMask = Pattern([1, TextTokens], 0f, 1f);
        negativeMask.AsSpan<float>()[TextTokens - 1] = 0f;
        return new SanaWmTextConditioning(Pattern(features, 0.02f), mask, Pattern(features, -0.015f), negativeMask);
    }

    private static SanaWmCameraConditioning Camera()
    {
        int rawFrames = (Frames - 1) * SanaWmCameraConditioning.TemporalStride + 1;
        float[][] poses = [.. Enumerable.Range(0, rawFrames)
            .Select(static i => new float[] { 1, 0, 0, 0.1f * i, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 })];
        int width = Width * SanaWmCameraConditioning.SpatialStride, height = Height * SanaWmCameraConditioning.SpatialStride;
        PinholeIntrinsics intrinsics = new(width, height, width / 2f, height / 2f);
        return SanaWmCameraConditioning.Prepare(poses, [intrinsics], CoverCropMapping.Create(width, height, width, height), width, height);
    }

    private static Tensor Pattern(ReadOnlySpan<long> dims, float scale, float offset = 0f)
    {
        Tensor tensor = new(new TensorShape(dims), DType.F32);
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++)
            values[i] = offset + ((i % 97) - 48) * scale;
        return tensor;
    }

    // Two one-item host tensors as a batch of two.
    private static Tensor Join(Tensor first, Tensor second)
    {
        long[] dims = new long[first.Shape.Rank];
        first.Shape.CopyDimsTo(dims);
        dims[0] = 2;
        Tensor joined = new(new TensorShape(dims), DType.F32);
        Span<float> values = joined.AsSpan<float>();
        first.AsReadOnlySpan<float>().CopyTo(values);
        second.AsReadOnlySpan<float>().CopyTo(values[(int)first.ElementCount..]);
        return joined;
    }

    private static void AssertClose(Tensor expected, Tensor actual, string what)
    {
        Assert.Equal(expected.Shape, actual.Shape);
        ReadOnlySpan<float> x = expected.AsReadOnlySpan<float>(), y = actual.AsReadOnlySpan<float>();
        float largest = 0f, difference = 0f;
        for (int i = 0; i < x.Length; i++)
        {
            Assert.True(float.IsFinite(y[i]), $"{what}: non-finite value at {i}");
            largest = MathF.Max(largest, MathF.Abs(x[i]));
            difference = MathF.Max(difference, MathF.Abs(x[i] - y[i]));
        }
        Assert.True(largest > 0f, $"{what}: the expected velocity is all zero");
        Assert.True(difference <= Tolerance, $"{what}: max difference {difference} above {Tolerance}");
    }
}
