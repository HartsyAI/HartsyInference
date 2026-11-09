using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.SanaWm;
using HartsyInference.Diffusion.Streaming;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary><see cref="SanaWmDit.ForwardChunk"/> on the synthetic layout, chunk by chunk through a
/// <see cref="SanaWmStreamCache"/>: the first chunk equals the whole-clip forward (nothing is carried yet), a later chunk reads
/// the carried state, denoise passes leave the cache untouched, and only the save pass plus <see cref="SanaWmStreamCache.Commit"/>
/// advance it.</summary>
public sealed class SanaWmStreamingDitTests : IDisposable
{
    private const int Height = 2, Width = 2, TextTokens = 4;
    private readonly string _path = Path.Combine(Path.GetTempPath(), $"sanawm-stream-{Guid.NewGuid():N}.safetensors");

    public void Dispose()
    {
        if (File.Exists(_path))
            File.Delete(_path);
    }

    [Fact]
    public void ForwardChunk_StreamsChunksThroughTheCache()
    {
        using CpuBackend cpu = new();
        IBackend backend = cpu;
        SanaWmSyntheticLayout.WriteFile(_path);
        using SanaWmCheckpoint checkpoint = SanaWmCheckpointConverter.Load(_path);
        SanaWmConfig config = checkpoint.Config;
        using SanaWmDit dit = new(backend, checkpoint);
        TokenWindowPolicy window = TokenWindowPolicy.RecentChunks(2, sink: true);
        using SanaWmStreamCache cache = new(backend, config, Height, Width, window);
        using Tensor text = Pattern([1, 1, TextTokens, config.TextWidth], 0.02f);
        using Tensor mask = Ones([1, TextTokens]);

        // Chunk sizes cover a chunk shorter than the key conv's three-frame tail and enough chunks to roll the softmax window.
        int[] chunkFrames = [2, 1, 1, 1, 2];
        int offset = 0;
        for (int chunk = 0; chunk < chunkFrames.Length; chunk++)
        {
            int frames = chunkFrames[chunk];
            using Tensor x = Pattern([1, config.InChannels, frames, Height, Width], 0.03f + 0.01f * chunk);
            using Tensor plucker = Pattern([1, config.PluckerChannels, frames, Height, Width], 0.05f);
            float[] conditions = CameraConditions(offset, frames);
            float[] timesteps = new float[frames];
            Array.Fill(timesteps, 727f);
            if (chunk == 0)
                timesteps[0] = 0f;

            using Tensor streamed = dit.ForwardChunk(x, timesteps, text, mask, conditions, plucker, cache, commit: false);
            AssertFinite(streamed);
            if (chunk == 0)
            {
                using Tensor clip = dit.Forward(x, timesteps, text, mask, conditions, plucker);
                AssertClose(clip, streamed, 1e-6f, "chunk 0 against the whole-clip forward");
            }
            else
            {
                using SanaWmStreamCache fresh = new(backend, config, Height, Width, window);
                using Tensor alone = dit.ForwardChunk(x, timesteps, text, mask, conditions, plucker, fresh, commit: false);
                Assert.True(MaxDiff(alone, streamed) > 1e-6f, $"chunk {chunk} does not read the carried state");
            }

            using (Tensor again = dit.ForwardChunk(x, timesteps, text, mask, conditions, plucker, cache, commit: false))
                AssertClose(streamed, again, 0f, $"chunk {chunk}, repeated denoise pass");

            Assert.Equal(offset, cache.CommittedFrames);
            dit.ForwardChunk(x, new float[frames], text, mask, conditions, plucker, cache, commit: true).Dispose();
            cache.Commit(frames);
            offset += frames;
            Assert.Equal(offset, cache.CommittedFrames);
        }
    }

    private static float[] CameraConditions(int offset, int frames)
    {
        float[] conditions = new float[frames * 20];
        for (int f = 0; f < frames; f++)
        {
            float[] pose = [1, 0, 0, 0.1f * (offset + f), 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
            pose.CopyTo(conditions, f * 20);
            conditions[f * 20 + 16] = 2f;
            conditions[f * 20 + 17] = 2f;
            conditions[f * 20 + 18] = 1f;
            conditions[f * 20 + 19] = 1f;
        }

        return conditions;
    }

    private static Tensor Pattern(ReadOnlySpan<long> dims, float scale)
    {
        Tensor tensor = new(new TensorShape(dims), DType.F32, DeviceKind.Cpu);
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++)
            values[i] = ((i % 97) - 48) * scale;
        return tensor;
    }

    private static Tensor Ones(ReadOnlySpan<long> dims)
    {
        Tensor tensor = new(new TensorShape(dims), DType.F32, DeviceKind.Cpu);
        tensor.AsSpan<float>().Fill(1f);
        return tensor;
    }

    private static float MaxDiff(Tensor a, Tensor b)
    {
        ReadOnlySpan<float> x = a.AsReadOnlySpan<float>(), y = b.AsReadOnlySpan<float>();
        Assert.Equal(x.Length, y.Length);
        float max = 0f;
        for (int i = 0; i < x.Length; i++)
            max = MathF.Max(max, MathF.Abs(x[i] - y[i]));
        return max;
    }

    private static void AssertClose(Tensor expected, Tensor actual, float tolerance, string what)
    {
        float diff = MaxDiff(expected, actual);
        Assert.True(diff <= tolerance, $"{what}: max difference {diff} above {tolerance}");
    }

    private static void AssertFinite(Tensor tensor)
    {
        foreach (float value in tensor.AsReadOnlySpan<float>())
            Assert.True(float.IsFinite(value), "non-finite value");
    }
}
