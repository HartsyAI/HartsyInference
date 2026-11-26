using HartsyInference.Audio.Models.VoiceIsolator;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests.VoiceIsolator;

/// <summary>The stream attention cache on a device against the same stream on the CPU: a run long enough to grow the cache past its first
/// 64 slots and to compact it at the capacity. Also checks that the cache stays device-current (its host copy is not read back).
/// Each test reports itself as skipped when its device is absent.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class VoiceIsolatorStreamCacheGpuTests(ITestOutputHelper output)
{
    private const int Bands = 2;
    private const int Heads = 2;
    private const int HeadDim = 8;
    private const int Frames = 200;
    private const int Block = 4;
    private const int Cache = 80;     // grows past 64 slots; capacity = 120 forces a compaction every 40 frames
    private const float Tol = 1e-4f;

    [CudaFact]
    public void Cuda_StreamCacheMatchesTheCpuStream()
    {
        using CudaBackend cuda = new(0, PtxDirectory());
        CompareWithCpu(cuda);
    }

    [VulkanFact]
    public void Vulkan_StreamCacheMatchesTheCpuStream()
    {
        using VulkanBackend vulkan = new();
        CompareWithCpu(vulkan);
    }

    private void CompareWithCpu(IBackend device)
    {
        Random rng = new(19);
        float[] q = Values(rng, Bands * Heads * Frames * HeadDim);
        float[] k = Values(rng, Bands * Heads * Frames * HeadDim);
        float[] v = Values(rng, Bands * Heads * Frames * HeadDim);
        float[] cpu = Stream(new CpuBackend(), q, k, v, out _);
        float[] gpu = Stream(device, q, k, v, out bool hostCurrent);
        Assert.Equal(cpu.Length, gpu.Length);
        float worst = 0f;
        for (int i = 0; i < cpu.Length; i++) worst = Math.Max(worst, Math.Abs(cpu[i] - gpu[i]));
        output.WriteLine($"{device.GetType().Name}: {Frames} frames, cache {Cache}, max abs difference {worst:E3}, "
            + $"cache host-current after the run: {hostCurrent}");
        Assert.True(worst < Tol, $"max abs difference {worst} exceeds {Tol}");
        // Growth and compaction ran through the backend; a host read of the cache would make it host-current again.
        Assert.False(hostCurrent, "the stream cache was read back to the host during the run");
    }

    /// <summary>Runs the stream over every block; <paramref name="hostCurrent"/> reports whether the lane's keys still hold a
    /// host-current copy (true on the CPU; false once a device op has written the cache and nothing read it back).</summary>
    private static float[] Stream(IBackend backend, float[] q, float[] k, float[] v, out bool hostCurrent)
    {
        VoiceIsolatorStreamAttention attention = new(Cache, Cache + Math.Max(Cache / 2, Block), 0, HeadDim);
        float[] streamed = new float[q.Length];
        using VoiceIsolatorAttentionLane lane = new();
        for (int start = 0; start < Frames; start += Block)
        {
            using Tensor qb = Block4(backend, q, start);
            using Tensor kb = Block4(backend, k, start);
            using Tensor vb = Block4(backend, v, start);
            Tensor attended = attention.Attend(backend, lane, qb, kb, vb, Bands, Heads, Block, HeadDim);
            float[] values = attended.AsReadOnlySpan<float>().ToArray();
            for (int b = 0; b < Bands * Heads; b++)
            {
                Array.Copy(values, b * Block * HeadDim, streamed, (b * Frames + start) * HeadDim, Block * HeadDim);
            }
            attended.Dispose();
        }
        // Probe before anything reads the keys back: a CPU lane is host-current by construction.
        hostCurrent = lane.Keys!.HostBytesCurrent;
        return streamed;
    }

    private static Tensor Block4(IBackend backend, float[] source, int start)
    {
        Tensor t = new(new TensorShape(Bands, Heads, Block, HeadDim), DType.F32);
        float[] block = new float[Bands * Heads * Block * HeadDim];
        for (int b = 0; b < Bands * Heads; b++)
        {
            Array.Copy(source, (b * Frames + start) * HeadDim, block, b * Block * HeadDim, Block * HeadDim);
        }
        block.CopyTo(t.AsSpan<float>());
        return t;
    }

    private static float[] Values(Random rng, int n)
    {
        float[] values = new float[n];
        for (int i = 0; i < n; i++) values[i] = (float)(rng.NextDouble() * 2 - 1);
        return values;
    }

    private static string PtxDirectory()
    {
        for (DirectoryInfo? dir = new(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "src", "HartsyInference.Cuda", "Ptx");
            if (Directory.Exists(candidate)) return candidate;
        }
        throw new DirectoryNotFoundException("HartsyInference.Cuda/Ptx was not found above the test binary.");
    }
}
