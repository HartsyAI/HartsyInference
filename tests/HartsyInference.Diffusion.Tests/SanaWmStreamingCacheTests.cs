using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.SanaWm;
using HartsyInference.Diffusion.Streaming;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>SANA-WM streaming's carried temporal context against upstream's own cached ops, chunk by chunk
/// (<c>Fixtures/generate_sana_wm_streaming_cache_fixtures.py</c>, run on CUDA because FLA's short conv is Triton-only): the GDN
/// key short conv with the previous chunk's input frames as left context (<c>_cached_temporal_short_conv</c>, chunks of 4, 3
/// and 3 frames) and the FFN whose temporal conv reads the previous chunk's last frame (<c>CachedGLUMBConvTemp</c>, chunks of
/// 4 and 3). The engine runs both on the CPU backend in F32.</summary>
public sealed class SanaWmStreamingCacheTests
{
    private const int Height = 2, Width = 3, Channels = 8;
    private const int Spatial = Height * Width;

    [Fact]
    public void KeyShortConv_WithCarriedContext_MatchesUpstreamCachedShortConv()
    {
        using SafeTensorsLoader fixture = Load();
        using CpuBackend cpu = new();
        IBackend backend = cpu;
        using Tensor kernel = SanaWmTemporalConv.SymmetricKernel(fixture.GetTensor("tconv.weight"), Channels, "tconv.weight");
        using FrameTailCache tail = new(backend, SanaWmBlockStream.KeyTailFrames, 0, Spatial);
        int[] chunkFrames = [4, 3, 3];
        for (int chunk = 0; chunk < chunkFrames.Length; chunk++)
        {
            int frames = chunkFrames[chunk];
            Tensor input = fixture.GetTensor($"tconv.chunk{chunk}.input");
            using Tensor output = SanaWmTemporalConv.ApplyWithContext(backend, input, kernel, frames, Spatial, Channels, tail.Current,
                tail.HeldFrames);
            AssertClose($"key short conv, chunk {chunk}", fixture.GetTensor($"tconv.chunk{chunk}.output"), output, backend);
            tail.Stage(input);
            tail.Commit();
        }
    }

    [Fact]
    public void Ffn_WithCarriedLastFrame_MatchesUpstreamCachedGlumbConvTemp()
    {
        using SafeTensorsLoader fixture = Load();
        using CpuBackend cpu = new();
        IBackend backend = cpu;
        using SanaWmGlumbTempFfn ffn = new(fixture.GetTensor("ffn.inverted_conv.conv.weight"), fixture.GetTensor("ffn.inverted_conv.conv.bias"),
            fixture.GetTensor("ffn.depth_conv.conv.weight"), fixture.GetTensor("ffn.depth_conv.conv.bias"),
            fixture.GetTensor("ffn.point_conv.conv.weight"), fixture.GetTensor("ffn.t_conv.weight"));
        using FrameTailCache tail = new(backend, SanaWmBlockStream.FfnTailFrames, 0, Spatial);
        int[] chunkFrames = [4, 3];
        for (int chunk = 0; chunk < chunkFrames.Length; chunk++)
        {
            int frames = chunkFrames[chunk];
            Tensor flat = fixture.GetTensor($"ffn.chunk{chunk}.input");
            using Tensor input = flat.Reshape(new TensorShape(1, (long)frames * Spatial, Channels));
            using Tensor output = ffn.Forward(backend, input, frames, Height, Width, "ffn", tail, commit: true);
            AssertClose($"FFN, chunk {chunk}", fixture.GetTensor($"ffn.chunk{chunk}.output"), output, backend);
            tail.Commit();
        }
    }

    private static SafeTensorsLoader Load()
    {
        SafeTensorsLoader loader = new();
        loader.Load(Path.Combine(AppContext.BaseDirectory, "Fixtures", "sana_wm_streaming_caches.safetensors"));
        return loader;
    }

    // Upstream ran F32 on CUDA (TF32 off for convs is not guaranteed there), so the bar is a relative tolerance, not equality.
    private static void AssertClose(string label, Tensor expected, Tensor actual, IBackend backend)
    {
        backend.OffloadActivation(actual);
        ReadOnlySpan<float> e = expected.AsReadOnlySpan<float>(), a = actual.AsReadOnlySpan<float>();
        Assert.Equal(e.Length, a.Length);
        double scale = 0, worst = 0;
        for (int i = 0; i < e.Length; i++)
        {
            scale = Math.Max(scale, Math.Abs(e[i]));
            worst = Math.Max(worst, Math.Abs(e[i] - a[i]));
        }

        Assert.True(worst <= 2e-3 * Math.Max(scale, 1), $"{label}: max abs difference {worst:E3} against scale {scale:E3}");
    }
}
