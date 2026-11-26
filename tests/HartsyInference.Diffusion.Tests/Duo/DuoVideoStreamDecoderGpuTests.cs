using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.ModelAssets.Duo;
using HartsyInference.Tests.Shared;
using Xunit;

namespace HartsyInference.Diffusion.Tests.Duo;

/// <summary>The Duo video stream decoder on CUDA against the CPU: the same chunks through the same fake decoder publish the same RGB bytes.
/// The fake decoder is pure arithmetic, so the only thing that differs between the two runs is where the chunks, windows and crops live.
/// Reported as skipped, never passed, when no CUDA device opens; no released weights are run.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class DuoVideoStreamDecoderGpuTests
{
    private const int Height = 2, Width = 2, Plane = Height * Width, Chunks = 3;

    [CudaDeviceFact]
    public void TheCudaStreamPublishesTheSameBytesAsTheCpuStream()
    {
        byte[] cpu = Publish(TestBackends.Cpu);
        using CudaBackend cuda = new(0, PtxDirectory());
        byte[] gpu = Publish(cuda);

        Assert.Equal(cpu.Length, gpu.Length);
        Assert.Equal(cpu, gpu);
    }

    /// <summary>Streams <see cref="Chunks"/> chunks and ends the stream; returns every published block's RGB in order.</summary>
    private static byte[] Publish(IBackend backend)
    {
        using DuoVideoStreamDecoder decoder = new(backend, Decode);
        List<byte> rgb = [];
        int blocks = 0;
        for (int i = 0; i < Chunks; i++)
        {
            using Tensor chunk = Chunk(i);
            foreach (DuoDecodedBlock block in decoder.OnChunk(chunk))
            {
                rgb.AddRange(block.Rgb);
                blocks++;
            }
        }
        foreach (DuoDecodedBlock block in decoder.End())
        {
            rgb.AddRange(block.Rgb);
            blocks++;
        }
        Assert.Equal(Chunks, blocks);
        return rgb.ToArray();
    }

    /// <summary>A pure function of the window: every output pixel depends on the window's first value, so a window built wrong changes the bytes.</summary>
    private static Tensor Decode(Tensor window)
    {
        int frames = DuoVideoDecodePlan.RawFramesFor((int)window.Shape[2]);
        int steps = (int)window.Shape[2], h = (int)window.Shape[3], w = (int)window.Shape[4], plane = h * w;
        float[] source = window.AsSpan<float>().ToArray();
        Tensor output = new(new TensorShape(new long[] { 1, 3, frames, h, w }), DType.F32);
        Span<float> o = output.AsSpan<float>();
        for (int c = 0; c < 3; c++)
        {
            for (int f = 0; f < frames; f++)
            {
                for (int p = 0; p < plane; p++)
                {
                    float value = source[((c * steps) + Math.Min(f, steps - 1)) * plane + p];
                    o[((c * frames) + f) * plane + p] = MathF.Tanh(value * 0.25f + f * 0.01f);
                }
            }
        }
        return output;
    }

    private static Tensor Chunk(int index)
    {
        Tensor chunk = new(new TensorShape(new long[] { 1, 24, 5, Height, Width }), DType.F32);
        Span<float> s = chunk.AsSpan<float>();
        for (int c = 0; c < 24; c++)
        {
            for (int t = 0; t < 5; t++)
            {
                for (int p = 0; p < Plane; p++)
                {
                    s[((c * 5) + t) * Plane + p] = index * 1000 + c * 100 + t * 10 + p;
                }
            }
        }
        return chunk;
    }

    private static string PtxDirectory()
    {
        for (DirectoryInfo? dir = new(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "src", "HartsyInference.Cuda", "Ptx");
            if (Directory.Exists(candidate))
            {
                return candidate;
            }
        }
        throw new DirectoryNotFoundException("Could not find src/HartsyInference.Cuda/Ptx above the test binary.");
    }
}
