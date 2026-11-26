using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Duo;
using HartsyInference.ModelAssets.Duo;
using HartsyInference.Tests.Shared;
using Xunit;

namespace HartsyInference.Diffusion.Tests.Duo;

/// <summary>The Duo audio stream decoder on CUDA against the CPU: the same blocks through the same pure fake window decoder publish the same
/// samples. The windows, the history and the crops are built by the backend, which is what differs between the two runs. Reported as skipped,
/// never passed, when no CUDA device opens; no released weights are run.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class DuoAudioStreamDecoderGpuTests
{
    private const int Width = 32, SamplesPerStep = 4, Blocks = 4, Lookahead = 8, Context = 16;
    private const float Tolerance = 1e-5f;
    private static readonly int BlockSteps = DuoCodecSpec.Get("duo_5_17").AudioLatents;

    [CudaDeviceFact]
    public void TheCudaStreamPublishesTheSameSamplesAsTheCpuStream()
    {
        float[] cpu = Publish(TestBackends.Cpu);
        using CudaBackend cuda = new(0, PtxDirectory());
        float[] gpu = Publish(cuda);

        Assert.Equal(cpu.Length, gpu.Length);
        for (int i = 0; i < cpu.Length; i++)
        {
            Assert.True(MathF.Abs(cpu[i] - gpu[i]) <= Tolerance, $"sample {i}: CPU {cpu[i]}, CUDA {gpu[i]}.");
        }
    }

    /// <summary>Streams <see cref="Blocks"/> blocks and ends the stream; returns every published block's samples, in order.</summary>
    private static float[] Publish(IBackend backend)
    {
        using DuoAudioStreamDecoder stream = new(backend, Decode, Width, SamplesPerStep, BlockSteps, Lookahead, Context);
        List<float> samples = [];
        for (int b = 0; b < Blocks; b++)
        {
            samples.AddRange(stream.OnBlock(Rows(b)));
        }
        samples.AddRange(stream.End());
        return samples.ToArray();
    }

    /// <summary>A pure function of the window: each output sample depends on the window's first two channels at its step, so a window or a crop
    /// built wrong changes the samples.</summary>
    private static Tensor Decode(Tensor window)
    {
        int steps = (int)window.Shape[3];
        float[] source = window.AsSpan<float>().ToArray();
        Tensor output = new(new TensorShape(new long[] { 1, 2, steps * SamplesPerStep }), DType.F32);
        Span<float> o = output.AsSpan<float>();
        for (int c = 0; c < 2; c++)
        {
            for (int s = 0; s < steps * SamplesPerStep; s++)
            {
                int t = s / SamplesPerStep;
                float a = source[((0 * 2) + c) * steps + t], b = source[((1 * 2) + c) * steps + t];
                o[c * steps * SamplesPerStep + s] = MathF.Tanh((a + b) * 0.25f + s * 0.01f);
            }
        }
        return output;
    }

    /// <summary>Block <paramref name="b"/> in [channel][step][width] order.</summary>
    private static float[] Rows(int b)
    {
        float[] rows = new float[2 * BlockSteps * Width];
        for (int i = 0; i < rows.Length; i++)
        {
            rows[i] = MathF.Sin(b * 1.7f + i * 0.13f);
        }
        return rows;
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
