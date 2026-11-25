using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.FastFrames;
using HartsyInference.Tests.Shared;
using Xunit;

namespace HartsyInference.Engine.Tests;

/// <summary>Regression for the washed-out S output: the FastFrames pixel decoder on a CUDA backend must match CpuBackend across
/// a clip. Each test is a CUDA device fact, skipped where no CUDA device opens. The CPU backend has no device
/// copies, so a device-only aliasing bug cannot be reproduced on CPU; the CPU reference tests guard the math.</summary>
public sealed class FastFramesDecoderGpuParityTests
{
    private const int Frames = 3;
    private const int GridH = 4, GridW = 6;

    /// <summary>Plain F32 GEMMs (the exact lane): the decoder on CUDA matches CPU to rounding.</summary>
    [CudaDeviceFact]
    public void DecodeFrameOnCudaWithExactGemmsMatchesCpuAcrossFrames() => Compare(exactGemm: true, tolerance: 1e-5);

    /// <summary>The default lane a generation runs: the engine's TF32 GEMMs. Measured on a 4090 with the real S decoder: relative
    /// difference 1.2e-4 over three frames; the bound leaves room for that.</summary>
    [CudaDeviceFact]
    public void DecodeFrameOnCudaWithDefaultGemmsStaysWithinTheTf32Tolerance() => Compare(exactGemm: false, tolerance: 1e-3);

    private static void Compare(bool exactGemm, double tolerance)
    {
        FastFramesPixelDecoderConfig config = new(Dim: 64, InputDim: 64, LowBlocks: 1, HighBlocks: 1, FfnMultiplier: 4);
        Dictionary<string, Tensor> weights = RandomWeights(config, seed: 21);
        List<(float[] Hidden, float[] Temb)> inputs = RandomInputs(config, seed: 22);
        try
        {
            float[][] cpu = Run(new CpuBackend(), config, weights, inputs);
            using IBackend cuda = HartsyInference.Engine.BackendFactory.CreateCuda(0);
            // The exact lane turns TF32 off for this pass only; the default lane runs the engine's default, which is TF32.
            using IDisposable? exact = exactGemm ? new ExactGemmScope(cuda) : null;
            float[][] device = Run(cuda, config, weights, inputs);

            for (int f = 0; f < Frames; f++)
            {
                double meanAbs = device[f].Select(x => Math.Abs((double)x)).Average();
                Assert.True(meanAbs > 1e-3, $"frame {f}: CUDA output is near zero (mean |x| {meanAbs:G3}).");
                double rel = Relative(cpu[f], device[f]);
                Assert.True(rel <= tolerance, $"frame {f}: CUDA vs CPU relative difference {rel:G6} exceeds {tolerance:G1}.");
            }
        }
        finally
        {
            foreach (Tensor t in weights.Values)
            {
                t.Dispose();
            }
        }
    }

    private static float[][] Run(IBackend backend, FastFramesPixelDecoderConfig config, Dictionary<string, Tensor> weights,
        List<(float[] Hidden, float[] Temb)> inputs)
    {
        using FastFramesPixelDecoder decoder = new(config);
        decoder.LoadWeights(weights);
        using FastFramesPixelDecoder.State state = decoder.CreateState();
        float[][] outputs = new float[inputs.Count][];
        for (int f = 0; f < inputs.Count; f++)
        {
            using Tensor hidden = new(new TensorShape(GridH * GridW, config.InputDim), DType.F32);
            inputs[f].Hidden.AsSpan().CopyTo(hidden.AsSpan<float>());
            using Tensor temb = new(new TensorShape(1, config.InputDim), DType.F32);
            inputs[f].Temb.AsSpan().CopyTo(temb.AsSpan<float>());
            using Tensor pixels = decoder.DecodeFrame(backend, hidden, temb, GridH, GridW, state);
            outputs[f] = pixels.AsSpan<float>().ToArray();
        }
        return outputs;
    }

    private static List<(float[] Hidden, float[] Temb)> RandomInputs(FastFramesPixelDecoderConfig config, int seed)
    {
        Random rng = new(seed);
        List<(float[], float[])> inputs = [];
        for (int f = 0; f < Frames; f++)
        {
            inputs.Add((RandomValues(rng, GridH * GridW * config.InputDim, 1f), RandomValues(rng, config.InputDim, 0.5f)));
        }
        return inputs;
    }

    private static Dictionary<string, Tensor> RandomWeights(FastFramesPixelDecoderConfig config, int seed)
    {
        Random rng = new(seed);
        Dictionary<string, Tensor> weights = new(StringComparer.Ordinal);
        foreach ((string name, long[] shape) in FastFramesPixelDecoder.ExpectedShapes(config, prefixed: false))
        {
            Tensor tensor = new(new TensorShape(shape), DType.F32);
            RandomValues(rng, (int)tensor.Shape.ElementCount, 0.05f).AsSpan().CopyTo(tensor.AsSpan<float>());
            weights[name] = tensor;
        }
        return weights;
    }

    private static float[] RandomValues(Random rng, int count, float scale)
    {
        float[] values = new float[count];
        for (int i = 0; i < count; i++)
        {
            values[i] = (float)(rng.NextDouble() * 2 - 1) * scale;
        }
        return values;
    }

    private static double Relative(float[] reference, float[] actual)
    {
        double num = 0, den = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            double d = actual[i] - reference[i];
            num += d * d;
            den += (double)reference[i] * reference[i];
        }
        return Math.Sqrt(num / Math.Max(den, 1e-30));
    }
}
