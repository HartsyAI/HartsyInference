using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Tests.Shared;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>One INT8 linear of the avatar DiT at a time, on the CUDA resident path against three references in double: the exact q·s with the
/// activations as they are, the upstream rule bf16(bf16(s)·q), and the exact q·s with the activations quantized per row to int8 (the W8A8 model).
/// Measurement only: the figures go to the test output; the bound is set in the status note, not here.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class LongCatInt8LinearOpTests
{
    private const int Tokens = 32;

    /// <summary>The correctness bound of this test, relative RMS of the CUDA INT8 output against the per-row int8 activation model (exact q·s, the
    /// activations quantized per row to int8). Written before its run. The measured figures are 3.7e-8 to 5.7e-8 over the 40 avatar linears.</summary>
    internal const double ModelRelativeBound = 1e-6;
    private readonly ITestOutputHelper _output;

    public LongCatInt8LinearOpTests(ITestOutputHelper output) => _output = output;

    [CudaDeviceFact]
    public void Cuda_Int8Linear_AvatarShapes_ErrorBreakdown()
    {
        Dictionary<string, Tensor> weights = LongCatTestData.AvatarWeights(LongCatTinyAvatarRelease.Config, seed: 71);
        Dictionary<string, Tensor> quantized = LongCatTestData.QuantizePerRow(weights);
        using CudaBackend backend = GpuDevices.OpenCuda();
        List<string> failures = [];
        _output.WriteLine($"{"linear",-40} {"shape",-14} {"exact",10} {"upstream",10} {"a8-model",10} {"clip-128",9}");

        foreach (string key in weights.Keys.OrderBy(k => k, StringComparer.Ordinal))
        {
            if (!key.EndsWith(".weight", StringComparison.Ordinal) || weights[key].Shape.Rank != 2) continue;
            string stem = key[..^".weight".Length];
            if (!quantized.TryGetValue(stem + ".weight_int8", out Tensor? int8) || !quantized.TryGetValue(stem + ".weight_scale", out Tensor? scale))
                continue;
            int n = (int)weights[key].Shape[0], k = (int)weights[key].Shape[1];
            float[] x = Activations(Tokens, k, seed: n * 31 + k);
            float[] values = Values(int8);
            float[] scales = scale.AsReadOnlySpan<float>().ToArray();
            float[] exact = Dequant(values, scales, n, k, upstream: false);
            float[] upstream = Dequant(values, scales, n, k, upstream: true);
            float[] xq = ActQuant(x, Tokens, k);

            int8.QuantInfo = new QuantWeightInfo { Format = "int8_tensorwise", RowScale = scale };
            using Tensor input = new(new TensorShape(Tokens, k), DType.F32);
            x.AsSpan().CopyTo(input.AsSpan<float>());
            using Tensor output = new(new TensorShape(Tokens, n), DType.F32);
            backend.Linear(output, input, int8, null);
            float[] device = output.AsReadOnlySpan<float>().ToArray();

            double eExact = RelRms(device, MatMul(x, exact, Tokens, n, k));
            double eUpstream = RelRms(device, MatMul(x, upstream, Tokens, n, k));
            double eActivation = RelRms(device, MatMul(xq, exact, Tokens, n, k));
            int clipped = values.Count(v => v == -128f);
            _output.WriteLine($"{stem,-40} {$"{n}x{k}",-14} {eExact,10:E3} {eUpstream,10:E3} {eActivation,10:E3} {clipped,9}");
            if (double.IsNaN(eExact) || !(eActivation <= ModelRelativeBound)) failures.Add($"{stem} ({n}x{k}): {eActivation:E3}");
        }
        Assert.True(failures.Count == 0, $"relative RMS against the per-row int8 activation model above {ModelRelativeBound:E0}: " + string.Join("; ", failures));
    }

    private static float[] Activations(int rows, int cols, int seed)
    {
        Random rng = new(seed);
        return LongCatTestData.RandomValues(rng, rows * cols);
    }

    private static float[] Values(Tensor int8)
    {
        ReadOnlySpan<sbyte> q = MemoryMarshal.Cast<byte, sbyte>(int8.AsSpan<byte>());
        float[] values = new float[q.Length];
        for (int i = 0; i < q.Length; i++) values[i] = q[i];
        return values;
    }

    /// <summary>Exact q·s in double, or the upstream form bf16(bf16(s)·q) with both roundings to BF16 (round to nearest even).</summary>
    private static float[] Dequant(float[] q, float[] s, int n, int k, bool upstream)
    {
        float[] w = new float[n * k];
        for (int r = 0; r < n; r++)
            for (int c = 0; c < k; c++)
                w[r * k + c] = upstream ? Bf16(Bf16(s[r]) * q[r * k + c]) : (float)(q[r * k + c] * (double)s[r]);
        return w;
    }

    /// <summary>Per-row symmetric int8 of the activations, dequantized (the model of a W8A8 path).</summary>
    private static float[] ActQuant(float[] x, int rows, int cols)
    {
        float[] y = new float[x.Length];
        for (int r = 0; r < rows; r++)
        {
            float max = 0f;
            for (int c = 0; c < cols; c++) max = MathF.Max(max, MathF.Abs(x[r * cols + c]));
            float step = max / 127f;
            for (int c = 0; c < cols; c++) y[r * cols + c] = step == 0f ? 0f : MathF.Round(x[r * cols + c] / step) * step;
        }
        return y;
    }

    private static float[] MatMul(float[] x, float[] w, int rows, int n, int k)
    {
        float[] y = new float[rows * n];
        for (int r = 0; r < rows; r++)
            for (int o = 0; o < n; o++)
            {
                double acc = 0;
                for (int c = 0; c < k; c++) acc += x[r * k + c] * (double)w[o * k + c];
                y[r * n + o] = (float)acc;
            }
        return y;
    }

    private static double RelRms(float[] a, float[] reference)
    {
        double num = 0, den = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double d = a[i] - reference[i];
            num += d * d;
            den += reference[i] * (double)reference[i];
        }
        return Math.Sqrt(num / den);
    }

    private static float Bf16(float value)
    {
        uint bits = BitConverter.SingleToUInt32Bits(value);
        uint lsb = (bits >> 16) & 1u;
        return BitConverter.UInt32BitsToSingle((bits + 0x7FFFu + lsb) & 0xFFFF0000u);
    }
}
