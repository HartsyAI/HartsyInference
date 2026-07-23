using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.Gguf.Codecs;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Ground-truth correctness for the int8-activation (Q8_1) dp4a GEMV paths
/// (<see cref="CudaBackend.EnableDp4aGemv"/>, default ON, decode M&lt;=8). Unlike the float-path <see cref="FusedGemvGroundTruthTests"/>, the dp4a path is LOSSY:
/// the F32 activation is rounded to per-32-block int8 before the dot product. Two independent gates per case:
/// (1) an exact-simulation reference — the CPU replicates the activation quantization bit-for-bit
/// (amax/127 scale, round-to-nearest-even, clamp) and dots the dequantized weight against the
/// dequantized-int8 activation, so the only residual is F32 accumulation order → tight tolerance; and
/// (2) an analytic error bound vs the UNQUANTIZED reference — per element the Q8_1 rounding error is at most
/// scale/2, so |gpu - exactRef| ≤ Σ_i |w_i|·(scale_block(i)/2) computed from the actual data, a derived
/// bound rather than a guessed tolerance. Also asserts the dp4a branch actually engaged (its output must
/// differ from the float kernel's — if bit-identical, the env-flag dispatch silently fell through).</summary>
[Collection("CudaSerial")]
public sealed class Dp4aGemvGroundTruthTests
{
    private readonly ITestOutputHelper _output;
    public Dp4aGemvGroundTruthTests(ITestOutputHelper output) => _output = output;

    [Theory]
    [InlineData("Q4_K", 256, 64, 1, false)]
    [InlineData("Q4_K", 2560, 320, 1, false)]   // Qwen3-4B hidden size (real production K)
    [InlineData("Q4_K", 512, 96, 4, true)]      // batched decode + bias
    [InlineData("Q8_0", 32, 8, 1, false)]       // single-block minimal
    [InlineData("Q8_0", 2048, 256, 1, false)]   // Llama-3.2-1B hidden size (real production K)
    [InlineData("Q8_0", 224, 96, 4, true)]      // K%32 but not %128 (tail-masked groups) + batch + bias
    [InlineData("Q6_K", 256, 64, 1, false)]     // single-super-block minimal
    [InlineData("Q6_K", 2560, 320, 1, false)]   // Qwen3-4B ffn_down/lm_head K
    [InlineData("Q6_K", 512, 96, 4, true)]      // batched decode + bias
    public unsafe void Dp4aGemv_MatchesExactSimulationAndErrorBound(
        string dtypeName, int inDim, int outDim, int batch, bool withBias)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        DType quantDtype = dtypeName switch
        {
            "Q4_K" => DType.Q4_K,
            "Q6_K" => DType.Q6_K,
            "Q8_0" => DType.Q8_0,
            _ => throw new ArgumentOutOfRangeException(nameof(dtypeName)),
        };

        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");

        using CudaBackend backend = new(0, ptxDir);
        Tensor input = new(new TensorShape(batch, inDim), DType.F32);
        Tensor weightF32 = new(new TensorShape(outDim, inDim), DType.F32);
        Tensor weightQuant = new(new TensorShape(outDim, inDim), quantDtype);
        Tensor? bias = withBias ? new Tensor(new TensorShape(outDim), DType.F32) : null;
        Tensor outDp4a = new(new TensorShape(batch, outDim), DType.F32);
        Tensor outFloat = new(new TensorShape(batch, outDim), DType.F32);
        try
        {
            Random rng = new(29);
            float* ip = (float*)input.DataPointer;
            for (long i = 0; i < (long)batch * inDim; i++) ip[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * 0.5);
            float* wp = (float*)weightF32.DataPointer;
            long wc = (long)outDim * inDim;
            for (long i = 0; i < wc; i++) wp[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * 0.3);
            GgufCodecRegistry.Get(quantDtype).QuantizeFromF32(wp, (byte*)weightQuant.DataPointer, wc);
            if (bias is not null)
            {
                float* bp = (float*)bias.DataPointer;
                for (int i = 0; i < outDim; i++) bp[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * 0.8);
            }

            Assert.True(backend.EnableDp4aGemv, "dp4a GEMV should be ON by default (standard profile)");
            backend.Linear(outDp4a, input, weightQuant, bias);
            backend.Sync();
            backend.EnableDp4aGemv = false;
            backend.Linear(outFloat, input, weightQuant, bias);
            backend.Sync();

            using Tensor weightDequant = GgufDequantizer.Dequantize(weightQuant, DType.F32);
            float* dw = (float*)weightDequant.DataPointer;
            float* bp2 = bias is null ? null : (float*)bias.DataPointer;

            // CPU replica of quantize_activation_q8_1_f32: per-32-block amax/127 scale,
            // round-to-nearest-even, clamp to [-127,127].
            int kblocks = inDim / 32;
            float[] xd = new float[batch * kblocks];
            sbyte[] xq = new sbyte[batch * inDim];
            for (int m = 0; m < batch; m++)
            {
                for (int b = 0; b < kblocks; b++)
                {
                    float amax = 0f;
                    for (int i = 0; i < 32; i++) amax = MathF.Max(amax, MathF.Abs(ip[(long)m * inDim + b * 32 + i]));
                    float scale = amax / 127.0f;
                    float inv = scale > 0f ? 1.0f / scale : 0f;
                    xd[m * kblocks + b] = scale;
                    for (int i = 0; i < 32; i++)
                    {
                        float v = ip[(long)m * inDim + b * 32 + i];
                        int q = (int)MathF.Round(v * inv, MidpointRounding.ToEven);
                        q = Math.Clamp(q, -127, 127);
                        xq[(long)m * inDim + b * 32 + i] = (sbyte)q;
                    }
                }
            }

            double sumExact = 0.0, sumLossless = 0.0;
            float maxExact = 0f;
            int boundViolations = 0;
            float maxDp4aVsFloatDiff = 0f;
            for (int m = 0; m < batch; m++)
            {
                for (int n = 0; n < outDim; n++)
                {
                    double exactRef = bp2 is null ? 0.0 : bp2[n];      // dequantized-int8 activation
                    double floatRef = bp2 is null ? 0.0 : bp2[n];      // unquantized activation
                    double errBound = 0.0;
                    for (int k = 0; k < inDim; k++)
                    {
                        double w = dw[(long)n * inDim + k];
                        float scale = xd[m * kblocks + k / 32];
                        exactRef += w * (scale * xq[(long)m * inDim + k]);
                        floatRef += w * ip[(long)m * inDim + k];
                        errBound += Math.Abs(w) * (scale * 0.5);
                    }
                    float gpu = ((float*)outDp4a.DataPointer)[m * outDim + n];
                    float errExact = (float)Math.Abs(exactRef - gpu);
                    sumExact += errExact;
                    if (errExact > maxExact) maxExact = errExact;
                    double errVsFloat = Math.Abs(floatRef - gpu);
                    sumLossless += errVsFloat;
                    // 1e-4 slack absorbs F32 accumulation-order noise on top of the analytic rounding bound.
                    if (errVsFloat > errBound + 1e-4) boundViolations++;
                    float diff = MathF.Abs(gpu - ((float*)outFloat.DataPointer)[m * outDim + n]);
                    if (diff > maxDp4aVsFloatDiff) maxDp4aVsFloatDiff = diff;
                }
            }
            float avgExact = (float)(sumExact / (batch * outDim));
            _output.WriteLine($"{dtypeName} K={inDim} N={outDim} M={batch} bias={withBias}: " +
                $"exact-sim avg_err={avgExact:E3} max_err={maxExact:E3}, " +
                $"lossy avg_err vs f32 ref={(float)(sumLossless / (batch * outDim)):E3}, " +
                $"bound violations={boundViolations}, dp4a-vs-floatkernel max diff={maxDp4aVsFloatDiff:E3}");

            // Gate 1: exact simulation — only F32 accumulation order differs. Empirically ~1e-6; 5e-4 is
            // ~3 orders above the noise floor and ~1-2 orders below the lossy Q8_1 error a layout bug
            // would surface as.
            Assert.True(avgExact < 5e-4f,
                $"{dtypeName}: dp4a output disagrees with the exact CPU simulation (avg_err={avgExact:E3}) — layout/scale bug, not quantization loss");
            // Gate 2: the analytic per-element rounding bound vs the unquantized reference.
            Assert.True(boundViolations == 0,
                $"{dtypeName}: {boundViolations} outputs exceed the derived Q8_1 rounding-error bound");
            // Gate 3: the dp4a branch actually ran (dispatch didn't silently fall through to the float kernel).
            Assert.True(maxDp4aVsFloatDiff > 0f,
                $"{dtypeName}: dp4a and float-kernel outputs are bit-identical — EnableDp4aGemv dispatch did not engage");
        }
        finally
        {
            input.Dispose(); weightF32.Dispose(); weightQuant.Dispose(); outDp4a.Dispose(); outFloat.Dispose(); bias?.Dispose();
        }
    }
}
