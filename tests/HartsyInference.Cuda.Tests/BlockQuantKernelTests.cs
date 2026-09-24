using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.ModelAssets.BlockScale;
using HartsyInference.ModelAssets.Nvfp4;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>The block-scaled activation quantizer feeding <see cref="BlockScaledGemmExecutor"/>, checked on any CUDA card: the packed e2m1 bytes, the E4M3 block scales in cuBLASLt's blocked layout and the per-tensor scalars must decode back to the input, and the layout must be the one the engine's existing consumer of it — the resident nvfp4 unpack — already reads.</summary>
[Collection("CudaSerial")]
public sealed class BlockQuantKernelTests
{
    private readonly ITestOutputHelper _output;
    public BlockQuantKernelTests(ITestOutputHelper output) => _output = output;

    private static readonly float[] E2M1 =
    [
        +0.0f, +0.5f, +1.0f, +1.5f, +2.0f, +3.0f, +4.0f, +6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    ];

    private static string PtxDir()
    {
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        return Directory.Exists(ptxDir) ? ptxDir
            : Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
    }

    [Theory]
    [InlineData(3, 64, false)]
    [InlineData(200, 320, false)]
    [InlineData(129, 1024, true)]
    public unsafe void QuantizedActivationDecodesBackToTheInput(int rows, int cols, bool f16)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        using CudaBackend backend = new CudaBackend(0, PtxDir());

        int paddedRows = (rows + 127) / 128 * 128, paddedCols = (cols / 16 + 3) / 4 * 4;
        using Tensor input = new Tensor(new TensorShape(rows, cols), f16 ? DType.F16 : DType.F32);
        using Tensor packed = new Tensor(new TensorShape(rows, cols / 2), DType.U8);
        using Tensor scales = new Tensor(new TensorShape(paddedRows, paddedCols), DType.F8E4M3);
        using Tensor scalars = new Tensor(new TensorShape(3), DType.F32);
        Random rng = new Random(29);
        float[] x = new float[rows * cols];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 8.0 - 4.0);
        x[rows * cols / 3] = -37.5f;   // the tensor amax, negative on purpose
        for (int i = 0; i < x.Length; i++)
        {
            if (f16) ((Half*)input.DataPointer)[i] = (Half)x[i]; else ((float*)input.DataPointer)[i] = x[i];
            if (f16) x[i] = (float)(Half)x[i];
        }
        const float weightScale = 0.75f;

        backend.BlockQuantizeActivationForTest(packed, scales, scalars, input, weightScale);
        backend.Sync();

        float sf = ((float*)scalars.DataPointer)[0];
        Assert.True(MathF.Abs(sf - 37.5f / (448f * 6f)) < 1e-7f, $"sf {sf} != amax/(448·6)");
        Assert.True(MathF.Abs(((float*)scalars.DataPointer)[1] - weightScale * sf) < 1e-7f, "alpha != weightScale·sf");
        Assert.Equal(0f, ((float*)scalars.DataPointer)[2]);

        byte* p = (byte*)packed.DataPointer;
        byte* s = (byte*)scales.DataPointer;
        double sxy = 0, sxx = 0, syy = 0;
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                byte scaleByte = s[BlockScaleSwizzle.SwizzledIndex(r, c / 16, paddedCols)];
                Assert.True((scaleByte & 0x7F) <= 0x7E, $"scale byte 0x{scaleByte:X2} at ({r},{c / 16}) is the NaN code");
                byte pair = p[(long)r * (cols / 2) + c / 2];
                float q = E2M1[c % 2 == 0 ? pair >> 4 : pair & 0xF];
                float v = q * DecodeE4M3(scaleByte) * sf;
                float t = x[r * cols + c];
                Assert.False(float.IsNaN(v), $"NaN at ({r},{c})");
                sxy += (double)v * t; sxx += (double)v * v; syy += (double)t * t;
            }
        }
        double corr = sxy / Math.Sqrt(sxx * syy);
        _output.WriteLine($"{rows}x{cols} {(f16 ? "f16" : "f32")}: corr={corr:F5} sf={sf:E4}");
        Assert.True(corr > 0.99, $"decoded activation correlates {corr:F4} with the input; nvfp4's own error is ~0.995");

        // The padding of the blocked layout is never written by the quantizer and must read as zero.
        for (int r = 0; r < paddedRows; r++)
        {
            for (int bc = 0; bc < paddedCols; bc++)
            {
                if (r < rows && bc < cols / 16) continue;
                Assert.Equal(0, s[BlockScaleSwizzle.SwizzledIndex(r, bc, paddedCols)]);
            }
        }
    }

    /// <summary>The quantizer's output, presented as a resident nvfp4 weight, must be what the engine's existing
    /// reader of that layout — the resident unpack behind <c>backend.Linear</c> — decodes. A K×K identity activation
    /// makes the F32 output the dequantized weight transposed, so the comparison is against the host decode
    /// through the same kernel every nvfp4 checkpoint takes.</summary>
    [Fact]
    public unsafe void QuantizedActivationIsWhatTheResidentNvfp4PathReads()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        using CudaBackend backend = new CudaBackend(0, PtxDir());

        const int rows = 129, cols = 1024;
        int paddedRows = (rows + 127) / 128 * 128, paddedCols = (cols / 16 + 3) / 4 * 4;
        using Tensor input = new Tensor(new TensorShape(rows, cols), DType.F32);
        using Tensor packed = new Tensor(new TensorShape(rows, cols / 2), DType.U8);
        using Tensor scales = new Tensor(new TensorShape(paddedRows, paddedCols), DType.F8E4M3);
        using Tensor scalars = new Tensor(new TensorShape(3), DType.F32);
        Random rng = new Random(31);
        for (int i = 0; i < rows * cols; i++) ((float*)input.DataPointer)[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        backend.BlockQuantizeActivationForTest(packed, scales, scalars, input, weightScale: 1f);
        backend.Sync();
        float sf = ((float*)scalars.DataPointer)[0];

        using Tensor global = new Tensor(new TensorShape(1), DType.F32);
        ((float*)global.DataPointer)[0] = sf;
        Assert.True(Nvfp4Codec.TryAttachResident(packed, scales, global, hasPreQuantScale: false, out Tensor resident),
            "the quantizer's output is not a weight the resident path accepts");
        using Tensor identity = new Tensor(new TensorShape(cols, cols), DType.F32);
        for (int i = 0; i < cols; i++) ((float*)identity.DataPointer)[(long)i * cols + i] = 1f;
        using Tensor output = new Tensor(new TensorShape(cols, rows), DType.F32);
        backend.Linear(output, identity, resident, bias: null);
        backend.Sync();

        byte* p = (byte*)packed.DataPointer;
        byte* s = (byte*)scales.DataPointer;
        float* o = (float*)output.DataPointer;
        float maxErr = 0f;
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                byte pair = p[(long)r * (cols / 2) + c / 2];
                float expected = E2M1[c % 2 == 0 ? pair >> 4 : pair & 0xF]
                    * DecodeE4M3(s[BlockScaleSwizzle.SwizzledIndex(r, c / 16, paddedCols)]) * sf;
                float actual = o[(long)c * rows + r];
                float err = MathF.Abs(actual - expected);
                if (err > maxErr) maxErr = err;
                // The unpack materializes to 16-bit before the GEMM, so the output carries that rounding and nothing else.
                Assert.True(err <= MathF.Abs(expected) * 1e-2f + 1e-6f, $"({r},{c}): {actual} vs {expected}");
            }
        }
        _output.WriteLine($"identity round-trip max abs err {maxErr:E3}");
    }

    private static float DecodeE4M3(byte b)
    {
        int sign = (b & 0x80) != 0 ? -1 : 1;
        int expField = (b >> 3) & 0xF;
        int mant = b & 0x7;
        float value = expField == 0 ? mant * MathF.Pow(2f, -9) : (1f + mant / 8f) * MathF.Pow(2f, expField - 7);
        return sign * value;
    }
}
