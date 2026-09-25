using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.ModelAssets.BlockScale;
using HartsyInference.ModelAssets.Nvfp4;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Separates two things the existing bring-up gate cannot tell apart: whether the native NVFP4 GEMM computes
/// the wrong product, or whether it computes the right one and the comparison is simply measuring how lossy a 4-bit
/// activation is. The existing test compares W4A4 against W4A16, so its error is dominated by the activation
/// quantization itself. Here the reference is fed the SAME activation the native path quantized — decoded back on the
/// host — so the only remaining difference is the GEMM. A small error means the kernel is right and the other test's
/// budget was written too tight; a large one means the kernel is wrong.</summary>
[Collection("CudaSerial")]
public sealed class Nvfp4GemmReferenceTests
{
    private readonly ITestOutputHelper _output;
    public Nvfp4GemmReferenceTests(ITestOutputHelper output) => _output = output;

    private static readonly float[] E2M1 = [0f, 0.5f, 1f, 1.5f, 2f, 3f, 4f, 6f, -0f, -0.5f, -1f, -1.5f, -2f, -3f, -4f, -6f];

    private static float DecodeE4M3(byte b)
    {
        int s = (b >> 7) & 1, e = (b >> 3) & 0xF, m = b & 0x7;
        float mag = e == 0 ? MathF.ScaleB(m / 8f, -6) : MathF.ScaleB(1f + m / 8f, e - 7);
        return s == 1 ? -mag : mag;
    }

    [Theory]
    [InlineData(64, 256, 128)]
    [InlineData(200, 320, 256)]
    [InlineData(1024, 1024, 1024)]
    public unsafe void NativeNvfp4Gemm_MatchesTheSameActivationThroughTheUnpackPath(int m, int n, int k)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir)) ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        using CudaBackend backend = new CudaBackend(0, ptxDir);
        if (!backend.BlockScaledExecutor.IsSupported) { _output.WriteLine("SKIPPED: needs Blackwell"); return; }

        int paddedRows = (n + 127) / 128 * 128, paddedCols = (k / 16 + 3) / 4 * 4;
        int actPaddedRows = (m + 127) / 128 * 128;
        using Tensor packed = new Tensor(new TensorShape(n, k / 2), DType.U8);
        using Tensor scales = new Tensor(new TensorShape(paddedRows, paddedCols), DType.F8E4M3);
        using Tensor global = new Tensor(new TensorShape(1), DType.F32);
        using Tensor input = new Tensor(new TensorShape(m, k), DType.F32);
        using Tensor outNative = new Tensor(new TensorShape(m, n), DType.F16);
        using Tensor outRefSameAct = new Tensor(new TensorShape(m, n), DType.F16);
        using Tensor outRefFullAct = new Tensor(new TensorShape(m, n), DType.F16);

        Random rng = new Random(37);
        byte* wp = (byte*)packed.DataPointer;
        for (long i = 0; i < (long)n * (k / 2); i++) wp[i] = (byte)rng.Next(256);
        byte* sp = (byte*)scales.DataPointer;
        for (long i = 0; i < (long)paddedRows * paddedCols; i++) sp[i] = (byte)(0x20 + rng.Next(0x21));
        ((float*)global.DataPointer)[0] = 0.01f;
        for (long i = 0; i < (long)m * k; i++) ((float*)input.DataPointer)[i] = (float)(rng.NextDouble() * 8.0 - 4.0);
        Assert.True(Nvfp4Codec.TryAttachResident(packed, scales, global, hasPreQuantScale: false, out Tensor weight));

        backend.EnableNativeFp4Gemm = true;
        backend.Linear(outNative, input, weight, null);
        backend.Sync();

        // The exact activation the native path multiplied: quantize on the GPU, decode on the host.
        using Tensor actPacked = new Tensor(new TensorShape(m, k / 2), DType.U8);
        using Tensor actScales = new Tensor(new TensorShape(actPaddedRows, paddedCols), DType.F8E4M3);
        using Tensor actScalars = new Tensor(new TensorShape(4), DType.F32);
        backend.BlockQuantizeActivationForTest(actPacked, actScales, actScalars, input, weightScale: 1f);
        backend.Sync();
        float sf = ((float*)actScalars.DataPointer)[0];
        using Tensor inputQ = new Tensor(new TensorShape(m, k), DType.F32);
        byte* ap = (byte*)actPacked.DataPointer;
        byte* asp = (byte*)actScales.DataPointer;
        float* iq = (float*)inputQ.DataPointer;
        for (int r = 0; r < m; r++)
        {
            for (int c = 0; c < k; c++)
            {
                byte pair = ap[(long)r * (k / 2) + c / 2];
                float q = E2M1[c % 2 == 0 ? pair >> 4 : pair & 0xF];
                iq[(long)r * k + c] = q * DecodeE4M3(asp[BlockScaleSwizzle.SwizzledIndex(r, c / 16, paddedCols)]) * sf;
            }
        }

        backend.EnableNativeFp4Gemm = false;
        backend.Linear(outRefSameAct, inputQ, weight, null);   // same activation, unpacked weight
        backend.Linear(outRefFullAct, input, weight, null);    // full-precision activation: what the old test used
        backend.Sync();

        float Rel(Tensor a, Tensor b)
        {
            Half* x = (Half*)a.DataPointer; Half* y = (Half*)b.DataPointer;
            double num = 0, den = 0;
            for (long i = 0; i < (long)m * n; i++) { num += MathF.Abs((float)x[i] - (float)y[i]); den += MathF.Abs((float)y[i]); }
            return (float)(num / Math.Max(den, 1e-9));
        }
        float vsSameAct = Rel(outNative, outRefSameAct);
        float vsFullAct = Rel(outNative, outRefFullAct);
        float quantLoss = Rel(outRefSameAct, outRefFullAct);
        _output.WriteLine($"RESULT {m}x{n}x{k}: native-vs-same-activation={vsSameAct:E3}  native-vs-full-activation={vsFullAct:E3}  activation-quantization-loss-alone={quantLoss:E3}");
        Assert.True(vsSameAct < 2e-2f,
            $"native GEMM differs from the unpack path on the SAME activation by {vsSameAct:E3}; that is the kernel, not quantization");
    }
}
