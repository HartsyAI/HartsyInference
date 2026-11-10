using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Core.Tests.Gemm;

/// <summary>A BF16 output must be written as BF16 (2 bytes per element), not as F32 as the old cuBLAS C-type mapping did.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class BatchedMatMulBf16GpuTests
{
    private const int Batch = 2, M = 3, K = 5, N = 4;

    private readonly ITestOutputHelper _output;
    public BatchedMatMulBf16GpuTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void BatchedMatMul_Bf16Output_MatchesReferenceElementwise()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        // Small integers are exact in BF16 and in the F32 accumulator, so the reference is bit-exact.
        using CudaBackend backend = new(0, PtxDir());
        using Tensor a = new(new TensorShape(Batch, M, K), DType.BF16);
        using Tensor b = new(new TensorShape(Batch, K, N), DType.BF16);
        using Tensor output = new(new TensorShape(Batch, M, N), DType.BF16);

        Span<ushort> aBits = a.AsSpan<ushort>();
        Span<ushort> bBits = b.AsSpan<ushort>();
        for (int i = 0; i < aBits.Length; i++) aBits[i] = Bf16Bits((i % 7) - 3);
        for (int i = 0; i < bBits.Length; i++) bBits[i] = Bf16Bits((i % 5) - 2);

        ushort[] expected = new ushort[Batch * M * N];
        for (int bi = 0; bi < Batch; bi++)
        for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
        {
            float sum = 0f;
            for (int kk = 0; kk < K; kk++)
                sum += FromBf16(aBits[(bi * M + i) * K + kk]) * FromBf16(bBits[(bi * K + kk) * N + j]);
            expected[(bi * M + i) * N + j] = Bf16Bits(sum);
        }

        backend.BatchedMatMul(output, a, b);

        // Materializes the device-side output to host (AsSpan takes the same EnsureCpuData path as DataPointer).
        backend.OffloadActivation(output);
        Assert.Equal(expected, output.AsSpan<ushort>().ToArray());
    }

    private static ushort Bf16Bits(float value) => (ushort)(BitConverter.SingleToUInt32Bits(value) >> 16);

    private static float FromBf16(ushort bits) => BitConverter.UInt32BitsToSingle((uint)bits << 16);

    private static string PtxDir()
    {
        for (DirectoryInfo? dir = new(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "src", "HartsyInference.Cuda", "Ptx");
            if (Directory.Exists(candidate)) return candidate;
        }
        throw new DirectoryNotFoundException("src/HartsyInference.Cuda/Ptx not found above the test output directory.");
    }
}
