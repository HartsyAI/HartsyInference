using HartsyInference.Core.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Which operand and output types <c>cublasGemmGroupedBatchedEx</c> accepts on this machine's cuBLAS. The MoE prefill path uses the first
/// combination that works; this records the facts instead of leaving them to a log line.</summary>
[Collection("CudaSerial")]
public sealed unsafe class CublasGroupedGemmProbeTests
{
    private readonly ITestOutputHelper _output;
    public CublasGroupedGemmProbeTests(ITestOutputHelper output) => _output = output;

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void Probe_GroupedGemmTypeCombinations()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir)) ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        using CudaBackend cuda = new(0, ptxDir);
        nint handle = default;
        CublasApi.cublasCreate(out handle);

        const int groups = 4, m = 64, n = 96, k = 128;
        // Big zeroed buffers; the contents do not matter for support.
        ulong a = CudaMemory.Allocate(1 << 22), b = CudaMemory.Allocate(1 << 22), c = CudaMemory.Allocate(1 << 22);
        ulong ptrs = CudaMemory.Allocate(3 * groups * sizeof(ulong));
        ulong[] host = new ulong[3 * groups];
        for (int g = 0; g < groups; g++)
        {
            host[g] = a + (ulong)(g * 65536);
            host[groups + g] = b + (ulong)(g * 65536);
            host[2 * groups + g] = c + (ulong)(g * 65536);
        }
        fixed (ulong* hp = host) CudaMemory.CopyHostToDevice(ptrs, hp, (nuint)(host.Length * sizeof(ulong)));

        (string name, int ab, int cType, int compute, int opA)[] combos =
        [
            ("BF16 x BF16 -> F32, A^T", CublasApi.CUDA_R_16BF, CublasApi.CUDA_R_32F, CublasApi.CUBLAS_COMPUTE_32F, CublasApi.CUBLAS_OP_T),
            ("BF16 x BF16 -> BF16, A^T", CublasApi.CUDA_R_16BF, CublasApi.CUDA_R_16BF, CublasApi.CUBLAS_COMPUTE_32F, CublasApi.CUBLAS_OP_T),
            ("F16 x F16 -> F16, A^T", CublasApi.CUDA_R_16F, CublasApi.CUDA_R_16F, CublasApi.CUBLAS_COMPUTE_32F, CublasApi.CUBLAS_OP_T),
            ("F16 x F16 -> F32, A^T", CublasApi.CUDA_R_16F, CublasApi.CUDA_R_32F, CublasApi.CUBLAS_COMPUTE_32F, CublasApi.CUBLAS_OP_T),
            ("BF16 x BF16 -> BF16, A", CublasApi.CUDA_R_16BF, CublasApi.CUDA_R_16BF, CublasApi.CUBLAS_COMPUTE_32F, CublasApi.CUBLAS_OP_N),
            ("F32 x F32 -> F32, A^T", CublasApi.CUDA_R_32F, CublasApi.CUDA_R_32F, CublasApi.CUBLAS_COMPUTE_32F, CublasApi.CUBLAS_OP_T),
        ];
        foreach ((string name, int ab, int cType, int compute, int opA) in combos)
        {
            int[] transa = new int[groups], transb = new int[groups], mm = new int[groups], nn = new int[groups], kk = new int[groups];
            int[] lda = new int[groups], ldb = new int[groups], ldc = new int[groups], size = new int[groups];
            float[] alpha = new float[groups], beta = new float[groups];
            for (int g = 0; g < groups; g++)
            {
                transa[g] = opA; transb[g] = CublasApi.CUBLAS_OP_N; mm[g] = n; nn[g] = m; kk[g] = k;
                lda[g] = opA == CublasApi.CUBLAS_OP_T ? k : n; ldb[g] = k; ldc[g] = n; size[g] = 1; alpha[g] = 1f;
            }
            int status;
            fixed (int* pTa = transa, pTb = transb, pM = mm, pN = nn, pK = kk, pLa = lda, pLb = ldb, pLc = ldc, pSz = size)
            fixed (float* pAl = alpha, pBe = beta)
                status = CublasApi.cublasGemmGroupedBatchedEx(handle, pTa, pTb, pM, pN, pK, pAl, ptrs, ab, pLa, ptrs + (ulong)(groups * 8), ab, pLb,
                    pBe, ptrs + (ulong)(2 * groups * 8), cType, pLc, groups, pSz, compute);
            _output.WriteLine($"{name}: status {status}");
        }
        CublasApi.cublasDestroy(handle);
    }
}
