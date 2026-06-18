namespace HartsyInference.Cuda;

/// <summary>Launcher for the hand-written tensor-core HGEMM (<c>hgemm_mma_sm80.ptx</c>), computing <c>C[M,N] = alpha · A[M,K] · B^T</c> with B stored row-major <c>[N,K]</c> (the Linear weight). One warp computes one 16x8 output tile via <c>mma.sync.m16n8k16</c> with FP32 accumulation and F16 in/out.
///
/// <para><b>Alignment.</b> The kernel omits per-element bounds checks for speed, so it only handles <c>M%16==0, N%8==0, K%16==0</c>. Use <see cref="IsAligned"/> to decide; unaligned shapes must fall back to cuBLAS.</para>
///
/// <para><b>Validation-pending.</b> The PTX fragment layout has not been verified on GPU. Gate any production use behind a benchmark + a numerical diff against cuBLAS (see <c>TensorCoreGemmTests</c>); the engine keeps this off by default (<see cref="CudaBackend.EnableTensorCoreGemm"/>).</para></summary>
public sealed class TensorCoreGemm : IDisposable
{
    private readonly string _ptxDir;
    private CudaModule? _module;
    private nint _function;
    private int _disposed;

    /// <summary>Requires SM 8.0+ (mma.sync.m16n8k16). True on Ampere/Ada/Hopper, false on Turing and below.</summary>
    public bool IsSupported { get; }

    /// <summary>Creates the launcher. PTX is loaded lazily on first <see cref="Run"/>.</summary>
    public TensorCoreGemm(string ptxDir, int smMajor)
    {
        _ptxDir = ptxDir ?? throw new ArgumentNullException(nameof(ptxDir));
        IsSupported = smMajor >= 8;
    }

    /// <summary>Whether the kernel can handle these dimensions without bounds checks.</summary>
    public static bool IsAligned(int m, int n, int k) => (m % 16) == 0 && (n % 8) == 0 && (k % 16) == 0;

    /// <summary>Launches the tensor-core GEMM. Caller must have verified <see cref="IsSupported"/> and <see cref="IsAligned"/>. All pointers are device F16 (A, B, C); C is overwritten (no accumulation into existing C).</summary>
    public unsafe void Run(ulong a, ulong b, ulong c, int m, int n, int k, float alpha, nint stream)
    {
        if (!IsSupported)
            throw new InvalidOperationException("TensorCoreGemm requires SM 8.0+. Check IsSupported first.");
        if (!IsAligned(m, n, k))
            throw new ArgumentException($"TensorCoreGemm requires M%16==0, N%8==0, K%16==0; got M={m}, N={n}, K={k}.");
        ThrowIfDisposed();
        EnsureLoaded();

        ulong aArg = a, bArg = b, cArg = c;
        uint mArg = (uint)m, nArg = (uint)n, kArg = (uint)k;
        float alphaArg = alpha;
        void** args = stackalloc void*[7];
        args[0] = &aArg;
        args[1] = &bArg;
        args[2] = &cArg;
        args[3] = &mArg;
        args[4] = &nArg;
        args[5] = &kArg;
        args[6] = &alphaArg;

        uint gridX = (uint)(n / 8);   // one warp per 8 output columns
        uint gridY = (uint)(m / 16);  // one warp per 16 output rows
        CudaDriverApi.cuLaunchKernel(
            _function, gridX, gridY, 1,
            32, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    private void EnsureLoaded()
    {
        if (_module is not null) return;
        _module = CudaModule.LoadFromFile(Path.Combine(_ptxDir, "hgemm_mma_sm80.ptx"));
        _function = _module.GetFunction("hgemm_mma_sm80");
    }

    private void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0)
            throw new ObjectDisposedException(nameof(TensorCoreGemm));
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _module?.Dispose();
        _module = null;
        GC.SuppressFinalize(this);
    }
}
