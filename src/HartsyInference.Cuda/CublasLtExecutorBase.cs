using System.Runtime.CompilerServices;

namespace HartsyInference.Cuda;

/// <summary>What every cuBLASLt executor owns — one handle, one persistent workspace, and the teardown that returns both — plus the descriptor and layout calls they all make. A derived class decides support before construction; when it is false nothing is allocated, so callers can construct unconditionally and ask <see cref="IsSupported"/>.</summary>
/// <remarks>Layouts follow the engine's row-major convention <c>D[M, N] = B[M, K] · A[N, K]ᵀ</c>: the weight is operand A under OP_T, the activation operand B under OP_N, and cuBLASLt sees column-major (K×N), (K×M), (N×M). A handle that fails to create leaves the executor unsupported with a warning rather than throwing, so a broken cuBLASLt sends its callers down their fallback path instead of taking every GEMM with it.</remarks>
public abstract unsafe class CublasLtExecutorBase : IDisposable
{
    private nint _ltHandle;
    private ulong _workspace;
    private readonly nuint _workspaceBytes;
    private int _disposed;

    /// <summary>Whether this executor can run on the current device. False means the constructor allocated nothing.</summary>
    public bool IsSupported { get; }

    protected nint LtHandle => _ltHandle;
    protected nuint WorkspaceBytes => _workspaceBytes;

    protected CublasLtExecutorBase(bool supported)
    {
        if (!supported) return;
        if (CublasLtApi.cublasLtCreate(out _ltHandle) != 0)
        {
            _ltHandle = 0;
            HartsyInference.Core.Logging.Logs.Warning(
                $"[Cuda] cublasLtCreate failed; {GetType().Name} is unavailable and its callers take their fallback path.");
            return;
        }
        IsSupported = true;
        _workspaceBytes = (nuint)CublasLtApi.DefaultWorkspaceBytes;
        _workspace = CudaMemory.AllocatePersistent(_workspaceBytes);
    }

    /// <summary>A matmul descriptor with OP_T on operand A (the weight) and OP_N on B (the activation).</summary>
    protected static nint CreateTnMatmulDesc(int computeType, int scaleType)
    {
        CublasLtApi.cublasLtMatmulDescCreate(out nint desc, computeType, scaleType).ThrowOnCublasError();
        SetDescAttribute(desc, CublasLtApi.CUBLASLT_MATMUL_DESC_TRANSA, CublasApi.CUBLAS_OP_T);
        SetDescAttribute(desc, CublasLtApi.CUBLASLT_MATMUL_DESC_TRANSB, CublasApi.CUBLAS_OP_N);
        return desc;
    }

    protected static void SetDescAttribute<T>(nint desc, int attribute, T value) where T : unmanaged
    {
        CublasLtApi.cublasLtMatmulDescSetAttribute(desc, attribute, &value, (nuint)sizeof(T)).ThrowOnCublasError();
    }

    /// <summary>The three layouts of <c>D[M, N] = B[M, K] · A[N, K]ᵀ</c> in cuBLASLt's column-major terms. Each handle is written as soon as it exists, so a throw part-way leaves the caller holding what was created.</summary>
    protected static void CreateTnLayouts(int aType, int bType, int dType, int m, int n, int k,
        out nint layoutA, out nint layoutB, out nint layoutD)
    {
        layoutA = layoutB = layoutD = 0;
        CublasLtApi.cublasLtMatrixLayoutCreate(out layoutA, aType, (ulong)k, (ulong)n, k).ThrowOnCublasError();
        CublasLtApi.cublasLtMatrixLayoutCreate(out layoutB, bType, (ulong)k, (ulong)m, k).ThrowOnCublasError();
        CublasLtApi.cublasLtMatrixLayoutCreate(out layoutD, dType, (ulong)n, (ulong)m, n).ThrowOnCublasError();
    }

    /// <summary>In-place <c>D = alpha·A·B + beta·D</c> on this executor's handle and workspace; returns the cuBLAS status.</summary>
    protected int Matmul(nint desc, void* alpha, ulong a, nint layoutA, ulong b, nint layoutB, void* beta,
        ulong d, nint layoutD, nint algo, nint stream)
        => CublasLtApi.cublasLtMatmul(_ltHandle, desc, alpha, a, layoutA, b, layoutB, beta, d, layoutD, d, layoutD,
            algo, (nint)_workspace, _workspaceBytes, stream);

    protected static void DestroyLayouts(nint layoutA, nint layoutB, nint layoutD)
    {
        if (layoutA != 0) CublasLtApi.cublasLtMatrixLayoutDestroy(layoutA);
        if (layoutB != 0) CublasLtApi.cublasLtMatrixLayoutDestroy(layoutB);
        if (layoutD != 0) CublasLtApi.cublasLtMatrixLayoutDestroy(layoutD);
    }

    protected static void DestroyDesc(nint desc)
    {
        if (desc != 0) CublasLtApi.cublasLtMatmulDescDestroy(desc);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    protected void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0)
            throw new ObjectDisposedException(GetType().Name);
    }

    public void Dispose()
    {
        Release();
        GC.SuppressFinalize(this);
    }

    ~CublasLtExecutorBase() => Release();

    private void Release()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        if (_workspace != 0)
        {
            CudaMemory.Free(_workspace);
            _workspace = 0;
        }
        if (_ltHandle != 0)
        {
            CublasLtApi.cublasLtDestroy(_ltHandle);
            _ltHandle = 0;
        }
    }
}
