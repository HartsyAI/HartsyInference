namespace HartsyInference.Cuda;

/// <summary>Wraps a CUDA stream handle with deterministic disposal.</summary>
public sealed class CudaStream : IDisposable
{
    private nint _handle;

    /// <summary>The underlying CUDA stream handle.</summary>
    public nint Handle
    {
        get
        {
            nint h = _handle;
            if (h == 0)
                throw new ObjectDisposedException(nameof(CudaStream));
            return h;
        }
    }

    /// <summary>Creates a new CUDA stream.</summary>
    /// <param name="nonBlocking">If true, the stream does not synchronize with the default stream.</param>
    public CudaStream(bool nonBlocking = true)
    {
        uint flags = nonBlocking ? CudaDriverApi.CU_STREAM_NON_BLOCKING : CudaDriverApi.CU_STREAM_DEFAULT;
        CudaDriverApi.cuStreamCreate(out _handle, flags).ThrowOnError();
    }

    /// <summary>Wraps an existing stream handle. The caller retains ownership unless transferOwnership is true.</summary>
    internal CudaStream(nint existingHandle)
    {
        _handle = existingHandle;
    }

    /// <summary>Blocks until all work in this stream is complete.</summary>
    public void Synchronize()
    {
        CudaDriverApi.cuStreamSynchronize(Handle).ThrowOnError();
    }

    /// <summary>Returns true if all work in this stream is complete, false if still executing.</summary>
    public bool IsComplete()
    {
        int result = CudaDriverApi.cuStreamQuery(Handle);
        if (result == 0) return true;
        if (result == 600) return false; // CUDA_ERROR_NOT_READY
        result.ThrowOnError();
        return false;
    }

    public void Dispose()
    {
        nint h = Interlocked.Exchange(ref _handle, 0);
        if (h != 0)
        {
            CudaDriverApi.cuStreamDestroy(h);
        }
        GC.SuppressFinalize(this);
    }

    ~CudaStream()
    {
        nint h = Interlocked.Exchange(ref _handle, 0);
        if (h != 0)
        {
            CudaDriverApi.cuStreamDestroy(h);
        }
    }
}
