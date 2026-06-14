using System.Runtime.InteropServices;
using HartsyInference.Core.Exceptions;

namespace HartsyInference.Cuda;

/// <summary>Exception thrown when a CUDA Driver API or cuBLAS call returns a non-zero error code.</summary>
public sealed class CudaException : HartsyInferenceException
{
    /// <summary>The raw CUDA error code returned by the driver.</summary>
    public int ErrorCode { get; }

    public CudaException(int errorCode, string message)
        : base($"CUDA error {errorCode}: {message}")
    {
        ErrorCode = errorCode;
    }

    public CudaException(int errorCode, string message, Exception innerException)
        : base($"CUDA error {errorCode}: {message}", innerException)
    {
        ErrorCode = errorCode;
    }
}

/// <summary>Extension methods for checking CUDA and cuBLAS return codes.</summary>
public static class CudaErrorChecking
{
    /// <summary>Throws CudaException if the CUDA Driver API return code is non-zero.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ThrowOnError(this int cudaResult)
    {
        if (cudaResult != 0)
        {
            ThrowCudaError(cudaResult);
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void ThrowCudaError(int errorCode)
    {
        string errorName = "UNKNOWN";
        string errorString = "Unknown CUDA error";

        try
        {
            int nameResult = CudaDriverApi.cuGetErrorName(errorCode, out nint pName);
            if (nameResult == 0 && pName != 0)
            {
                errorName = Marshal.PtrToStringAnsi(pName) ?? "UNKNOWN";
            }

            int strResult = CudaDriverApi.cuGetErrorString(errorCode, out nint pStr);
            if (strResult == 0 && pStr != 0)
            {
                errorString = Marshal.PtrToStringAnsi(pStr) ?? "Unknown CUDA error";
            }
        }
        catch
        {
            // If we can't look up the error, use the code directly
        }

        throw new CudaException(errorCode, $"{errorName} — {errorString}");
    }

    /// <summary>Throws CudaException if the cuBLAS return code is non-zero.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ThrowOnCublasError(this int cublasStatus)
    {
        if (cublasStatus != 0)
        {
            ThrowCublasError(cublasStatus);
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void ThrowCublasError(int status)
    {
        string message = status switch
        {
            1 => "CUBLAS_STATUS_NOT_INITIALIZED",
            3 => "CUBLAS_STATUS_ALLOC_FAILED",
            7 => "CUBLAS_STATUS_INVALID_VALUE",
            8 => "CUBLAS_STATUS_ARCH_MISMATCH",
            11 => "CUBLAS_STATUS_MAPPING_ERROR",
            13 => "CUBLAS_STATUS_EXECUTION_FAILED",
            14 => "CUBLAS_STATUS_INTERNAL_ERROR",
            15 => "CUBLAS_STATUS_NOT_SUPPORTED",
            16 => "CUBLAS_STATUS_LICENSE_ERROR",
            _ => $"CUBLAS_STATUS_UNKNOWN ({status})",
        };

        throw new CudaException(status, message);
    }
}
