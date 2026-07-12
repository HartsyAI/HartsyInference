namespace HartsyInference.Cuda;

/// <summary>A cuDNN backend-API call failed, carrying the raw <c>cudnnStatus_t</c> so callers can classify
/// the failure programmatically instead of parsing the formatted message. Internal — never crosses
/// <see cref="CudaBackend"/>'s public boundary; call sites catch it and decide whether to retry or fall back.</summary>
internal sealed class CudnnStatusException : Exception
{
    /// <summary>The raw <c>cudnnStatus_t</c> value.</summary>
    public int Status { get; }

    public CudnnStatusException(int status, string what)
        : base($"cuDNN {what} failed: {CudnnApi.ErrorString(status)}")
    {
        Status = status;
    }

    /// <summary>True for structural/permanent failures that will never succeed no matter how many times
    /// they're retried (bad parameters, an unsupported shape/architecture) — false for everything else,
    /// including resource-pressure-shaped failures (allocation/internal-error codes) and any future,
    /// currently-unrecognized status category. cuDNN 9's backend API groups status codes by category —
    /// <c>#define CUDNN_STATUS_CATEGORY(code) ((code) / 1000 * 1000)</c> (cudnn_graph.h) — with
    /// <c>BAD_PARAM=2000</c> and <c>NOT_SUPPORTED=3000</c> (which subsumes e.g. an architecture mismatch)
    /// being the only categories that mean "this will never work," versus <c>INTERNAL_ERROR=4000</c>
    /// (the category the observed <c>HOST_ALLOCATION_FAILED=4003</c> falls under) and
    /// <c>EXECUTION_FAILED=5000</c>, which are resource/runtime conditions that can clear up. Defaulting
    /// unrecognized categories to "not permanent" is the deliberately safer direction: worst case a
    /// genuinely-broken box burns a few bounded, logged retries before settling into a long backoff —
    /// it never permanently gives up on something that might actually recover.</summary>
    public bool IsPermanent => Status / 1000 * 1000 is 2000 or 3000;
}
