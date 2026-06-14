using HartsyInference.Core.Exceptions;

namespace HartsyInference.Vulkan;

/// <summary>Exception thrown when a Vulkan API call returns a negative VkResult.</summary>
public sealed class VulkanException : HartsyInferenceException
{
    /// <summary>The raw Vulkan result code returned by the driver.</summary>
    public int ErrorCode { get; }

    public VulkanException(int errorCode, string message)
        : base($"Vulkan error {errorCode} ({(VkResult)errorCode}): {message}")
    {
        ErrorCode = errorCode;
    }

    public VulkanException(int errorCode, string message, Exception inner)
        : base($"Vulkan error {errorCode} ({(VkResult)errorCode}): {message}", inner)
    {
        ErrorCode = errorCode;
    }
}

/// <summary>Extension methods for checking VkResult return codes. Negative = error.</summary>
public static class VulkanErrorChecking
{
    /// <summary>Throws VulkanException if the result is a negative error code.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ThrowOnError(this VkResult result, string operation)
    {
        if ((int)result < 0)
            ThrowVk((int)result, operation);
    }

    /// <summary>Throws VulkanException if the result is a negative error code.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ThrowOnError(this int result, string operation)
    {
        if (result < 0)
            ThrowVk(result, operation);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void ThrowVk(int code, string op)
    {
        throw new VulkanException(code, op);
    }
}
