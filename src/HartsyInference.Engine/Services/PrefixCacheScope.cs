using System.Diagnostics.CodeAnalysis;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>The key a retained prefix is stored under: the tenant, the model that holds it, the image it was built with (none for text), and the caller's prefix key. Each
/// part before the last is length-prefixed, so two different scopes can never produce the same key, whatever characters the names contain.</summary>
public static class PrefixCacheScope
{
    /// <summary>The store key for <paramref name="prefixKey"/> under <paramref name="tenant"/>'s copy of <paramref name="model"/>.</summary>
    public static string Key(string tenant, string model, string prefixKey, string? imageHash = null) =>
        $"{Part(tenant)}|{Part(model)}|{Part(imageHash ?? "")}|{prefixKey}";

    /// <summary>The store key for <paramref name="request"/>'s prefix on <paramref name="slot"/>, or false when the request takes no prefix-cache hit there: it names no
    /// prefix key, or the slot holds an SSM model or the V4.1 host, which keep their own sequence state and take no prefix hits in this version. Checkout and check-in both
    /// use this one answer.</summary>
    internal static bool TryKey(TextDeviceSlot slot, TextRequest request, [NotNullWhen(true)] out string? key)
    {
        if (request.PrefixCacheKey is not { Length: > 0 } prefixKey || slot.SsmPipeline is not null || slot.DeepSeekV41 is not null)
        {
            key = null;
            return false;
        }
        key = Key(request.TenantId ?? TenantContext.Local, slot.LoadedPath ?? "", prefixKey);
        return true;
    }

    private static string Part(string text) => $"{text.Length}:{text}";
}
