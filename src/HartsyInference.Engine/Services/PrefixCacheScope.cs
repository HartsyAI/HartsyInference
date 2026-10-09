namespace HartsyInference.Engine.Services;

/// <summary>The key a retained prefix is stored under: the tenant, the model that holds it, the image it was built with (none for text), and the caller's prefix key. Each
/// part before the last is length-prefixed, so two different scopes can never produce the same key, whatever characters the names contain.</summary>
public static class PrefixCacheScope
{
    /// <summary>The store key for <paramref name="prefixKey"/> under <paramref name="tenant"/>'s copy of <paramref name="model"/>.</summary>
    public static string Key(string tenant, string model, string prefixKey, string? imageHash = null) =>
        $"{Part(tenant)}|{Part(model)}|{Part(imageHash ?? "")}|{prefixKey}";

    private static string Part(string text) => $"{text.Length}:{text}";
}
