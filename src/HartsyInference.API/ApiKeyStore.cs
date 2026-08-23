namespace HartsyInference.API;

/// <summary>Caller identity a presented API key resolves to — carried on <c>HttpContext.Items</c> for the rate limiter's partition key, <see cref="UsageTracker"/>'s counter key, and the request's <c>Activity</c> tag.</summary>
public sealed record ApiKeyIdentity(string Name, int RateLimitPerMinute);

/// <summary>In-memory lookup from a presented API key secret to its <see cref="ApiKeyIdentity"/>. Built once at startup from <see cref="HartsyInferenceServerOptions.ApiKeys"/> (with the legacy single <see cref="HartsyInferenceServerOptions.ApiKey"/> folded in as a <c>"default"</c> entry) — no persistence, matches every other piece of server state (<c>WorldSessionRegistry</c>, <c>ModelCacheStore</c>) in this project's single-process, no-database design.</summary>
public sealed class ApiKeyStore
{
    private readonly Dictionary<string, ApiKeyIdentity> _byKey;

    public ApiKeyStore(HartsyInferenceServerOptions options)
    {
        _byKey = new Dictionary<string, ApiKeyIdentity>(StringComparer.Ordinal);

        if (!string.IsNullOrEmpty(options.ApiKey))
            _byKey[options.ApiKey] = new ApiKeyIdentity("default", options.DefaultRateLimitPerMinute);

        foreach (ApiKeyOptions entry in options.ApiKeys)
        {
            if (string.IsNullOrEmpty(entry.Key))
                continue;
            string name = string.IsNullOrWhiteSpace(entry.Name) ? "default" : entry.Name;
            _byKey[entry.Key] = new ApiKeyIdentity(name, entry.RateLimitPerMinute ?? options.DefaultRateLimitPerMinute);
        }
    }

    /// <summary>True when at least one key is configured — mirrors the old <c>!string.IsNullOrEmpty(ApiKey)</c> gate that decided whether auth middleware runs at all.</summary>
    public bool HasAnyKeys => _byKey.Count > 0;

    public bool TryResolve(string presentedKey, out ApiKeyIdentity identity) =>
        _byKey.TryGetValue(presentedKey, out identity!);
}
