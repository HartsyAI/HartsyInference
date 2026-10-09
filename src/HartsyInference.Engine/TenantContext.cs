namespace HartsyInference.Engine;

/// <summary>The tenant a request belongs to, for the server's per-tenant state (today the prefix cache). The API sets <see cref="Current"/> from the caller's API key identity
/// before a request runs, so the tenant comes from the server's own identity record and never from a field the client sends.</summary>
public static class TenantContext
{
    /// <summary>The tenant of a request that carries no identity: the server runs with auth off, or a library caller did not name one.</summary>
    public const string Local = "local";

    private static readonly AsyncLocal<string?> s_current = new();

    /// <summary>The tenant of the request now running on this async flow, or null when none was set.</summary>
    public static string? Current
    {
        get => s_current.Value;
        set => s_current.Value = value;
    }

    /// <summary>The tenant a request runs under: the one it names (library callers only; the API never sets it from input), else the current identity, else <see cref="Local"/>.
    /// A blank name is no name, so it cannot become a tenant of its own.</summary>
    public static string Resolve(string? requested)
    {
        if (!string.IsNullOrWhiteSpace(requested)) return requested;
        string? current = s_current.Value;
        return string.IsNullOrWhiteSpace(current) ? Local : current;
    }
}
