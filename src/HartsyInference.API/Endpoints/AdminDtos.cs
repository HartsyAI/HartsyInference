namespace HartsyInference.API.Endpoints;

/// <summary>Request body for <c>POST /admin/models/pull</c>.</summary>
public sealed class PullModelRequest
{
    /// <summary>Catalog id to download the preset assets for.</summary>
    public string Model { get; set; } = "";
}

/// <summary>Request body for <c>POST /admin/backend</c>.</summary>
public sealed class SetBackendRequest
{
    /// <summary>Backend selector: <c>auto</c>/<c>cpu</c>/<c>cuda</c>/<c>vulkan</c>.</summary>
    public string Backend { get; set; } = "";
}
