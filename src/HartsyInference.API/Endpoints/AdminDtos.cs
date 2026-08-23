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

/// <summary>Optional request body for <c>POST /admin/memory/free</c>. Body may be omitted entirely for the
/// default soft free.</summary>
public sealed class FreeMemoryRequest
{
    /// <summary>When true, fully disposes and recreates the backend instead of the default soft evict+trim. Soft free (<c>IInferenceEngine.FreeMemory()</c>) can only reclaim GPU memory the Engine has a live reference to — a pipeline construction that fails partway (e.g. an OOM mid-load) never gets registered anywhere, so whatever it already allocated before failing is untracked and unreachable by a soft free. Tearing down the whole CUDA context is the only way to reclaim that. Heavier: every currently-loaded model, not just a broken one, needs to reload on next use.</summary>
    public bool Hard { get; set; }
}
