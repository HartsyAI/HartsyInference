namespace HartsyInference.API.Endpoints;

/// <summary>Request body for <c>POST /admin/models/pull</c>.</summary>
public sealed class PullModelRequest
{
    /// <summary>Catalog id to download the preset assets for.</summary>
    public string Model { get; set; } = "";
}

/// <summary>Request body for <c>POST /admin/models/quantize</c>. Precision is a request field rather than a
/// server setting on purpose: which precision a file is written at describes that file, not this machine.</summary>
public sealed class QuantizeModelRequest
{
    /// <summary>Checkpoint to read; any container the engine can open, including a GGUF being made smaller.</summary>
    public string ModelPath { get; set; } = "";

    /// <summary>File to write.</summary>
    public string Out { get; set; } = "";

    /// <summary>Output format: <c>gguf</c>, <c>fp8-scaled</c> or <c>int8-convrot</c>.</summary>
    public string Format { get; set; } = "gguf";

    /// <summary>GGUF precision preset; ignored by the other formats.</summary>
    public string Quant { get; set; } = "Q8_0";

    /// <summary>Value for the output's <c>general.architecture</c>, which is how a reader picks a key mapper.</summary>
    public string? Architecture { get; set; }

    /// <summary>Whether an existing output may be replaced.</summary>
    public bool Overwrite { get; set; }
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
