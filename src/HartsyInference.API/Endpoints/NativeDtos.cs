using HartsyInference.Engine.Requests;

namespace HartsyInference.API.Endpoints;

/// <summary>Envelope for every native generation route: model selection alongside the untouched native request
/// record. <c>model</c>/<c>modelPath</c> live in the body (not the query string) so this shape stays identical
/// across every modality and lines up with where the OpenAI-compat layer already carries <c>model</c>.</summary>
public sealed class NativeImageRequest
{
    /// <summary>Catalog id, local path, or HuggingFace repo id.</summary>
    public required string Model { get; set; }

    /// <summary>Explicit checkpoint path override; wins over catalog/HF resolution.</summary>
    public string? ModelPath { get; set; }

    /// <summary>The native image request, unmodified.</summary>
    public required ImageRequest Request { get; set; }
}

/// <summary>Envelope for <c>/v1/native/text*</c>.</summary>
public sealed class NativeTextRequest
{
    /// <summary>Catalog id, local path, or HuggingFace repo id.</summary>
    public required string Model { get; set; }

    /// <summary>Explicit checkpoint path override; wins over catalog/HF resolution.</summary>
    public string? ModelPath { get; set; }

    /// <summary>The native text request, unmodified.</summary>
    public required TextRequest Request { get; set; }
}

/// <summary>Request body for <c>GET /v1/native/text/count-tokens</c>.</summary>
public sealed class CountTokensRequest
{
    /// <summary>Catalog id, local path, or HuggingFace repo id.</summary>
    public required string Model { get; set; }

    /// <summary>Explicit checkpoint path override; wins over catalog/HF resolution.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Text to tokenize.</summary>
    public required string Text { get; set; }
}
