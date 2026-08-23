using HartsyInference.Engine.Requests;

namespace HartsyInference.API.Endpoints;

/// <summary>Persistence controls shared by every route that produces a file. Generated media is written to disk by default — the CLI has always done this and an HTTP client should not have to re-encode a base64 payload just to keep what it asked for.</summary>
public abstract class NativeArtifactRequest
{
    /// <summary>Set false to return the artifact in the response only, writing nothing to disk.</summary>
    public bool? Save { get; set; }

    /// <summary>Directory to write into; unset uses the server's output root (<c>HARTSYINFERENCE_OUTPUT</c>, else <c>&lt;repo&gt;/Output</c>). Note this lets a client direct writes anywhere the server process can reach.</summary>
    public string? OutputDir { get; set; }
}

/// <summary>Envelope for every native generation route: model selection alongside the untouched native request record. <c>model</c>/<c>modelPath</c> live in the body (not the query string) so this shape stays identical across every modality and lines up with where the OpenAI-compat layer already carries <c>model</c>.</summary>
public sealed class NativeImageRequest : NativeArtifactRequest
{
    /// <summary>Catalog id, local path, or HuggingFace repo id.</summary>
    public required string Model { get; set; }

    /// <summary>Explicit checkpoint path override; wins over catalog/HF resolution. Must be an absolute path — a relative one resolves against the server process's working directory (which depends on how it was launched, e.g. <c>dotnet run --project X</c> uses X's directory, not the repo root), not anything an HTTP client can know. Rarely needed: a plain catalog id in <c>model</c> resolves correctly on its own.</summary>
    public string? ModelPath { get; set; }

    /// <summary>The native image request, unmodified.</summary>
    public required ImageRequest Request { get; set; }

    /// <summary>Base64 PNG (a <c>data:</c> URI is accepted) to use as the img2img init image. A convenience over setting <c>request.img2img.initImage</c> directly, which carries raw RGB24 bytes an HTTP client rarely has; when both are supplied this one wins.</summary>
    public string? InitImageBase64 { get; set; }

    /// <summary>Denoise strength in 0..1 for <see cref="InitImageBase64"/>; unset keeps the request's own value.</summary>
    public double? Creativity { get; set; }

    /// <summary>Base64 PNG inpaint mask (white = regenerate, black = preserve). Requires an init image.</summary>
    public string? MaskBase64 { get; set; }

    /// <summary>Grows the mask by this many pixels before use.</summary>
    public int? MaskGrow { get; set; }

    /// <summary>Blurs the mask by this many pixels for a softer transition.</summary>
    public int? MaskBlur { get; set; }

    /// <summary>"Inpaint only masked": crop to the mask's bounding box grown by this many pixels, generate there, and composite back, so the masked region gets the model's full resolution. 0 disables it.</summary>
    public int? MaskShrinkGrow { get; set; }
}

/// <summary>Envelope for <c>/v1/native/text*</c>.</summary>
public sealed class NativeTextRequest
{
    /// <summary>Catalog id, local path, or HuggingFace repo id.</summary>
    public required string Model { get; set; }

    /// <summary>Explicit checkpoint path override; wins over catalog/HF resolution. Must be an absolute path — a relative one resolves against the server process's working directory (which depends on how it was launched, e.g. <c>dotnet run --project X</c> uses X's directory, not the repo root), not anything an HTTP client can know. Rarely needed: a plain catalog id in <c>model</c> resolves correctly on its own.</summary>
    public string? ModelPath { get; set; }

    /// <summary>The native text request, unmodified.</summary>
    public required TextRequest Request { get; set; }
}

/// <summary>Request body for <c>GET /v1/native/text/count-tokens</c>.</summary>
public sealed class CountTokensRequest
{
    /// <summary>Catalog id, local path, or HuggingFace repo id.</summary>
    public required string Model { get; set; }

    /// <summary>Explicit checkpoint path override; wins over catalog/HF resolution. Must be an absolute path — a relative one resolves against the server process's working directory (which depends on how it was launched, e.g. <c>dotnet run --project X</c> uses X's directory, not the repo root), not anything an HTTP client can know. Rarely needed: a plain catalog id in <c>model</c> resolves correctly on its own.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Text to tokenize.</summary>
    public required string Text { get; set; }
}
