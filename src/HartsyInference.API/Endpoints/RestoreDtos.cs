using HartsyInference.Engine.Requests;

namespace HartsyInference.API.Endpoints;

/// <summary>Envelope for <c>/v1/native/restore</c> and <c>/v1/native/restore/stream</c>. The inner request's
/// <see cref="RestoreRequest.Video"/> carries container bytes base64-encoded by JSON — large uploads are the
/// caller's bandwidth to spend; every native route is JSON-only (no multipart anywhere in this API).</summary>
public sealed class NativeRestoreRequest
{
    /// <summary>Catalog id (seedvr2-3b, seedvr2-7b), local path, or HuggingFace repo id.</summary>
    public required string Model { get; set; }

    /// <summary>Explicit DiT safetensors path override; VAE + embeddings resolve as siblings. Must be an
    /// absolute path (a relative one resolves against the server's working directory).</summary>
    public string? ModelPath { get; set; }

    /// <summary>The native restore request, unmodified.</summary>
    public required RestoreRequest Request { get; set; }
}
