namespace HartsyInference.ThreeD.Document;

/// <summary>An encoded texture image carried as it came (PNG or JPEG bytes), never decoded or re-encoded.</summary>
/// <param name="MimeType">The media type, <c>image/png</c> or <c>image/jpeg</c>.</param>
/// <param name="Data">The encoded file bytes.</param>
public sealed record MeshImage(string MimeType, byte[] Data)
{
    /// <summary>The file extension for <see cref="MimeType"/>, without the dot.</summary>
    public string Extension => MimeType == "image/jpeg" ? "jpg" : "png";
}
