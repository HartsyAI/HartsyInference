namespace HartsyInference.Engine;

/// <summary>The safetensors repack of a pickle asset, as a weights mirror hosts it. The engine asks the mirror for it first and falls back to the asset's upstream file when the mirror is not configured, does not answer or serves bytes that miss <see cref="Sha256"/>.
/// A repack holds the pickle's tensors under the same names, and sits beside the pickle's path under the same name with the extension <c>.safetensors</c>.</summary>
public sealed record AssetRepack
{
    /// <summary>The file's path under the mirror's base URL, with <c>/</c> separators, e.g. <c>hunyuan3d-paint21/unet/diffusion_pytorch_model.safetensors</c>.</summary>
    public required string MirrorPath { get; init; }

    /// <summary>SHA-256 hex hash of the repack, checked before it is moved into place.</summary>
    public required string Sha256 { get; init; }
}
