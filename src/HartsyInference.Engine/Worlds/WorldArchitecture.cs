namespace HartsyInference.Engine.Worlds;

/// <summary>Which loader and fixed architecture config a world checkpoint set is built with.</summary>
public enum WorldArchitecture
{
    /// <summary>Oasis-500m: an axial-attention DiT with a ViT-VAE. Also the loader for catalog ids the service does not know.</summary>
    Oasis,

    /// <summary>DIAMOND: a single EDM denoiser checkpoint.</summary>
    Diamond,

    /// <summary>Hunyuan-GameCraft: a DiT with a 3D VAE, a Llava-Llama-3-8B text encoder and a CLIP-L text encoder.</summary>
    HunyuanGameCraft,
}
