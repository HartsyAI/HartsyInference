namespace HartsyInference.Engine.Worlds;

/// <summary>Which loader and fixed architecture config a world checkpoint set is built with.</summary>
public enum WorldArchitecture
{
    /// <summary>Oasis-500m: an axial-attention DiT with a ViT-VAE.</summary>
    Oasis,

    /// <summary>DIAMOND: a single EDM denoiser checkpoint.</summary>
    Diamond,

    /// <summary>Hunyuan-GameCraft: a DiT with a 3D VAE, a Llava-Llama-3-8B text encoder and a CLIP-L text encoder.</summary>
    HunyuanGameCraft,

    /// <summary>SANA-WM stage 1: a camera-controlled DiT, the LTX-2 video VAE, and a Gemma-2-2B text encoder, loaded one stage
    /// at a time per run.</summary>
    SanaWm,

    /// <summary>SANA-WM streaming: the chunk-causal distilled DiT (the stage-1 layout), the causal LTX-2 video VAE, and the
    /// same Gemma-2-2B text encoder. The DiT and the VAE decoder stay loaded together for the length of a stream.</summary>
    SanaWmStreaming,
}
