namespace HartsyInference.Engine.Audio;

/// <summary>Load-time precision for the big codec-token LMs (YuE Stage-1/2) — a VRAM-fit decision, not a math
/// one, so it is a policy the engine resolves rather than a hardcoded loader constant.</summary>
public enum AudioLmQuant
{
    /// <summary>Q4_K (~3.5 GB for a 7B; decode hits the fast dp4a GEMV) — the single-GPU fit default.</summary>
    Q4K,

    /// <summary>Q8_0 — near-lossless at ~7.5 GB for a 7B, but decode takes the naive F32 GEMV
    /// (~an order of magnitude slower per weight-read than Q4_K's dp4a path).</summary>
    Q8,

    /// <summary>No quantization — checkpoint precision (bf16). The quality path a layer-split placement
    /// exists to make affordable: pool VRAM across GPUs instead of crushing the LM to fit one card.</summary>
    Off,
}
