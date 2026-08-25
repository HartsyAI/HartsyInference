namespace HartsyInference.Engine.Recipes;

/// <summary>The memory and multi-device behaviours a recipe declares it actually wires; anything configured that the resolved recipe does not declare is reported once, by name, instead of quietly doing nothing.</summary>
/// <remarks>Every one of these is configured somewhere other than the recipe — a backend setting, a second backend
/// built by the placement, a request override — so a recipe that simply does not consume one still produces a working
/// generation that ignores it. From the outside that is indistinguishable from "wired and not helping", which is the
/// failure this enum exists to make impossible. Declaring nothing is the honest default for a family nobody has wired,
/// and is what 37 of the 39 recipes were silently doing before this existed.</remarks>
[Flags]
public enum MemoryCapabilities
{
    /// <summary>Nothing beyond running on the primary backend: every VRAM lever and every placement device is ignored.</summary>
    None = 0,

    /// <summary>The denoiser exposes streamable blocks AND the pipeline drives them, so weight streaming can trade speed for headroom.</summary>
    BlockStreaming = 1,

    /// <summary>The DiT's block range is split across shard backends to pool their VRAM.</summary>
    DitSharding = 2,

    /// <summary>The CFG uncond branch runs on a second backend concurrently with cond.</summary>
    CfgParallel = 4,

    /// <summary>The token sequence is split across context-parallel ranks.</summary>
    ContextParallel = 8,

    /// <summary>Text encoders and/or the VAE can run on backends other than the primary.</summary>
    ComponentPlacement = 16,

    /// <summary>Weights are released at phase boundaries so a later phase can have the space.</summary>
    PhaseUnload = 32,

    /// <summary>The work is chunked or tiled, so shrinking the chunk lowers the activation peak.</summary>
    Chunking = 64,

    /// <summary>Cross-step caches can be stored at half precision.</summary>
    HalfPrecisionCaches = 128,

    /// <summary>Quantized weights can stay compressed on device with a transient dequant per call.</summary>
    QuantizedCompute = 256,
}
