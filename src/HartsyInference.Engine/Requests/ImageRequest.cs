namespace HartsyInference.Engine.Requests;

/// <summary>Native, transport-agnostic text-to-image request. Flat common props cover the everyday knobs; the nullable
/// composition objects add LoRA/ControlNet/IP-Adapter/Refiner/img2img/inpaint/regional/variation-seed; and
/// <see cref="Extra"/> carries arch-specific or host-registered params the flat contract does not name. Transports
/// (HTTP, CLI, the SwarmUI backend) map their own inputs onto this — it is the single contract image generation accepts.</summary>
public sealed record ImageRequest
{
    /// <summary>The positive prompt.</summary>
    public required string Prompt { get; init; }

    /// <summary>The negative prompt, or null/empty for none.</summary>
    public string? NegativePrompt { get; init; }

    /// <summary>Output width in pixels; null uses the family's native training width.</summary>
    public int? Width { get; init; }

    /// <summary>Output height in pixels; null uses the family's native training height.</summary>
    public int? Height { get; init; }

    /// <summary>Number of denoising steps; null uses the family's officially recommended step count.</summary>
    public int? Steps { get; init; }

    /// <summary>Classifier-free guidance scale; null uses the family's officially recommended scale (1.0 for distilled models).</summary>
    public float? CfgScale { get; init; }

    /// <summary>CFG-Rescale strength, 0..1; null/0 = off. Pulls a high-CFG guided prediction back toward the
    /// conditional's magnitude to reduce oversaturation/burnt highlights. Only consumed by recipes that wire it
    /// in (SDXL as of 2026-08-10); ignored elsewhere.</summary>
    public float? CfgRescale { get; init; }

    /// <summary>TCFG (Tangential Damping CFG) toggle; null/false = off. Filters the tangential component out of
    /// the unconditional prediction before the CFG combine (https://huggingface.co/papers/2503.18137). Composes
    /// with <see cref="CfgRescale"/>. Only consumed by recipes that wire it in (SDXL as of 2026-08-11); ignored
    /// elsewhere.</summary>
    public bool? Tcfg { get; init; }

    /// <summary>Seamless-tileable axis: <c>null</c>/<c>"false"</c> = off, <c>"true"</c> = both axes, <c>"X-Only"</c>/
    /// <c>"Y-Only"</c> = one axis. Same vocabulary as SwarmUI core's shared <c>SeamlessTileable</c> param. Only
    /// consumed by recipes that wire it in (SDXL as of 2026-08-11); ignored elsewhere.</summary>
    public string? SeamlessTiling { get; init; }

    /// <summary>RNG seed; negative means a random seed is chosen per request.</summary>
    public long Seed { get; init; } = -1;

    /// <summary>CLIP-skip: number of final text-encoder layers to skip; null/0 = none.</summary>
    public int? ClipSkip { get; init; }

    /// <summary>Sampler name (e.g. "euler", "dpmpp_2m"); null uses the recipe default.</summary>
    public string? Sampler { get; init; }

    /// <summary>Scheduler / sigma schedule name; null uses the recipe default.</summary>
    public string? Scheduler { get; init; }

    /// <summary>Sigma-shift adjustment applied to the noise schedule; null uses the recipe default.</summary>
    public double? SigmaShift { get; init; }

    /// <summary>Fraction of the schedule (0..1) at which to stop early; null runs the full schedule.</summary>
    public double? EndStepsEarly { get; init; }

    /// <summary>InstructPix2Pix image-guidance CFG (second CFG term); null when not an ip2p model.</summary>
    public double? InstructPix2PixCfg { get; init; }

    /// <summary>Number of images to generate in one batch.</summary>
    public int Batch { get; init; } = 1;

    /// <summary>Per-request swappable-component overrides (VAE / text encoders); null keeps recipe defaults.</summary>
    public ComponentOverrides? Components { get; init; }

    /// <summary>LoRA stack to fuse; null for none.</summary>
    public LoraStack? Loras { get; init; }

    /// <summary>ControlNet conditioning layers; null/empty for none.</summary>
    public IReadOnlyList<ControlNetConditioning>? ControlNets { get; init; }

    /// <summary>Image-prompt (IP-Adapter / Redux / FaceID) conditioning; null for none.</summary>
    public IpAdapter? IpAdapter { get; init; }

    /// <summary>Second-pass refiner; null for none.</summary>
    public Refiner? Refiner { get; init; }

    /// <summary>Image-to-image init; null for pure text-to-image.</summary>
    public Img2Img? Img2Img { get; init; }

    /// <summary>Inpaint mask; null for none.</summary>
    public Inpaint? Inpaint { get; init; }

    /// <summary>Regional / segment prompting; null for none.</summary>
    public Regional? Regional { get; init; }

    /// <summary>Variation-seed blending; null for none.</summary>
    public VariationSeed? VariationSeed { get; init; }

    /// <summary>Arch-specific or host-registered params not named by the flat contract (keys are host-defined).</summary>
    public IReadOnlyDictionary<string, object> Extra { get; init; } = new Dictionary<string, object>();
}
