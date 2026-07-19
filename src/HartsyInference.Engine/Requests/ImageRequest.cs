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

    /// <summary>Output width in pixels.</summary>
    public int Width { get; init; } = 1024;

    /// <summary>Output height in pixels.</summary>
    public int Height { get; init; } = 1024;

    /// <summary>Number of denoising steps.</summary>
    public int Steps { get; init; } = 20;

    /// <summary>Classifier-free guidance scale.</summary>
    public float CfgScale { get; init; } = 7.5f;

    /// <summary>RNG seed; negative means a random seed is chosen per request.</summary>
    public long Seed { get; init; } = -1;

    /// <summary>CLIP-skip: number of final text-encoder layers to skip (0 = none).</summary>
    public int ClipSkip { get; init; }

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
