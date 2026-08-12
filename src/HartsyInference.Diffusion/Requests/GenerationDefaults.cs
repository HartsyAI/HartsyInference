namespace HartsyInference.Diffusion.Requests;

/// <summary>Per-model reference generation defaults. Since <see cref="TextToImageRequest.Steps"/> /
/// <see cref="TextToImageRequest.CfgScale"/> / <see cref="TextToImageRequest.Width"/> /
/// <see cref="TextToImageRequest.Height"/> are nullable, each pipeline resolves an omitted (null) value against
/// its model's entry here via <see cref="Resolve"/>. Values mirror the official diffusers / upstream pipeline
/// <c>__call__</c> defaults.</summary>
public readonly record struct GenerationDefaults(int Steps, float CfgScale, int Width, int Height)
{
    /// <summary>Resolves a request's (possibly null) Steps/CfgScale/Width/Height against these defaults.</summary>
    public (int Steps, float CfgScale, int Width, int Height) Resolve(TextToImageRequest r)
        => (r.Steps ?? Steps, r.CfgScale ?? CfgScale, r.Width ?? Width, r.Height ?? Height);

    /// <summary>The historical engine-generic default (20 / 7.5 / 512²). Used as the fallback for models that
    /// have no tuned reference default of their own.</summary>
    public static GenerationDefaults Generic => new(20, 7.5f, 512, 512);

    // ── Image models (reference __call__ defaults) ──
    /// <summary>SD 1.5: 50 steps, cfg 7.5, 512².</summary>
    public static GenerationDefaults Sd15 => new(50, 7.5f, 512, 512);
    /// <summary>SDXL base: 50 steps, cfg 5.0, 1024².</summary>
    public static GenerationDefaults Sdxl => new(50, 5.0f, 1024, 1024);
    /// <summary>SD 3.5: 28 steps, cfg 7.0, 1024².</summary>
    public static GenerationDefaults Sd35 => new(28, 7.0f, 1024, 1024);
    /// <summary>Flux dev / Tools / Krea: 28 steps, guidance 3.5 (Krea 4.5), 1024².</summary>
    public static GenerationDefaults FluxDev => new(28, 3.5f, 1024, 1024);
    /// <summary>Flux schnell: 4 steps, guidance ignored, 1024².</summary>
    public static GenerationDefaults FluxSchnell => new(4, 3.5f, 1024, 1024);
    /// <summary>Flux.2 dev / Klein: 50 steps, guidance 4.0, 1024².</summary>
    public static GenerationDefaults Flux2 => new(50, 4.0f, 1024, 1024);
    /// <summary>Z-Image Turbo: 8 steps, cfg 1.0, 1024².</summary>
    public static GenerationDefaults ZImageTurbo => new(8, 1.0f, 1024, 1024);

    /// <summary>Z-Image Base: official undistilled CFG sampling defaults.</summary>
    public static GenerationDefaults ZImageBase => new(50, 5.0f, 1024, 1024);
    /// <summary>Qwen-Image: 50 steps, true-cfg 4.0, 1024².</summary>
    public static GenerationDefaults QwenImage => new(50, 4.0f, 1024, 1024);
    /// <summary>AuraFlow: 50 steps, guidance 3.5, 1024².</summary>
    public static GenerationDefaults AuraFlow => new(50, 3.5f, 1024, 1024);
    /// <summary>Lumina-Image-2.0: 30 steps, guidance 4.0, 1024².</summary>
    public static GenerationDefaults Lumina2 => new(30, 4.0f, 1024, 1024);
    /// <summary>Chroma: 35 steps, guidance 5.0, 1024² (Radiance 50 / 3.5).</summary>
    public static GenerationDefaults Chroma => new(35, 5.0f, 1024, 1024);
    /// <summary>HiDream Full: 50 steps, cfg 5.0, 1024² (Dev 28/0, Fast 16/0).</summary>
    public static GenerationDefaults HiDreamFull => new(50, 5.0f, 1024, 1024);
    /// <summary>Kandinsky-5 image: 50 steps, guidance 3.5, 1024².</summary>
    public static GenerationDefaults Kandinsky5Image => new(50, 3.5f, 1024, 1024);
    /// <summary>OmniGen2: 28 steps, text-guidance 4.0, 1024².</summary>
    public static GenerationDefaults OmniGen2 => new(28, 4.0f, 1024, 1024);
    /// <summary>HunyuanImage 2.1: 50 steps, cfg 3.5, 2048² (distilled 8 / 3.25).</summary>
    public static GenerationDefaults HunyuanImage => new(50, 3.5f, 2048, 2048);
    /// <summary>ERNIE-Image: 50 steps, cfg 4.0, 1024² (Turbo 8 / 1.0).</summary>
    public static GenerationDefaults ErnieImage => new(50, 4.0f, 1024, 1024);
    /// <summary>F-Lite: 30 steps, cfg 6.0, 1024².</summary>
    public static GenerationDefaults FLite => new(30, 6.0f, 1024, 1024);
}
