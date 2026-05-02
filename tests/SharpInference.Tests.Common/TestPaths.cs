namespace SharpInference.Tests.Common;

/// <summary>Single source of truth for all model and tokenizer paths used by tests. Cross-OS via RepoRoot. Every path is overridable via an environment variable for CI or non-standard layouts.</summary>
public static class TestPaths
{
    /// <summary>Models directory. Override with SHARPINFERENCE_MODELS_DIR.</summary>
    public static string ModelsDir { get; } =
        Env("SHARPINFERENCE_MODELS_DIR") ?? Path.Combine(RepoRoot.Path, "Models");

    /// <summary>Output directory for generated images. Override with SHARPINFERENCE_OUTPUT_DIR.</summary>
    public static string OutputDir { get; } =
        Env("SHARPINFERENCE_OUTPUT_DIR") ?? Path.Combine(RepoRoot.Path, "Output");

    /// <summary>Flux family checkpoint paths.</summary>
    public static class Flux
    {
        public static string Schnell      => Resolve("FLUX_SCHNELL_PATH",      Path.Combine(ModelsDir, "Stable-Diffusion", "Flux", "flux1-schnell-fp8.safetensors"));
        public static string Dev          => Resolve("FLUX_DEV_PATH",          Path.Combine(ModelsDir, "Stable-Diffusion", "Flux", "flux1-dev-fp8.safetensors"));
        public static string KreaFp8      => Resolve("FLUX_KREA_FP8_PATH",     Path.Combine(ModelsDir, "Stable-Diffusion", "Flux", "flux1-krea-dev_fp8_scaled.safetensors"));
        public static string Kontext      => Resolve("FLUX_KONTEXT_PATH",      Path.Combine(ModelsDir, "Stable-Diffusion", "Flux", "flux1-dev-kontext_fp8_scaled.safetensors"));
        public static string Canny        => Resolve("FLUX_CANNY_PATH",        Path.Combine(ModelsDir, "Stable-Diffusion", "Flux", "flux1-canny-dev.safetensors"));
        public static string Fill         => Resolve("FLUX_FILL_PATH",         Path.Combine(ModelsDir, "Stable-Diffusion", "Flux", "flux1-fill-dev.safetensors"));
    }

    /// <summary>Flux 2 family checkpoint paths.</summary>
    public static class Flux2
    {
        public static string Dev          => Resolve("FLUX2_DEV_PATH",         Path.Combine(ModelsDir, "Stable-Diffusion", "Flux2", "flux2_dev_fp8mixed.safetensors"));
        public static string Klein        => Resolve("FLUX2_KLEIN_PATH",       Path.Combine(ModelsDir, "Stable-Diffusion", "Flux2", "flux-2-klein-4b.safetensors"));
    }

    /// <summary>Z-Image family checkpoint paths.</summary>
    public static class ZImage
    {
        public static string Turbo        => Resolve("ZIMAGE_TURBO_PATH",      Path.Combine(ModelsDir, "Stable-Diffusion", "ZImage", "SwarmUI_Z-Image-Turbo-FP8Mix.safetensors"));
        public static string BaseBf16     => Resolve("ZIMAGE_BASE_BF16_PATH",  Path.Combine(ModelsDir, "Stable-Diffusion", "ZImage", "z_image_base-bf16.safetensors"));
        public static string BaseFp8      => Resolve("ZIMAGE_BASE_FP8_PATH",   Path.Combine(ModelsDir, "Stable-Diffusion", "ZImage", "z_image_base-nvfp8-mixed.safetensors"));

        /// <summary>Resolves to the first existing Z-Image-Base checkpoint (FP8 → BF16). Override with ZIMAGE_BASE_PATH.</summary>
        public static string Base
        {
            get
            {
                string? envOverride = Env("ZIMAGE_BASE_PATH");
                if (envOverride is not null)
                    return envOverride;
                if (File.Exists(BaseFp8)) return BaseFp8;
                return BaseBf16;
            }
        }
    }

    /// <summary>Stable Diffusion 1.5 paths. Assets are not bundled — tests skip when missing.</summary>
    public static class Sd15
    {
        public static string SingleFile   => Resolve("SD15_SINGLE_FILE_PATH",  Path.Combine(ModelsDir, "Stable-Diffusion", "SD15", "v1-5-pruned-emaonly.safetensors"));
        public static string DiffusersDir => Resolve("SD15_MODEL_DIR",         Path.Combine(ModelsDir, "Stable-Diffusion", "SD15"));
    }

    /// <summary>SDXL paths. Assets are not bundled — tests skip when missing.</summary>
    public static class Sdxl
    {
        public static string SingleFile   => Resolve("SDXL_SINGLE_FILE_PATH",  Path.Combine(ModelsDir, "Stable-Diffusion", "SDXL", "Juggernaut_XL_-_Ragnarok_by_RunDiffusion.safetensors"));
        public static string DiffusersDir => Resolve("SDXL_MODEL_DIR",         Path.Combine(ModelsDir, "Stable-Diffusion", "SDXL"));
    }

    /// <summary>SD3 paths. Assets are not bundled — tests skip when missing.</summary>
    public static class Sd3
    {
        public static string SingleFile   => Resolve("SD3_SINGLE_FILE_PATH",   Path.Combine(ModelsDir, "Stable-Diffusion", "SD3", "sd3_medium_incl_clips_t5xxlfp16.safetensors"));
        public static string DiffusersDir => Resolve("SD3_MODEL_DIR",          Path.Combine(ModelsDir, "Stable-Diffusion", "SD3"));
    }

    /// <summary>SD3.5 paths. Assets are not bundled — tests skip when missing. FP8-bundled single-file checkpoints from Comfy-Org/stable-diffusion-3.5-fp8 are the default; set env vars to override for FP16 / community quants.</summary>
    public static class Sd35
    {
        public static string Medium       => Resolve("SD35_MEDIUM_PATH",       Path.Combine(ModelsDir, "Stable-Diffusion", "SD3", "sd3.5_medium_incl_clips_t5xxlfp8scaled.safetensors"));
        public static string Large        => Resolve("SD35_LARGE_PATH",        Path.Combine(ModelsDir, "Stable-Diffusion", "SD3", "sd3.5_large_fp8_scaled.safetensors"));
        public static string LargeTurbo   => Resolve("SD35_LARGE_TURBO_PATH",  Path.Combine(ModelsDir, "Stable-Diffusion", "SD3", "sd3.5_large_turbo.safetensors"));
    }

    /// <summary>Standalone text encoder weights.</summary>
    public static class TextEncoders
    {
        public static string Mistral3SmallFp8 => Resolve("MISTRAL_FP8_PATH",   Path.Combine(ModelsDir, "text_encoders", "mistral_3_small_flux2_fp8.safetensors"));
        public static string Qwen3_4B         => Resolve("QWEN3_4B_PATH",      Path.Combine(ModelsDir, "text_encoders", "qwen_3_4b.safetensors"));
    }

    /// <summary>Standalone VAE weights.</summary>
    public static class Vae
    {
        public static string Flux2          => Resolve("FLUX2_VAE_PATH",       Path.Combine(ModelsDir, "VAE", "flux2-vae.safetensors"));
        /// <summary>Flux1 dev checkpoint, used as the VAE source for Z-Image generation.</summary>
        public static string FluxVaeSource  => Resolve("FLUX_VAE_SOURCE_PATH", Flux.Dev);
    }

    /// <summary>LoRA adapter weights.</summary>
    public static class Lora
    {
        public static string YearbookFluxSchnell => Resolve("YEARBOOK_LORA_PATH", Path.Combine(ModelsDir, "Lora", "yearbook-photo-flux-schnell-v1.safetensors"));
    }

    /// <summary>Tokenizer asset paths.</summary>
    public static class Tokenizers
    {
        public static string ClipVocab     => Resolve("CLIP_VOCAB_PATH",      Path.Combine(ModelsDir, "Tokenizers", "CLIP", "clip_vocab.json"));
        public static string ClipMerges    => Resolve("CLIP_MERGES_PATH",     Path.Combine(ModelsDir, "Tokenizers", "CLIP", "clip_merges.txt"));
        public static string T5Spiece      => Resolve("T5_SPIECE_PATH",       Path.Combine(ModelsDir, "Tokenizers", "T5", "t5_spiece.model"));
        public static string T5XxlSpiece   => Resolve("T5_XXL_SPIECE_PATH",   Path.Combine(ModelsDir, "Tokenizers", "T5", "t5_xxl_spiece.model"));
        public static string LlamaTokenizer => Resolve("LLAMA_TOKENIZER_PATH", Path.Combine(ModelsDir, "Tokenizers", "Llama", "llama_tokenizer.model"));
        public static string Qwen3Dir      => Resolve("QWEN3_TOKENIZER_DIR",  Path.Combine(ModelsDir, "Tokenizers", "Qwen3"));
        public static string Qwen3Vocab    => Resolve("QWEN3_VOCAB_PATH",     Path.Combine(Qwen3Dir, "vocab.json"));
        public static string Qwen3Merges   => Resolve("QWEN3_MERGES_PATH",    Path.Combine(Qwen3Dir, "merges.txt"));
        public static string Qwen3Config   => Resolve("QWEN3_CONFIG_PATH",    Path.Combine(Qwen3Dir, "config.json"));
    }

    /// <summary>Resolves a cross-runtime reference image path inside OutputDir/References.</summary>
    public static string ReferenceImage(string envVar, string fileName) =>
        Resolve(envVar, Path.Combine(OutputDir, "References", fileName));

    private static string Resolve(string envVar, string fallback) =>
        Env(envVar) ?? fallback;

    private static string? Env(string name)
    {
        string? v = Environment.GetEnvironmentVariable(name);
        return string.IsNullOrEmpty(v) ? null : v;
    }
}
