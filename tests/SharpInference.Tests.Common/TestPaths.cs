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
        public static string SchnellQ4KS  => Resolve("FLUX_SCHNELL_Q4KS_PATH", Path.Combine(ModelsDir, "unet", "Flux", "flux1-schnell-Q4_K_S.gguf"));
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

    /// <summary>AuraFlow paths. Default: <c>fal/AuraFlow-v0.3</c> single-file safetensors. The transformer is
    /// 16.5 GB FP16 standalone; the Pile-T5-XL text encoder + SDXL VAE are loaded separately.</summary>
    public static class AuraFlow
    {
        /// <summary>FP8 scaled single-file from <c>calcuis/aura</c>. Bundles transformer + Pile-T5-XL text encoder + AuraFlow VAE in one file under prefixes <c>model.*</c>, <c>text_encoders.pile_t5xl.transformer.*</c>, <c>vae.*</c>.</summary>
        public static string V03Fp8      => Resolve("AURAFLOW_V03_FP8_PATH",   Path.Combine(ModelsDir, "Stable-Diffusion", "AuraFlow", "aura_flow_0.3_fp8_scaled.safetensors"));
        /// <summary>BF16 single-file from <c>fal/AuraFlow-v0.3</c>. Transformer-only — needs separate Pile-T5-XL.</summary>
        public static string V03         => Resolve("AURAFLOW_V03_PATH",       Path.Combine(ModelsDir, "Stable-Diffusion", "AuraFlow", "aura_flow_0.3.safetensors"));
        public static string PileT5XlDir => Resolve("PILE_T5_XL_DIR",          Path.Combine(ModelsDir, "text_encoders", "pile-t5-xl"));
        public static string PileT5XlSpiece => Resolve("PILE_T5_XL_SPIECE",    Path.Combine(Tokenizers.T5XxlSpiece));  // UMT5 vocab compatible
        public static string Vae         => Resolve("AURAFLOW_VAE_PATH",       Path.Combine(ModelsDir, "VAE", "aura_vae.safetensors"));
        public static string SdxlVae     => Resolve("SDXL_VAE_PATH",           Path.Combine(ModelsDir, "VAE", "sdxl_vae.safetensors"));
    }

    /// <summary>Chroma paths. Default: <c>silveroxides/Chroma1-HD-fp8-scaled</c> FP8 single-file safetensors
    /// (BFL-style native format with `distilled_guidance_layer.*` + `double_blocks.*` + `single_blocks.*`).
    /// Reuses T5-XXL bundled in our SD3.5 single-file + Flux VAE bundled in Flux Dev FP8.</summary>
    public static class Chroma
    {
        public static string V1          => Resolve("CHROMA_V1_PATH",          Path.Combine(ModelsDir, "Stable-Diffusion", "Chroma", "Chroma1-HD-fp8mixed-final.safetensors"));
        public static string T5XxlSpiece => Tokenizers.T5XxlSpiece;
        /// <summary>Source for a 16-channel VAE — defaults to Flux Dev FP8 single-file (which bundles a Flux VAE under <c>vae.*</c> keys).</summary>
        public static string VaePath     => Resolve("CHROMA_VAE_PATH",         Vae.FluxVaeSource);
        /// <summary>Source for T5-XXL — defaults to our SD3.5 Medium FP8 single-file (which bundles T5-XXL under <c>text_encoders.t5xxl.transformer.*</c>).</summary>
        public static string T5XxlSource => Resolve("CHROMA_T5XXL_SOURCE",     Sd35.Medium);
    }

    /// <summary>ERNIE-Image paths. Diffusers folder layout (transformer/, text_encoder/, vae/, scheduler/, tokenizer/).</summary>
    public static class ErnieImage
    {
        public static string V1Dir       => Resolve("ERNIE_IMAGE_V1_DIR",       Path.Combine(ModelsDir, "Stable-Diffusion", "ErnieImage", "v1"));
        public static string V1TurboDir  => Resolve("ERNIE_IMAGE_V1_TURBO_DIR", Path.Combine(ModelsDir, "Stable-Diffusion", "ErnieImage", "v1-turbo"));
    }

    /// <summary>Qwen-Image paths. Default expects a single-file checkpoint at <c>Models/Stable-Diffusion/QwenImage/qwen_image_v1.safetensors</c>; the Qwen2.5-VL-7B text encoder and the 16-channel VAE are loaded separately.</summary>
    public static class QwenImage
    {
        public static string V1            => Resolve("QWEN_IMAGE_V1_PATH",     Path.Combine(ModelsDir, "Stable-Diffusion", "QwenImage", "qwen_image_v1.safetensors"));
        public static string TextEncoder   => Resolve("QWEN_IMAGE_TE_PATH",     Path.Combine(ModelsDir, "text_encoders", "qwen2_5_vl_7b.safetensors"));
        public static string Vae           => Resolve("QWEN_IMAGE_VAE_PATH",    Path.Combine(ModelsDir, "VAE", "qwen_image_vae.safetensors"));
    }

    /// <summary>Anima (Cosmos-Predict2 2B Text2Image derivative by CircleStone Labs / Comfy Org). The
    /// transformer is shipped as a single-file safetensors (<c>anima-preview3-base.safetensors</c>);
    /// the Qwen-3 0.6B text encoder and Qwen-Image VAE are loaded separately.</summary>
    public static class Anima
    {
        public static string Transformer  => Resolve("ANIMA_TRANSFORMER_PATH",  Path.Combine(ModelsDir, "Stable-Diffusion", "Anima", "anima-preview3-base.safetensors"));
        public static string Preview3Base => Resolve("ANIMA_PREVIEW3_PATH",     Path.Combine(ModelsDir, "Stable-Diffusion", "Anima", "anima-preview3-base.safetensors"));
        public static string TextEncoder  => Resolve("ANIMA_TE_PATH",           Path.Combine(ModelsDir, "text_encoders", "qwen_3_06b_base.safetensors"));
        public static string Vae          => Resolve("ANIMA_VAE_PATH",          QwenImage.Vae);
        public static string PromptEmbeds => Resolve("ANIMA_PROMPT_EMBEDS",     Path.Combine(ModelsDir, "Stable-Diffusion", "Anima", "TestEmbeddings", "prompt.bin"));
    }

    /// <summary>Kandinsky 5.0 paths. Default expects the diffusers folder layout from
    /// <c>kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers</c> with <c>transformer/</c> + <c>vae/</c>
    /// subdirectories. The dual text encoders (Qwen2.5-VL-7B and CLIP-L) are not yet implemented in
    /// SharpInference, so end-to-end tests rely on pre-computed embeddings stored as raw F32 binaries.</summary>
    public static class Kandinsky5
    {
        /// <summary>Diffusers root for Kandinsky 5 Lite (contains transformer/, vae/, scheduler/, etc.).</summary>
        public static string LiteDir            => Resolve("KANDINSKY5_LITE_DIR",         Path.Combine(ModelsDir, "Stable-Diffusion", "Kandinsky5", "Kandinsky-5.0-T2I-Lite-sft-Diffusers"));
        /// <summary>Transformer subdirectory (default <c>{LiteDir}/transformer</c>).</summary>
        public static string TransformerDir     => Resolve("KANDINSKY5_TRANSFORMER_DIR",  Path.Combine(LiteDir, "transformer"));
        /// <summary>Standalone single-file transformer (alternative to <see cref="TransformerDir"/> for repackaged checkpoints).</summary>
        public static string TransformerSingle  => Resolve("KANDINSKY5_TRANSFORMER_PATH", Path.Combine(ModelsDir, "Stable-Diffusion", "Kandinsky5", "kandinsky5_t2i_lite.safetensors"));
        /// <summary>VAE source. Defaults to the bundled VAE under <c>{LiteDir}/vae</c>; if missing, fall back to the standalone Flux VAE.</summary>
        public static string Vae                => Resolve("KANDINSKY5_VAE_PATH",         Path.Combine(LiteDir, "vae", "diffusion_pytorch_model.safetensors"));
        /// <summary>Pre-computed Qwen2.5-VL sequence embeddings (raw F32 [S_t, in_text_dim=3584]).</summary>
        public static string PromptQwenEmbeds   => Resolve("KANDINSKY5_QWEN_EMBEDS",      Path.Combine(ModelsDir, "Stable-Diffusion", "Kandinsky5", "TestEmbeddings", "prompt_qwen.bin"));
        /// <summary>Pre-computed CLIP-L pooled embeddings (raw F32 [in_text_dim2=768]).</summary>
        public static string PromptClipPooled   => Resolve("KANDINSKY5_CLIP_POOLED",      Path.Combine(ModelsDir, "Stable-Diffusion", "Kandinsky5", "TestEmbeddings", "prompt_clip.bin"));
        /// <summary>Pre-computed Qwen2.5-VL sequence embeddings for the negative prompt.</summary>
        public static string NegPromptQwenEmbeds => Resolve("KANDINSKY5_NEG_QWEN_EMBEDS", Path.Combine(ModelsDir, "Stable-Diffusion", "Kandinsky5", "TestEmbeddings", "neg_prompt_qwen.bin"));
        /// <summary>Pre-computed CLIP-L pooled embeddings for the negative prompt.</summary>
        public static string NegPromptClipPooled => Resolve("KANDINSKY5_NEG_CLIP_POOLED", Path.Combine(ModelsDir, "Stable-Diffusion", "Kandinsky5", "TestEmbeddings", "neg_prompt_clip.bin"));
    }

    /// <summary>Lumina-Image-2.0 paths. The dual text encoder Gemma 2 2B is not yet implemented in
    /// SharpInference, so end-to-end tests rely on pre-computed embeddings shipped as raw F32 binaries —
    /// the same format Diffusers' pipeline accepts via <c>prompt_embeds</c>.</summary>
    public static class Lumina2
    {
        public static string Transformer       => Resolve("LUMINA2_TRANSFORMER_PATH", Path.Combine(ModelsDir, "Stable-Diffusion", "Lumina2", "lumina2_transformer.safetensors"));
        public static string Vae               => Resolve("LUMINA2_VAE_PATH",         Path.Combine(ModelsDir, "Stable-Diffusion", "Lumina2", "vae.safetensors"));
        public static string PromptEmbeds      => Resolve("LUMINA2_PROMPT_EMBEDS",    Path.Combine(ModelsDir, "Stable-Diffusion", "Lumina2", "TestEmbeddings", "prompt.bin"));
        public static string NegPromptEmbeds   => Resolve("LUMINA2_NEG_PROMPT_EMBEDS",Path.Combine(ModelsDir, "Stable-Diffusion", "Lumina2", "TestEmbeddings", "neg_prompt.bin"));
    }

    /// <summary>Hunyuan Image 2.1 paths. The dual text encoder stack (CLIP-L + ByT5/T5) is wired through
    /// the existing encoders; the test skips when checkpoint or tokenizer assets are missing.</summary>
    public static class HunyuanImage
    {
        public static string Transformer  => Resolve("HUNYUAN_IMAGE_TRANSFORMER_PATH", Path.Combine(ModelsDir, "Stable-Diffusion", "HunyuanImage", "hunyuan_image_2_1.safetensors"));
        public static string Vae          => Resolve("HUNYUAN_IMAGE_VAE_PATH",         Path.Combine(ModelsDir, "Stable-Diffusion", "HunyuanImage", "vae.safetensors"));
        public static string ClipL        => Resolve("HUNYUAN_IMAGE_CLIPL_PATH",       Path.Combine(ModelsDir, "Stable-Diffusion", "HunyuanImage", "clip_l.safetensors"));
        public static string T5           => Resolve("HUNYUAN_IMAGE_T5_PATH",          Path.Combine(ModelsDir, "Stable-Diffusion", "HunyuanImage", "t5xxl.safetensors"));
    }

    /// <summary>HiDream i1 paths. Quad text encoder (CLIP-L + CLIP-G + T5-XXL + Llama-3.1).</summary>
    public static class HiDream
    {
        public static string Transformer  => Resolve("HIDREAM_TRANSFORMER_PATH", Path.Combine(ModelsDir, "Stable-Diffusion", "HiDream", "hidream_i1.safetensors"));
        public static string Vae          => Resolve("HIDREAM_VAE_PATH",         Path.Combine(ModelsDir, "Stable-Diffusion", "HiDream", "vae.safetensors"));
        public static string ClipL        => Resolve("HIDREAM_CLIPL_PATH",       Path.Combine(ModelsDir, "Stable-Diffusion", "HiDream", "clip_l.safetensors"));
        public static string ClipG        => Resolve("HIDREAM_CLIPG_PATH",       Path.Combine(ModelsDir, "Stable-Diffusion", "HiDream", "clip_g.safetensors"));
        public static string T5           => Resolve("HIDREAM_T5_PATH",          Path.Combine(ModelsDir, "Stable-Diffusion", "HiDream", "t5xxl.safetensors"));
        public static string Llama        => Resolve("HIDREAM_LLAMA_PATH",       Path.Combine(ModelsDir, "Stable-Diffusion", "HiDream", "llama_3_1_8b.safetensors"));
    }

    /// <summary>OmniGen 2 paths. MLLM text encoder (Qwen2.5-VL).</summary>
    public static class OmniGen2
    {
        public static string Transformer  => Resolve("OMNIGEN2_TRANSFORMER_PATH", Path.Combine(ModelsDir, "Stable-Diffusion", "OmniGen2", "omnigen2.safetensors"));
        public static string Vae          => Resolve("OMNIGEN2_VAE_PATH",         Path.Combine(ModelsDir, "Stable-Diffusion", "OmniGen2", "vae.safetensors"));
        public static string TextEncoder  => Resolve("OMNIGEN2_TEXT_ENCODER",     Path.Combine(ModelsDir, "Stable-Diffusion", "OmniGen2", "qwen2_5_vl.safetensors"));
        public static string PromptEmbeds => Resolve("OMNIGEN2_PROMPT_EMBEDS",    Path.Combine(ModelsDir, "Stable-Diffusion", "OmniGen2", "TestEmbeddings", "prompt.bin"));
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

    /// <summary>Standalone CLIP checkpoint paths (Vision package). HuggingFace <c>CLIPModel</c> single-file
    /// safetensors layout: <c>text_model.*</c>, <c>vision_model.*</c>, <c>text_projection.weight</c>, <c>visual_projection.weight</c>.</summary>
    public static class Clip
    {
        public static string ViTLargePatch14 => Resolve("CLIP_VIT_LARGE_PATH",  Path.Combine(ModelsDir, "clip", "clip-vit-large-patch14.safetensors"));
        public static string ViTHugePatch14  => Resolve("CLIP_VIT_HUGE_PATH",   Path.Combine(ModelsDir, "clip", "clip-vit-h-14-laion2b.safetensors"));
        public static string ViTBigGPatch14  => Resolve("CLIP_VIT_BIGG_PATH",   Path.Combine(ModelsDir, "clip", "clip-vit-bigg-14-laion2b.safetensors"));
    }

    /// <summary>YOLO checkpoint paths (Vision package). Use the Python conversion script under
    /// <c>tests/python-reference/convert_yolov8_pt_to_safetensors.py</c> to produce these from
    /// Ultralytics <c>.pt</c> files — the conversion folds BN into Conv weights for fast inference.</summary>
    public static class Yolo
    {
        public static string V8nFolded => Resolve("YOLOV8N_FOLDED_PATH", Path.Combine(ModelsDir, "yolo", "yolov8n-folded.safetensors"));
        public static string V8sFolded => Resolve("YOLOV8S_FOLDED_PATH", Path.Combine(ModelsDir, "yolo", "yolov8s-folded.safetensors"));
        public static string V8mFolded => Resolve("YOLOV8M_FOLDED_PATH", Path.Combine(ModelsDir, "yolo", "yolov8m-folded.safetensors"));
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
        /// <summary>Llama 3 / 3.1 BPE vocab.json (extracted from <c>tokenizer.json</c>). Used by HiDream.</summary>
        public static string LlamaVocab    => Resolve("LLAMA_VOCAB_PATH",     Path.Combine(ModelsDir, "Tokenizers", "Llama", "vocab.json"));
        /// <summary>Llama 3 / 3.1 BPE merges.txt (extracted from <c>tokenizer.json</c>).</summary>
        public static string LlamaMerges   => Resolve("LLAMA_MERGES_PATH",    Path.Combine(ModelsDir, "Tokenizers", "Llama", "merges.txt"));
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
