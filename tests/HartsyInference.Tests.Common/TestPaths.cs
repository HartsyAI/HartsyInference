namespace HartsyInference.Tests.Common;

/// <summary>Single source of truth for all model and tokenizer paths used by tests. Cross-OS via RepoRoot. Every path is overridable via an environment variable for CI or non-standard layouts.</summary>
public static class TestPaths
{
    /// <summary>Models directory. Override with HARTSYINFERENCE_MODELS_DIR.</summary>
    public static string ModelsDir { get; } =
        Env("HARTSYINFERENCE_MODELS_DIR") ?? Path.Combine(RepoRoot.Path, "Models");

    /// <summary>Output directory for generated images. Override with HARTSYINFERENCE_OUTPUT_DIR.</summary>
    public static string OutputDir { get; } =
        Env("HARTSYINFERENCE_OUTPUT_DIR") ?? Path.Combine(RepoRoot.Path, "Output");

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
        public static string PileT5XlSpiece => Resolve("PILE_T5_XL_SPIECE",    Path.Combine(ModelsDir, "Tokenizers", "T5", "pile_t5xl_spiece.model"));  // Pile-T5 uses its OWN LLaMA-derived SentencePiece (vocab 32000) — NOT t5_xxl_spiece (different token ids). Fixes AuraFlow off-prompt conditioning.
        public static string Vae         => Resolve("AURAFLOW_VAE_PATH",       Path.Combine(ModelsDir, "VAE", "aura_vae.safetensors"));
        public static string SdxlVae     => Resolve("SDXL_VAE_PATH",           Path.Combine(ModelsDir, "VAE", "sdxl_vae.safetensors"));
    }

    /// <summary>Chroma paths. Default: <c>silveroxides/Chroma1-HD-fp8-scaled</c> FP8 single-file safetensors
    /// (BFL-style native format with `distilled_guidance_layer.*` + `double_blocks.*` + `single_blocks.*`).
    /// Reuses T5-XXL bundled in our SD3.5 single-file + Flux VAE bundled in Flux Dev FP8.</summary>
    public static class Chroma
    {
        public static string V1          => Resolve("CHROMA_V1_PATH",          Path.Combine(ModelsDir, "Stable-Diffusion", "Chroma", "Chroma1-HD-fp8mixed-final.safetensors"));
        /// <summary>Q4_K_M GGUF (`silveroxides/Chroma1-HD-GGUF`, 5.6 GB). Lets the FAST cached weight-cast path fit 24 GB where fp8+cache OOMs.</summary>
        public static string V1Gguf      => Resolve("CHROMA_V1_GGUF",          Path.Combine(ModelsDir, "Stable-Diffusion", "Chroma", "Chroma1-HD-Q4_K_M.gguf"));
        public static string T5XxlSpiece => Tokenizers.T5XxlSpiece;
        /// <summary>Source for a 16-channel VAE — defaults to Flux Dev FP8 single-file (which bundles a Flux VAE under <c>vae.*</c> keys).</summary>
        public static string VaePath     => Resolve("CHROMA_VAE_PATH",         Vae.FluxVaeSource);
        /// <summary>Source for T5-XXL — defaults to our SD3.5 Medium FP8 single-file (which bundles T5-XXL under <c>text_encoders.t5xxl.transformer.*</c>).</summary>
        public static string T5XxlSource => Resolve("CHROMA_T5XXL_SOURCE",     Sd35.Medium);
    }

    /// <summary>ACE-Step v1 (3.5B music DiT) paths. Expects the HF repo folder layout (<c>ace_step_transformer/</c>,
    /// <c>music_dcae_f8c8/</c>, <c>music_vocoder/</c>, <c>umt5-base/</c>); the lyric tokenizer needs the one-time
    /// <c>vocab.json</c>/<c>merges.txt</c> export described in <c>AceStepCheckpointConverter</c>.</summary>
    public static class AceStep
    {
        public static string RootDir => Resolve("ACE_STEP_DIR", Path.Combine(ModelsDir, "Music", "ACE-Step-v1-3.5B"));
        public static string TransformerPath => Path.Combine(RootDir, "ace_step_transformer", "diffusion_pytorch_model.safetensors");
        public static string DcaePath => Path.Combine(RootDir, "music_dcae_f8c8", "diffusion_pytorch_model.safetensors");
        public static string VocoderPath => Path.Combine(RootDir, "music_vocoder", "diffusion_pytorch_model.safetensors");
        public static string Umt5BaseDir => Path.Combine(RootDir, "umt5-base");
        public static string LyricVocabPath => Resolve("ACE_STEP_LYRIC_VOCAB", Path.Combine(RootDir, "vocab.json"));
        public static string LyricMergesPath => Resolve("ACE_STEP_LYRIC_MERGES", Path.Combine(RootDir, "merges.txt"));
    }

    /// <summary>ERNIE-Image paths. Diffusers folder layout (transformer/, text_encoder/, vae/, scheduler/, tokenizer/).</summary>
    public static class ErnieImage
    {
        public static string V1Dir       => Resolve("ERNIE_IMAGE_V1_DIR",       Path.Combine(ModelsDir, "Stable-Diffusion", "ErnieImage", "v1"));
        public static string V1TurboDir  => Resolve("ERNIE_IMAGE_V1_TURBO_DIR", Path.Combine(ModelsDir, "Stable-Diffusion", "ErnieImage", "v1-turbo"));
    }

    /// <summary>Ideogram 4 paths. Expects a root folder with either diffusers (<c>transformer/</c>, <c>unconditional_transformer/</c>, <c>text_encoder/</c>, <c>vae/</c>) or Comfy-Org (<c>diffusion_models/</c>, <c>text_encoders/</c>, <c>vae/</c>) layout. The Qwen3-VL chat tokenizer is loaded separately (see <see cref="Tokenizers.Qwen3Dir"/>).</summary>
    public static class Ideogram4
    {
        public static string Dir => Resolve("IDEOGRAM4_DIR", Path.Combine(ModelsDir, "Stable-Diffusion", "Ideogram4"));
    }

    /// <summary>Lance (ByteDance) paths. <see cref="Dir"/> is the variant folder holding <c>model.safetensors</c> (e.g. <c>Lance_3B</c>); <see cref="VaePath"/> is the Wan2.2 VAE converted to safetensors (the original ships as <c>Wan2.2_VAE.pth</c>). The Qwen2 chat tokenizer is loaded separately.</summary>
    public static class Lance
    {
        public static string Dir => Resolve("LANCE_3B_DIR", Path.Combine(ModelsDir, "Stable-Diffusion", "Lance", "Lance_3B"));
        public static string VideoDir => Resolve("LANCE_3B_VIDEO_DIR", Path.Combine(ModelsDir, "Stable-Diffusion", "Lance", "Lance_3B_Video"));
        public static string VaePath => Resolve("LANCE_VAE_PATH", Path.Combine(ModelsDir, "Stable-Diffusion", "Lance", "wan22_vae.safetensors"));
    }

    /// <summary>Wan-Video (Wan2.2 TI2V-5B) paths. The DiT single-file is the Comfy-Org repackage (original Wan
    /// naming — <c>WanVideoCheckpointConverter</c> renames to diffusers); the VAE is the same Wan2.2 VAE safetensors
    /// Lance uses (loaded via <c>LanceCheckpointConverter.LoadVae</c>); text encoding is umT5-XXL.</summary>
    public static class WanVideo
    {
        public static string Ti2V5B        => Resolve("WAN22_TI2V_5B_PATH",   Path.Combine(ModelsDir, "Stable-Diffusion", "Wan", "wan2.2_ti2v_5B_fp16.safetensors"));
        public static string VaePath       => Resolve("WAN22_VAE_PATH",       Lance.VaePath);
        public static string Umt5Xxl       => Resolve("UMT5_XXL_PATH",        Path.Combine(ModelsDir, "text_encoders", "umt5_xxl_fp8_e4m3fn_scaled.safetensors"));
        public static string Umt5XxlSpiece => Tokenizers.Umt5XxlSpiece;
        /// <summary>Optional Wan LoRA applied by the generation test when present (kohya/musubi, Comfy diffusion_model, or diffusers-PEFT format).</summary>
        public static string LoraPath      => Resolve("WAN_LORA_PATH",        Path.Combine(ModelsDir, "Lora", "wan_lora.safetensors"));
    }

    /// <summary>LTX-Video paths. The single file bundles DiT + VAE (original naming — <c>LtxVideoCheckpointConverter</c>
    /// renames + buckets); T5-XXL is extracted from the SD3.5 bundle by default, like Chroma.</summary>
    public static class LtxVideo
    {
        public static string SingleFile  => Resolve("LTX_VIDEO_PATH",        Path.Combine(ModelsDir, "Stable-Diffusion", "LtxVideo", "ltx-video-2b-v0.9.safetensors"));
        public static string T5XxlSource => Resolve("LTX_T5XXL_SOURCE",      Sd35.Medium);
        /// <summary>Standalone T5-XXL encoder (fp8-scaled comfy layout, <c>encoder.block.*</c>/<c>shared.weight</c>);
        /// preferred over the SD3.5-bundle extraction when present. Reuses the HiDream t5xxl file by default.</summary>
        public static string T5XxlStandalone => Resolve("LTX_T5XXL_STANDALONE", Path.Combine(ModelsDir, "Stable-Diffusion", "HiDream", "t5xxl.safetensors"));
        public static string T5XxlSpiece => Tokenizers.T5XxlSpiece;
    }

    /// <summary>LTX-2 (dual-stream audio+video) paths. The single file bundles the DiT + video VAE + audio VAE +
    /// vocoder (original naming); the Gemma-3-12B text encoder + its SentencePiece tokenizer ship separately.</summary>
    public static class LtxVideo2
    {
        public static string SingleFile     => Resolve("LTX2_PATH",            Path.Combine(ModelsDir, "Stable-Diffusion", "LtxVideo2", "ltx-2.3-22b-dev-fp8.safetensors"));
        public static string GemmaEncoder   => Resolve("LTX2_GEMMA_PATH",      Path.Combine(ModelsDir, "text_encoders", "gemma_3_12B_it_fp8_scaled.safetensors"));
        public static string GemmaTokenizer => Resolve("LTX2_GEMMA_TOKENIZER", Path.Combine(ModelsDir, "Tokenizers", "Gemma", "tokenizer.model"));
    }

    /// <summary>HunyuanVideo (Tencent 13B T2V) paths. The DiT single-file is the Comfy-Org repacked bf16 (original
    /// Tencent naming — <see cref="HartsyInference.ModelHandler.CheckpointConverters.HunyuanVideoCheckpointConverter"/>
    /// remaps to the hybrid DiT layout); the VAE is the HunyuanVideo 3D VAE; the primary text encoder is
    /// LLaVA-Llama-3-8B (fp8 scaled) run via <c>LlamaStyleEncoder</c>(Llama31_8B) and CLIP-L for the pooled vector.
    /// The Llama-3 + CLIP tokenizers are the embedded/standalone assets.</summary>
    public static class HunyuanVideo
    {
        public static string Dit    => Resolve("HUNYUAN_VIDEO_DIT_PATH",   Path.Combine(ModelsDir, "Stable-Diffusion", "HunyuanVideo", "hunyuan_video_t2v_720p_bf16.safetensors"));
        public static string Vae    => Resolve("HUNYUAN_VIDEO_VAE_PATH",   Path.Combine(ModelsDir, "Stable-Diffusion", "HunyuanVideo", "hunyuan_video_vae_bf16.safetensors"));
        public static string Llava  => Resolve("HUNYUAN_VIDEO_LLAVA_PATH", Path.Combine(ModelsDir, "text_encoders", "llava_llama3_fp8_scaled.safetensors"));
        public static string ClipL  => Resolve("HUNYUAN_VIDEO_CLIPL_PATH", Path.Combine(ModelsDir, "text_encoders", "clip_l.safetensors"));
    }

    /// <summary>Qwen-Image paths. Default expects a single-file checkpoint at <c>Models/Stable-Diffusion/QwenImage/qwen_image_v1.safetensors</c>; the Qwen2.5-VL-7B text encoder and the 16-channel VAE are loaded separately.</summary>
    public static class QwenImage
    {
        public static string V1            => Resolve("QWEN_IMAGE_V1_PATH",     Path.Combine(ModelsDir, "Stable-Diffusion", "QwenImage", "qwen_image_v1.safetensors"));
        /// <summary>Q4_K_M GGUF (QuantStack/Qwen-Image-GGUF) — ~13 GB, fits the 4090 where the 20 GB fp8 single-file does not. Bare diffusers keys, identity key-map.</summary>
        public static string V1Gguf        => Resolve("QWEN_IMAGE_V1_GGUF",     Path.Combine(ModelsDir, "Stable-Diffusion", "QwenImage", "Qwen_Image-Q4_K_M.gguf"));
        public static string TextEncoder   => Resolve("QWEN_IMAGE_TE_PATH",     Path.Combine(ModelsDir, "text_encoders", "qwen2_5_vl_7b.safetensors"));
        public static string Vae           => Resolve("QWEN_IMAGE_VAE_PATH",    Path.Combine(ModelsDir, "VAE", "qwen_image_vae.safetensors"));
    }

    /// <summary>Qwen-Image-Edit (20B). Image-conditioned editing branch of Qwen-Image; reuses the Qwen-Image transformer
    /// arch + the Qwen2.5-VL-7B encoder + 16-ch VAE. A Q5_K_M GGUF is staged (SwarmUI repack). The edit path needs a
    /// reference-image conditioning pipeline (not yet built — see MODEL_STATUS_IMAGE §Edit variants).</summary>
    public static class QwenImageEdit
    {
        public static string Gguf => Resolve("QWEN_IMAGE_EDIT_GGUF", Path.Combine(ModelsDir, "Stable-Diffusion", "QwenImageEdit", "qwen-image-edit-2511-Q5_K_M.gguf"));
    }

    /// <summary>Krea 2 (12.9B single-stream MMDiT). fp8_scaled checkpoints staged as
    /// <c>Krea2/{Base,Turbo}/</c> each with the transformer file in root + <c>text_encoder/</c> (Qwen3-VL-4B) +
    /// <c>vae/</c> (Qwen-Image VAE) — the dir layout <see cref="HartsyInference.ModelHandler.CheckpointConverters.Krea2CheckpointConverter"/> expects (pass the dir as rootPath).</summary>
    public static class Krea2
    {
        public static string BaseDir  => Resolve("KREA2_BASE_DIR",  Path.Combine(ModelsDir, "Stable-Diffusion", "Krea2", "Base"));
        public static string TurboDir => Resolve("KREA2_TURBO_DIR", Path.Combine(ModelsDir, "Stable-Diffusion", "Krea2", "Turbo"));
    }

    /// <summary>Boogu-Image 0.1 (10B OmniGen2/Lumina lineage). Comfy fp8_scaled, staged as <c>Boogu/{Base,Turbo,Edit}/</c>
    /// each with <c>transformer/</c> (fp8) + <c>mllm/</c> (Qwen3-VL-8B) + <c>vae/</c> (Flux VAE) — the layout
    /// <see cref="HartsyInference.ModelHandler.CheckpointConverters.BooguImageCheckpointConverter"/> expects.</summary>
    public static class Boogu
    {
        public static string BaseDir  => Resolve("BOOGU_BASE_DIR",  Path.Combine(ModelsDir, "Stable-Diffusion", "Boogu", "Base"));
        public static string TurboDir => Resolve("BOOGU_TURBO_DIR", Path.Combine(ModelsDir, "Stable-Diffusion", "Boogu", "Turbo"));
        public static string EditDir  => Resolve("BOOGU_EDIT_DIR",  Path.Combine(ModelsDir, "Stable-Diffusion", "Boogu", "Edit"));
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
    /// HartsyInference, so end-to-end tests rely on pre-computed embeddings stored as raw F32 binaries.</summary>
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
    /// HartsyInference, so end-to-end tests rely on pre-computed embeddings shipped as raw F32 binaries —
    /// the same format Diffusers' pipeline accepts via <c>prompt_embeds</c>.</summary>
    public static class Lumina2
    {
        // Lumina2CheckpointConverter + Lumina2Transformer.LoadWeights expect the DIFFUSERS key naming
        // (split attn.to_q/k/v, feed_forward.linear_1/2/3, time_caption_embed, norm_out) — the original AlphaVLLM
        // single-file (`lumina2_transformer.safetensors`: fused attention.qkv, t_embedder, final_layer) does NOT load.
        // Use the diffusers-format weights from Alpha-VLLM/Lumina-Image-2.0 (transformer/ merged + vae/). Verified ✅.
        public static string Transformer       => Resolve("LUMINA2_TRANSFORMER_PATH", Path.Combine(ModelsDir, "Stable-Diffusion", "Lumina2", "lumina2_transformer_diffusers.safetensors"));
        public static string Vae               => Resolve("LUMINA2_VAE_PATH",         Path.Combine(ModelsDir, "Stable-Diffusion", "Lumina2", "vae_diffusers", "vae.safetensors"));
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

    /// <summary>SigLIP checkpoint paths. HuggingFace single-file safetensors (no key remapping required — our SigLIP encoder uses the HF key names directly).</summary>
    public static class Siglip
    {
        public static string Base16_224 => Resolve("SIGLIP_BASE_224_PATH", Path.Combine(ModelsDir, "clip", "siglip-base-patch16-224.safetensors"));
    }

    /// <summary>Depth-Anything-V2 checkpoints — the official <c>.pth</c> files from HF
    /// <c>depth-anything/Depth-Anything-V2-Small</c> / <c>-Large</c> (torch.hub-style keys, loaded via
    /// <c>PytorchPickleLoader</c>, no conversion needed).</summary>
    public static class DepthAnything
    {
        public static string VitS => Resolve("DEPTH_ANYTHING_V2_VITS_PATH", Path.Combine(ModelsDir, "Vision", "DepthAnything", "depth_anything_v2_vits.pth"));
        public static string VitL => Resolve("DEPTH_ANYTHING_V2_VITL_PATH", Path.Combine(ModelsDir, "Vision", "DepthAnything", "depth_anything_v2_vitl.pth"));
    }

    /// <summary>ControlNet annotator checkpoints (lllyasviel/Annotators, non-gated).</summary>
    public static class Annotators
    {
        public static string Hed => Resolve("ANNOTATOR_HED_PATH", Path.Combine(ModelsDir, "Vision", "Annotators", "ControlNetHED.pth"));
        public static string LineartRealistic => Resolve("ANNOTATOR_LINEART_PATH", Path.Combine(ModelsDir, "Vision", "Annotators", "sk_model.pth"));
        public static string LineartCoarse => Resolve("ANNOTATOR_LINEART_COARSE_PATH", Path.Combine(ModelsDir, "Vision", "Annotators", "sk_model2.pth"));
        public static string NormalBae => Resolve("ANNOTATOR_NORMALBAE_PATH", Path.Combine(ModelsDir, "Vision", "Annotators", "scannet.pt"));
        public static string UperNetSeg => Resolve("ANNOTATOR_UPERNET_SEG_PATH", Path.Combine(ModelsDir, "Vision", "Annotators", "upernet_convnext_small.bin"));
    }

    /// <summary>YOLO checkpoint paths (Vision package). Use the Python conversion script under
    /// <c>tests/python-reference/convert_yolov8_pt_to_safetensors.py</c> to produce these from
    /// Ultralytics <c>.pt</c> files — the conversion folds BN into Conv weights for fast inference.</summary>
    public static class Yolo
    {
        public static string V8nFolded => Resolve("YOLOV8N_FOLDED_PATH", Path.Combine(ModelsDir, "yolo", "yolov8n-folded.safetensors"));
        public static string V8sFolded => Resolve("YOLOV8S_FOLDED_PATH", Path.Combine(ModelsDir, "yolo", "yolov8s-folded.safetensors"));
        public static string V8mFolded => Resolve("YOLOV8M_FOLDED_PATH", Path.Combine(ModelsDir, "yolo", "yolov8m-folded.safetensors"));
        public static string V11nFolded => Resolve("YOLO11N_FOLDED_PATH", Path.Combine(ModelsDir, "yolo", "yolo11n-folded.safetensors"));
        public static string V11sFolded => Resolve("YOLO11S_FOLDED_PATH", Path.Combine(ModelsDir, "yolo", "yolo11s-folded.safetensors"));
        public static string V11mFolded => Resolve("YOLO11M_FOLDED_PATH", Path.Combine(ModelsDir, "yolo", "yolo11m-folded.safetensors"));
        public static string V8nSegFolded => Resolve("YOLOV8N_SEG_FOLDED_PATH", Path.Combine(ModelsDir, "yolo", "yolov8n-seg-folded.safetensors"));
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
        /// <summary>umT5 multilingual SentencePiece (256k vocab, from <c>google/umt5-xxl</c>). Used by Wan-Video.</summary>
        public static string Umt5XxlSpiece => Resolve("UMT5_XXL_SPIECE_PATH", Path.Combine(ModelsDir, "Tokenizers", "T5", "umt5_xxl_spiece.model"));
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

    /// <summary>IP-Adapter checkpoints + the CLIP-Vision-H image encoder they were trained against. Assets are not bundled — tests skip when missing.</summary>
    public static class IpAdapter
    {
        public static string SdxlStandard => Resolve("IPA_SDXL_PATH",       Path.Combine(ModelsDir, "ipadapter", "ip-adapter_sdxl_vit-h.safetensors"));
        public static string SdxlPlus     => Resolve("IPA_SDXL_PLUS_PATH",  Path.Combine(ModelsDir, "ipadapter", "ip-adapter-plus_sdxl_vit-h.safetensors"));
        public static string Sd15Standard => Resolve("IPA_SD15_PATH",       Path.Combine(ModelsDir, "ipadapter", "ip-adapter_sd15.safetensors"));
        public static string ClipVisionH  => Resolve("CLIP_VISION_H_PATH",  Path.Combine(ModelsDir, "clip_vision", "clip-vision-h-14.safetensors"));

        /// <summary>FaceID checkpoint (h94/IP-Adapter-FaceID torch-pickle .bin) + its companion kohya UNet LoRA.</summary>
        public static string SdxlFaceId     => Resolve("IPA_SDXL_FACEID_PATH",      Path.Combine(ModelsDir, "ipadapter", "ip-adapter-faceid_sdxl.bin"));
        public static string SdxlFaceIdLora => Resolve("IPA_SDXL_FACEID_LORA_PATH", Path.Combine(ModelsDir, "ipadapter", "ip-adapter-faceid_sdxl_lora.safetensors"));

        /// <summary>ArcFace IR-50 face-embedding weights (buffalo_l w600k_r50 converted via convert_arcface_onnx.py).</summary>
        public static string ArcFace        => Resolve("ARCFACE_WEIGHTS_PATH",      Path.Combine(ModelsDir, "ipadapter", "arcface_w600k_r50.safetensors"));

        /// <summary>Folded YOLO11n-pose weights used for face keypoint detection (convert_yolov8_pt_to_safetensors.py).</summary>
        public static string PoseWeights    => Resolve("FACEID_POSE_PATH",          Path.Combine(ModelsDir, "clip", "yolo11n-pose-folded.safetensors"));

        /// <summary>A clean single-person portrait photo (PNG) used as the FaceID identity reference.</summary>
        public static string PortraitImage  => Resolve("FACEID_PORTRAIT_PATH",      Path.Combine(OutputDir, "References", "portrait.png"));
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
