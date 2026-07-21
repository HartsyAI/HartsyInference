namespace HartsyInference.Cli.Infra;

/// <summary>Static, data-driven catalog of the models the engine can drive, mirroring the README support table.
/// Backs <c>hartsy list</c>, REPL model menus, and shell completion.</summary>
public static class ModelCatalog
{
    private static readonly List<CatalogEntry> Entries = Build();

    /// <summary>Every catalogued model in display order.</summary>
    public static IReadOnlyList<CatalogEntry> All => Entries;

    /// <summary>All models for one modality, in catalog order.</summary>
    public static IReadOnlyList<CatalogEntry> ForModality(Modality modality) =>
        Entries.Where(e => e.Modality == modality).ToList();

    /// <summary>Looks up a model by its CLI id (case-insensitive), or null when unknown.</summary>
    public static CatalogEntry? Find(string id) =>
        Entries.FirstOrDefault(e => string.Equals(e.Id, id, StringComparison.OrdinalIgnoreCase));

    private static CatalogEntry E(string id, Modality modality, string name, string arch, ModelStatus status, bool cli = false) =>
        new CatalogEntry { Id = id, Modality = modality, DisplayName = name, Architecture = arch, Status = status, CliDrivable = cli };

    private static List<CatalogEntry> Build()
    {
        const Modality img = Modality.Image;
        const Modality txt = Modality.Text;
        const Modality tts = Modality.Speech;
        const Modality mus = Modality.Music;
        const Modality stt = Modality.Transcribe;
        const Modality vis = Modality.Vision;
        const Modality vid = Modality.Video;
        const Modality d3 = Modality.Mesh;
        const Modality act = Modality.World;
        const ModelStatus ok = ModelStatus.Verified;
        const ModelStatus vp = ModelStatus.ValidationPending;
        const ModelStatus st = ModelStatus.Structural;

        return new List<CatalogEntry>
        {
            // Text / LLM
            E("qwen2", txt, "Qwen2.5 (0.5B → 7B)", "Qwen2 dense transformer", ok, cli: true),
            E("qwen3", txt, "Qwen3 (0.6B → 7B)", "Qwen3 dense transformer", ok, cli: true),
            E("llama3", txt, "Llama-3.x", "Llama dense transformer", st, cli: true),
            E("mistral", txt, "Mistral (dense)", "Mistral dense transformer", st, cli: true),
            E("gguf", txt, "Quantized GGUF (Q4/Q8)", "config-driven, any GGUF LLM", ok, cli: true),

            // Image / diffusion
            E("sd15", img, "Stable Diffusion 1.5", "UNet", ok),
            E("sdxl", img, "SDXL", "UNet (dual CLIP)", ok, cli: true),
            // No "sdxl-refiner" / "sdxl-inpaint" entries: neither is a standalone text-to-image family, so neither can
            // have an IArchitectureRecipe. The refiner is reachable as ImageRequest.Refiner on the sdxl entry (loaded by
            // Features/SdxlRefinerLoader), and inpainting as ImageRequest.Inpaint — listing them as selectable models
            // only produced a "no recipe lifted" throw.
            E("flux1", img, "Flux.1-dev", "single-stream DiT, flow-matching", ok),
            E("flux2", img, "Flux.2", "single-stream DiT, flow-matching", ok),
            E("chroma", img, "Chroma", "Flux-derivative DiT", ok),
            E("chroma-radiance", img, "Chroma Radiance", "Flux-derivative DiT", ok),
            E("sd3", img, "Stable Diffusion 3", "MMDiT (3 text encoders)", ok),
            E("qwen-image", img, "Qwen-Image", "MMDiT (Qwen2.5-VL)", ok),
            E("hunyuan-image", img, "Hunyuan Image 2.1", "17B MMDiT", ok),
            E("hidream", img, "HiDream i1", "MMDiT (quad encoder + MoE)", ok),
            E("auraflow", img, "AuraFlow", "MMDiT + single-DiT hybrid (Pile-T5-XL)", ok),
            E("lumina2", img, "Lumina 2.0", "NextDiT (Gemma-2)", ok),
            E("ernie-image", img, "ERNIE-Image", "single-stream DiT (Ministral-3B)", ok),
            E("kandinsky5", img, "Kandinsky 5", "DiT (Qwen2.5-VL + CLIP)", ok),
            E("omnigen2", img, "OmniGen 2", "MLLM-based DiT", ok),
            E("ideogram4", img, "Ideogram 4", "9.3B single-stream DiT", ok),
            E("f-lite", img, "F-Lite", "DiT (Qwen)", vp),
            E("lance-image", img, "Lance (Image)", "unified multimodal DiT", vp),
            E("zimage", img, "Z-Image Turbo", "NextDiT (Qwen3-4B)", ok),
            E("anima", img, "Anima", "Cosmos-Predict2-2B (T=1)", ok),
            E("zeta-chroma", img, "Zeta-Chroma", "Chroma-derivative DiT (Qwen3-4B)", ok),
            E("boogu", img, "Boogu Image", "single-stream DiT (Qwen3-VL-8B + Flux VAE)", ok),
            E("lens", img, "Lens · Lens-Turbo", "48-layer MoE DiT (Microsoft Lens)", ok),
            new CatalogEntry
            {
                Id = "krea2",
                Modality = img,
                DisplayName = "Krea 2 Turbo",
                Architecture = "Krea2 DiT (Qwen3-VL-4B + Qwen-Image VAE)",
                Status = ok,
                CliDrivable = false, // real-weight verified via --model-path (see MODEL_STATUS_IMAGE.md); the
                // catalog-slug path (`-m krea2`, this Assets-driven auto-download) has not been exercised end-to-end yet
                // Side models come straight from SideModels — the same SHA-256-pinned entries Krea2Recipe
                // downloads — so the catalog can never disagree with what the engine actually loads. Only the
                // transformer is spelled out here: it is the checkpoint itself and has no SideModels entry.
                Assets = new ModelAsset[]
                {
                    new() { Repo = "Comfy-Org/Krea-2", RepoPath = "diffusion_models/krea2_turbo_fp8_scaled.safetensors", TargetSubdir = "Stable-Diffusion/Krea2", Role = "transformer" },
                    SideModels.Qwen3VL_4B,
                    SideModels.QwenImageVae,
                },
            },

            // Transcription
            E("whisper", stt, "Whisper (tiny → large-v3)", "encoder-decoder", ok, cli: true),
            E("moonshine", stt, "Moonshine", "encoder-decoder", ok),

            // Text-to-speech
            E("piper", tts, "Piper (en_US-lessac-medium, …)", "VITS + espeak phonemes", ok, cli: true),
            E("kokoro", tts, "Kokoro-82M", "StyleTTS-family vocoder", ok),
            E("bark", tts, "Bark", "GPT-style TTS", ok),
            E("styletts2", tts, "StyleTTS2", "style-diffusion TTS", ok),
            E("spark-tts", tts, "Spark-TTS", "BiCodec LM", ok),
            E("cosyvoice", tts, "CosyVoice", "Qwen LM + flow", ok),
            E("vibevoice", tts, "VibeVoice", "diffusion TTS", ok),
            E("fish-speech", tts, "Fish-Speech / OpenAudio", "DualAR + tiktoken", vp),
            E("f5-tts", tts, "F5-TTS", "voice cloning, flow-matching DiT", vp),

            // Music / audio generation
            E("musicgen", mus, "MusicGen", "transformer + EnCodec", ok, cli: true),
            E("audiogen", mus, "AudioGen", "MusicGen-arch + T5", vp),
            E("ace-step", mus, "ACE-Step", "flow-matching DiT", ok),
            E("yue", mus, "YuE", "dual-stage Llama", ok),
            E("stable-audio", mus, "Stable Audio Open", "latent diffusion", st),

            // Vision
            E("clip", vis, "CLIP (ViT-L/14, H/14, bigG/14)", "ViT embeddings", ok, cli: true),
            E("siglip", vis, "SigLIP · SigLIP2", "ViT embeddings", ok),
            E("dinov2", vis, "DINOv2 · DINOv3", "dense features", ok),
            E("yolo8", vis, "YOLO8 (n → xl)", "object detection", ok, cli: true),
            E("yolo11", vis, "YOLO11 (n → xl)", "object detection", ok, cli: true),
            E("sam", vis, "SAM · SAM 2 · SAM 2.1", "segmentation", ok),
            E("retinaface", vis, "RetinaFace", "face detection + landmarks", ok),

            // Video
            E("ltx-video", vid, "LTX-Video", "DiT + video VAE", vp, cli: true),
            E("wan", vid, "Wan 2.2 (T2V + I2V)", "DiT + Wan VAE", vp),
            E("lance-video", vid, "Lance (Video, T2V)", "unified multimodal DiT", vp),
            E("kandinsky5-video", vid, "Kandinsky 5 Video", "DiT", vp),
            // Cosmos-Predict1 Video2World — discrete-token autoregressive video continuation (T5-11B cross-attn +
            // DV8x16x16 tokenizer + AR backbone). Engine-only; run via the sample invocation in VideoCommand help.
            E("cosmos-predict1-5b-v2w", vid, "Cosmos-Predict1 5B Video2World", "AR discrete-token transformer", vp),
            E("cosmos-predict1-13b-v2w", vid, "Cosmos-Predict1 13B Video2World", "AR discrete-token transformer", vp),

            // 3D
            E("triposr", d3, "TripoSR", "triplane / NeRF", st, cli: true),
            E("hunyuan3d", d3, "Hunyuan3D-2 (Shape)", "Flux MMDiT + VecSet VAE", st, cli: true),

            // Interactive / world models
            E("hunyuan-gamecraft", act, "Hunyuan-GameCraft 1.0", "HunyuanVideo MM-DiT", vp),
            E("matrix-game-3", act, "Matrix-Game 3.0", "5B (+28B MoE) memory-augmented", vp),
            E("matrix-game-2", act, "Matrix-Game 2.0", "Wan2.1-lineage 1.8B", vp),
            E("oasis", act, "Oasis-500m", "axial-attention DiT", vp, cli: true),
        };
    }
}
