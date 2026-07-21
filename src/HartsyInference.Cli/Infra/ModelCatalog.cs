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
            new CatalogEntry
            {
                Id = "sd15", Modality = img, DisplayName = "Stable Diffusion 1.5", Architecture = "UNet", Status = ok,
                CliDrivable = true, // `hartsy image -m sd15` verified end-to-end 2026-07-20
                Assets = new ModelAsset[]
                {
                    new() { Repo = "Comfy-Org/stable-diffusion-v1-5-archive", RepoPath = "v1-5-pruned-emaonly-fp16.safetensors",
                        TargetSubdir = "Stable-Diffusion/SD15", Role = "transformer",
                        Sha256 = "e9476a13728cd75d8279f6ec8bad753a66a1957ca375a1464dc63b37db6e3916" },
                },
            },
            E("sdxl", img, "SDXL", "UNet (dual CLIP)", ok, cli: true),
            // No "sdxl-refiner" / "sdxl-inpaint" entries: neither is a standalone text-to-image family, so neither can
            // have an IArchitectureRecipe. The refiner is reachable as ImageRequest.Refiner on the sdxl entry (loaded by
            // Features/SdxlRefinerLoader), and inpainting as ImageRequest.Inpaint — listing them as selectable models
            // only produced a "no recipe lifted" throw.
            new CatalogEntry
            {
                Id = "flux1", Modality = img, DisplayName = "Flux.1-dev", Architecture = "single-stream DiT, flow-matching", Status = ok,
                CliDrivable = true, // `hartsy image -m flux1` verified end-to-end 2026-07-20
                Assets = new ModelAsset[]
                {
                    // All-in-one fp8 (bundles CLIP-L/T5-XXL/VAE); FluxCheckpointConverter resolves any component it omits from SideModels.
                    new() { Repo = "Comfy-Org/flux1-dev", RepoPath = "flux1-dev-fp8.safetensors",
                        TargetSubdir = "Stable-Diffusion/Flux", Role = "transformer",
                        Sha256 = "8e91b68084b53a7fc44ed2a3756d821e355ac1a7b6fe29be760c1db532f3d88a" },
                },
            },
            new CatalogEntry
            {
                Id = "flux2", Modality = img, DisplayName = "Flux.2", Architecture = "single-stream DiT, flow-matching", Status = ok,
                CliDrivable = true, // `hartsy image -m flux2` (Dev, Q4_K_S GGUF) verified end-to-end 2026-07-21
                // after adding GGUF support to Flux2Recipe/Flux2RecipePipeline (previously safetensors-only
                // despite an existing, unused Flux2KeyMapper) — see MODEL_STATUS_IMAGE.md
                Assets = new ModelAsset[]
                {
                    // Dev variant, Q4_K_S GGUF (19.3GB vs the 35.5GB fp8mixed all-in-one safetensors).
                    new() { Repo = "city96/FLUX.2-dev-gguf", RepoPath = "flux2-dev-Q4_K_S.gguf",
                        TargetSubdir = "Stable-Diffusion/Flux2", Role = "transformer",
                        Sha256 = "b9c1c8295ed044f54c3a9894a800e003b4ecc94bfb0f63192b68bafc232c2b27" },
                    SideModels.MistralSmallFlux2,
                    SideModels.Flux2Vae,
                },
            },
            new CatalogEntry
            {
                Id = "chroma", Modality = img, DisplayName = "Chroma", Architecture = "Flux-derivative DiT", Status = ok,
                CliDrivable = true, // `hartsy image -m chroma` verified end-to-end 2026-07-21
                Assets = new ModelAsset[]
                {
                    new() { Repo = "silveroxides/Chroma1-HD-fp8-scaled", RepoPath = "Chroma1-HD-fp8mixed-final.safetensors",
                        TargetSubdir = "Stable-Diffusion/Chroma", Role = "transformer",
                        Sha256 = "c25284ddf89df1678ac4dd7bd8a59ec68fb2e484296d75977c549d58618eee91" },
                    SideModels.T5XxlEnconly,
                    SideModels.FluxAe,
                },
            },
            new CatalogEntry
            {
                Id = "chroma-radiance", Modality = img, DisplayName = "Chroma Radiance", Architecture = "Flux-derivative DiT", Status = ok,
                Assets = new ModelAsset[]
                {
                    // lodestones/Chroma-Radiance is gated; this is Comfy-Org's ungated repack of the same weights.
                    new() { Repo = "Comfy-Org/Chroma1-Radiance_Repackaged", RepoPath = "split_files/diffusion_models/chroma-radiance-x0.safetensors",
                        TargetSubdir = "Stable-Diffusion/ChromaRadiance", Role = "transformer",
                        Sha256 = "086e11d033ccd7470e67fa80e00a29902df2868cc84e16df0b48853be3a8672a" },
                    SideModels.T5XxlEnconly,
                    // No VAE — Chroma Radiance is pixel-space (NeRF-style decoder head baked into the transformer).
                },
            },
            new CatalogEntry
            {
                Id = "sd3", Modality = img, DisplayName = "Stable Diffusion 3", Architecture = "MMDiT (3 text encoders)", Status = ok,
                CliDrivable = true, // `hartsy image -m sd3` verified end-to-end 2026-07-20
                Assets = new ModelAsset[]
                {
                    // SD3.5 Medium, bundles CLIP-L/CLIP-G/T5-XXL/VAE all-in-one.
                    new() { Repo = "Comfy-Org/stable-diffusion-3.5-fp8", RepoPath = "sd3.5_medium_incl_clips_t5xxlfp8scaled.safetensors",
                        TargetSubdir = "Stable-Diffusion/SD3", Role = "transformer",
                        Sha256 = "1778e8857679042c176c21cd8a0da7b29bded68be018557477f84419df79bacf" },
                },
            },
            new CatalogEntry
            {
                Id = "qwen-image", Modality = img, DisplayName = "Qwen-Image", Architecture = "MMDiT (Qwen2.5-VL)", Status = ok,
                CliDrivable = true, // `hartsy image -m qwen-image` verified end-to-end 2026-07-21 (QuantStack
                // Q4_K_M GGUF; host-side load/convert of a 20B checkpoint is slow on this box — budget 15 min)
                Assets = new ModelAsset[]
                {
                    // Q4_K_M GGUF — fits 24GB where the ~40GB fp8 single-file does not.
                    new() { Repo = "QuantStack/Qwen-Image-GGUF", RepoPath = "Qwen_Image-Q4_K_M.gguf",
                        TargetSubdir = "Stable-Diffusion/QwenImage", Role = "transformer",
                        Sha256 = "645473886d7dbb0103f84c563c798f7b0867293d919752d4d6be6a432b0bc988" },
                    SideModels.Qwen2_5_VL_7B,
                    SideModels.QwenImageVae,
                },
            },
            new CatalogEntry
            {
                Id = "hunyuan-image", Modality = img, DisplayName = "Hunyuan Image 2.1", Architecture = "17B MMDiT", Status = ok,
                // CliDrivable intentionally left false: `hartsy image -m hunyuan-image` (QuantStack Q4_K_M GGUF)
                // downloads + runs to completion but produces an all-zero (black) image — open bug, not yet
                // root-caused (fixed one contributing latent bug in GgufKeyMapperRegistry ordering, which did
                // NOT resolve it — see MODEL_STATUS_IMAGE.md). 2048x2048 also separately OOMs at VAE decode.
                Assets = new ModelAsset[]
                {
                    new() { Repo = "QuantStack/HunyuanImage-2.1-GGUF", RepoPath = "HunyuanImage2.1-Q4_K_M.gguf",
                        TargetSubdir = "Stable-Diffusion/HunyuanImage", Role = "transformer",
                        Sha256 = "583800deb18fa90560d15b41ca0391e79c024784117b0b0263ace02e941237e1" },
                    SideModels.Qwen25Vl7BHunyuan,
                    SideModels.HunyuanImageVae,
                },
            },
            new CatalogEntry
            {
                Id = "hidream", Modality = img, DisplayName = "HiDream i1", Architecture = "MMDiT (quad encoder + MoE)", Status = ok,
                CliDrivable = true, // `hartsy image -m hidream` verified end-to-end 2026-07-20
                Assets = new ModelAsset[]
                {
                    new() { Repo = "Comfy-Org/HiDream-I1_ComfyUI", RepoPath = "split_files/diffusion_models/hidream_i1_dev_fp8.safetensors",
                        TargetSubdir = "Stable-Diffusion/HiDream", Role = "transformer",
                        Sha256 = "9a372d7384d56e34a8cc7fd77a0fa3d26d6b75d82c7582fd5347e2fd9e6f8664" },
                    SideModels.HiDreamClipL,
                    SideModels.HiDreamClipG,
                    SideModels.T5XxlEnconly,
                    SideModels.Llama31_8B,
                    SideModels.FluxAe,
                },
            },
            new CatalogEntry
            {
                Id = "auraflow", Modality = img, DisplayName = "AuraFlow", Architecture = "MMDiT + single-DiT hybrid (Pile-T5-XL)", Status = ok,
                CliDrivable = true, // `hartsy image -m auraflow` verified end-to-end 2026-07-20/21 after fixing
                // AuraFlowRecipe to use SideModels.PileT5XlSpiece (was silently using the wrong embedded
                // Google-T5 vocab — see MODEL_STATUS_IMAGE.md)
                Assets = new ModelAsset[]
                {
                    // fp8_scaled bundles the transformer + Pile-T5-XL text encoder + AuraFlow VAE in one file.
                    new() { Repo = "calcuis/aura", RepoPath = "aura_flow_0.3_fp8_scaled.safetensors",
                        TargetSubdir = "Stable-Diffusion/AuraFlow", Role = "transformer",
                        Sha256 = "1870f8fb113db07f7b43bef85403fd7c7b57c1302603fe9e0bc1d48c3a7936fe" },
                    SideModels.PileT5XlSpiece,
                },
            },
            new CatalogEntry
            {
                // The original AlphaVLLM single-file uses fused attention.qkv keys Lumina2Transformer.LoadWeights
                // doesn't accept (needs split attn.to_q/k/v); the raw diffusers folder is 2 shards with no
                // multi-shard merge support in Lumina2CheckpointConverter (single-file loader only). Use
                // Comfy-Org's single-file bf16 repack instead — same split attn.to_q/k/v keys, one file.
                // Must be the REAL diffusers-format weights (Alpha-VLLM/Lumina-Image-2.0's transformer/ shards) —
                // Lumina2Transformer.LoadWeights needs split attn.to_q/k/v + time_caption_embed.* keys.
                // Comfy-Org's single-file repack (lumina_2_model_bf16.safetensors) ships the ORIGINAL AlphaVLLM
                // naming (t_embedder.*, no time_caption_embed prefix) despite this engine's own prior docstring
                // claiming otherwise — verified 2026-07-21 by inspecting its safetensors header directly; do not
                // reuse it here. Lumina2CheckpointConverter.LoadAndConvert now merges multi-shard diffusers
                // folders (detects a sibling *.safetensors.index.json), so the 2 raw shards work directly.
                Id = "lumina2", Modality = img, DisplayName = "Lumina 2.0", Architecture = "NextDiT (Gemma-2)", Status = ok,
                CliDrivable = true, // `hartsy image -m lumina2` verified end-to-end 2026-07-21 (after fixing the
                // wrong-repack pick + adding multi-shard merge support — see MODEL_STATUS_IMAGE.md)
                Assets = new ModelAsset[]
                {
                    new() { Repo = "Alpha-VLLM/Lumina-Image-2.0", RepoPath = "transformer/config.json",
                        TargetSubdir = "Stable-Diffusion/Lumina2/transformer", Role = "config" },
                    new() { Repo = "Alpha-VLLM/Lumina-Image-2.0", RepoPath = "transformer/diffusion_pytorch_model.safetensors.index.json",
                        TargetSubdir = "Stable-Diffusion/Lumina2/transformer", Role = "config" },
                    new() { Repo = "Alpha-VLLM/Lumina-Image-2.0", RepoPath = "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
                        TargetSubdir = "Stable-Diffusion/Lumina2/transformer", Role = "transformer",
                        Sha256 = "132b4d213fdd3cfc14333746fc3eb8bbe6358cd73c3bc95ac4ccec230b97dca3" },
                    new() { Repo = "Alpha-VLLM/Lumina-Image-2.0", RepoPath = "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
                        TargetSubdir = "Stable-Diffusion/Lumina2/transformer", Role = "transformer",
                        Sha256 = "b9660a895cdedf3a023b131b9c655c32f7d0cd824d6b3f91f0810812bb592947" },
                    SideModels.Gemma2_2B,
                    SideModels.FluxAe,
                },
            },
            new CatalogEntry
            {
                Id = "ernie-image", Modality = img, DisplayName = "ERNIE-Image", Architecture = "single-stream DiT (Ministral-3B)", Status = ok,
                CliDrivable = true, // `hartsy image -m ernie-image` verified end-to-end 2026-07-21
                Assets = new ModelAsset[]
                {
                    new() { Repo = "Comfy-Org/ERNIE-Image", RepoPath = "diffusion_models/ernie-image.safetensors",
                        TargetSubdir = "Stable-Diffusion/ErnieImage", Role = "transformer",
                        Sha256 = "94a35abaa0899cccc34d2e37310abf74a0a714256526117bba782c7eb4eb91c7" },
                    SideModels.Ministral_3_3B,
                    SideModels.Flux2Vae,
                },
            },
            new CatalogEntry
            {
                Id = "kandinsky5", Modality = img, DisplayName = "Kandinsky 5", Architecture = "DiT (Qwen2.5-VL + CLIP)", Status = ok,
                CliDrivable = true, // `hartsy image -m kandinsky5` verified end-to-end 2026-07-21
                Assets = new ModelAsset[]
                {
                    new() { Repo = "kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers", RepoPath = "transformer/diffusion_pytorch_model.safetensors",
                        TargetSubdir = "Stable-Diffusion/Kandinsky5Lite", TargetName = "kandinsky5_t2i_lite_transformer.safetensors", Role = "transformer",
                        Sha256 = "be7a9e1f002f3aded7bed95dfe664a762e66ce766876f137444aab13ba48b63f" },
                    SideModels.Qwen2_5_VL_7B,
                    SideModels.ClipL,
                    SideModels.FluxAe,
                },
            },
            new CatalogEntry
            {
                Id = "omnigen2", Modality = img, DisplayName = "OmniGen 2", Architecture = "MLLM-based DiT", Status = ok,
                CliDrivable = true, // `hartsy image -m omnigen2` verified end-to-end 2026-07-21
                Assets = new ModelAsset[]
                {
                    new() { Repo = "Comfy-Org/Omnigen2_ComfyUI_repackaged", RepoPath = "split_files/diffusion_models/omnigen2_fp16.safetensors",
                        TargetSubdir = "Stable-Diffusion/OmniGen2", Role = "transformer",
                        Sha256 = "60dbde45107762d164bac463e1cf365e074b377fa843dc90cb2985fb211cd4de" },
                    SideModels.Qwen2_5_VL_3B,
                },
            },
            new CatalogEntry
            {
                Id = "ideogram4", Modality = img, DisplayName = "Ideogram 4", Architecture = "9.3B single-stream DiT", Status = ok,
                CliDrivable = true, // `hartsy image -m ideogram4` verified end-to-end 2026-07-20 (needs a
                // structured-JSON caption as the prompt argument, e.g. {"high_level_description":"...",...} — a
                // plain prompt is fed verbatim and produces off-prompt/watermark-hallucination output
                Assets = new ModelAsset[]
                {
                    new() { Repo = "Comfy-Org/Ideogram-4", RepoPath = "diffusion_models/ideogram4_fp8_scaled.safetensors",
                        TargetSubdir = "Stable-Diffusion/Ideogram4", Role = "transformer",
                        Sha256 = "49a946f1b0f8bcf5eab7d3b1ecc7b453c104e034cb1b592032745692724bd306" },
                    SideModels.Qwen3VL_8B,
                    SideModels.Ideogram4Unconditional,
                    SideModels.Flux2Vae,
                },
            },
            new CatalogEntry
            {
                // F-Lite ships as a diffusers folder (no single-file release) — every file is listed so the
                // full folder reconstitutes on disk; FLiteRecipe's ResolveFolderRoot walks up from any file
                // inside dit_model/text_encoder/vae to find the shared root.
                Id = "f-lite", Modality = img, DisplayName = "F-Lite", Architecture = "DiT (Qwen)", Status = ok,
                CliDrivable = true, // `hartsy image -m f-lite --model-path <dir>` verified end-to-end 2026-07-21
                // (already-local folder; the folder-root resolution itself needs no catalog-Assets exercise
                // beyond confirming the file list below matches Freepik/F-Lite's real layout byte-for-byte)
                Assets = new ModelAsset[]
                {
                    new() { Repo = "Freepik/F-Lite", RepoPath = "model_index.json", TargetSubdir = "Stable-Diffusion/F-Lite", Role = "config" },
                    new() { Repo = "Freepik/F-Lite", RepoPath = "dit_model/config.json", TargetSubdir = "Stable-Diffusion/F-Lite/dit_model", Role = "config" },
                    new() { Repo = "Freepik/F-Lite", RepoPath = "dit_model/diffusion_pytorch_model-00001-of-00002.safetensors", TargetSubdir = "Stable-Diffusion/F-Lite/dit_model", Role = "transformer" },
                    new() { Repo = "Freepik/F-Lite", RepoPath = "dit_model/diffusion_pytorch_model-00002-of-00002.safetensors", TargetSubdir = "Stable-Diffusion/F-Lite/dit_model", Role = "transformer" },
                    new() { Repo = "Freepik/F-Lite", RepoPath = "dit_model/diffusion_pytorch_model.safetensors.index.json", TargetSubdir = "Stable-Diffusion/F-Lite/dit_model", Role = "config" },
                    new() { Repo = "Freepik/F-Lite", RepoPath = "text_encoder/config.json", TargetSubdir = "Stable-Diffusion/F-Lite/text_encoder", Role = "text encoder" },
                    new() { Repo = "Freepik/F-Lite", RepoPath = "text_encoder/model-00001-of-00002.safetensors", TargetSubdir = "Stable-Diffusion/F-Lite/text_encoder", Role = "text encoder" },
                    new() { Repo = "Freepik/F-Lite", RepoPath = "text_encoder/model-00002-of-00002.safetensors", TargetSubdir = "Stable-Diffusion/F-Lite/text_encoder", Role = "text encoder" },
                    new() { Repo = "Freepik/F-Lite", RepoPath = "text_encoder/model.safetensors.index.json", TargetSubdir = "Stable-Diffusion/F-Lite/text_encoder", Role = "config" },
                    new() { Repo = "Freepik/F-Lite", RepoPath = "tokenizer/special_tokens_map.json", TargetSubdir = "Stable-Diffusion/F-Lite/tokenizer", Role = "config" },
                    new() { Repo = "Freepik/F-Lite", RepoPath = "tokenizer/spiece.model", TargetSubdir = "Stable-Diffusion/F-Lite/tokenizer", Role = "tokenizer" },
                    new() { Repo = "Freepik/F-Lite", RepoPath = "tokenizer/tokenizer.json", TargetSubdir = "Stable-Diffusion/F-Lite/tokenizer", Role = "tokenizer" },
                    new() { Repo = "Freepik/F-Lite", RepoPath = "tokenizer/tokenizer_config.json", TargetSubdir = "Stable-Diffusion/F-Lite/tokenizer", Role = "config" },
                    new() { Repo = "Freepik/F-Lite", RepoPath = "vae/config.json", TargetSubdir = "Stable-Diffusion/F-Lite/vae", Role = "config" },
                    new() { Repo = "Freepik/F-Lite", RepoPath = "vae/diffusion_pytorch_model.safetensors", TargetSubdir = "Stable-Diffusion/F-Lite/vae", Role = "vae" },
                },
            },
            new CatalogEntry
            {
                // Ships as a folder (model.safetensors + llm_config.json + Qwen2 tokenizer files) — every file
                // listed so the full folder reconstitutes; LanceImageRecipe.ResolveLanceFolder walks up from
                // any file inside to find the shared root.
                Id = "lance-image", Modality = img, DisplayName = "Lance (Image)", Architecture = "unified multimodal DiT", Status = ok, // was stale `vp`; MODEL_STATUS_IMAGE.md already records ✅ backbone parity
                CliDrivable = true, // `hartsy image -m lance-image` verified end-to-end 2026-07-21
                Assets = new ModelAsset[]
                {
                    new() { Repo = "bytedance-research/Lance", RepoPath = "Lance_3B/model.safetensors",
                        TargetSubdir = "Stable-Diffusion/Lance/Lance_3B", Role = "transformer",
                        Sha256 = "a2cfed3992486699aa550c1ea9b3519bd19dde475a0992daf2249f2486b268a3" },
                    new() { Repo = "bytedance-research/Lance", RepoPath = "Lance_3B/llm_config.json", TargetSubdir = "Stable-Diffusion/Lance/Lance_3B", Role = "config" },
                    new() { Repo = "bytedance-research/Lance", RepoPath = "Lance_3B/generation_config.json", TargetSubdir = "Stable-Diffusion/Lance/Lance_3B", Role = "config" },
                    new() { Repo = "bytedance-research/Lance", RepoPath = "Lance_3B/tokenizer.json", TargetSubdir = "Stable-Diffusion/Lance/Lance_3B", Role = "tokenizer" },
                    new() { Repo = "bytedance-research/Lance", RepoPath = "Lance_3B/vocab.json", TargetSubdir = "Stable-Diffusion/Lance/Lance_3B", Role = "tokenizer" },
                    new() { Repo = "bytedance-research/Lance", RepoPath = "Lance_3B/merges.txt", TargetSubdir = "Stable-Diffusion/Lance/Lance_3B", Role = "tokenizer" },
                    SideModels.Wan22Vae,
                },
            },
            new CatalogEntry
            {
                Id = "zimage", Modality = img, DisplayName = "Z-Image Turbo", Architecture = "NextDiT (Qwen3-4B)", Status = ok,
                CliDrivable = true, // `hartsy image -m zimage` verified end-to-end 2026-07-21
                Assets = new ModelAsset[]
                {
                    new() { Repo = "mcmonkey/swarm-models", RepoPath = "SwarmUI_Z-Image-Turbo-FP8Mix.safetensors",
                        TargetSubdir = "Stable-Diffusion/ZImage", Role = "transformer",
                        Sha256 = "ba92d3705131c8d9b05ca9c6fefe39444d4eb02db16c30aafa9fcf5f85230e06" },
                    SideModels.Qwen3_4B,
                    SideModels.FluxAe,
                },
            },
            new CatalogEntry
            {
                Id = "anima", Modality = img, DisplayName = "Anima", Architecture = "Cosmos-Predict2-2B (T=1)", Status = ok,
                CliDrivable = true, // `hartsy image -m anima --width 512 --height 512` verified end-to-end
                // 2026-07-21 (coherent anime-style output). NOTE: --width 1024 --height 1024 (the recipe's
                // own FamilyDefaults!) hung with zero denoise progress after 5 minutes — likely a real perf/
                // scaling gap at 1024, not investigated further here (out of scope for CLI/catalog wiring;
                // every prior verified run in MODEL_STATUS_IMAGE.md also used 512). Prefer 512 for now.
                Assets = new ModelAsset[]
                {
                    new() { Repo = "circlestone-labs/Anima", RepoPath = "split_files/diffusion_models/anima-base-v1.0.safetensors",
                        TargetSubdir = "Stable-Diffusion/Anima", Role = "transformer",
                        Sha256 = "bd43b7cffe1ed1153d9c41e7beb2f18cb1273eafbaa3af3edd6a173dc90a006e" },
                    SideModels.Qwen3_0_6B,
                    SideModels.QwenImageVae,
                },
            },
            new CatalogEntry
            {
                Id = "zeta-chroma", Modality = img, DisplayName = "Zeta-Chroma", Architecture = "Chroma-derivative DiT (Qwen3-4B)", Status = ok,
                CliDrivable = true, // `hartsy image -m zeta-chroma --width 768 --height 768` verified end-to-end
                // 2026-07-21. NOTE: --width 1024 --height 1024 (this recipe's own FamilyDefaults!) deterministically
                // VRAM-OOMs at step 49/50 on this 24GB GPU (same exact allocation size, twice) — pixel-space (no
                // VAE downsampling) activations are just too large at 1024 here. Prefer 768 on 24GB cards.
                Assets = new ModelAsset[]
                {
                    new() { Repo = "lodestones/Zeta-Chroma", RepoPath = "zeta-chroma-base-x0-pixel-no-dino.safetensors",
                        TargetSubdir = "Stable-Diffusion/ZetaChroma", Role = "transformer",
                        Sha256 = "f405438286b72d9541a761c7c6de987ab37fd76bb951f4171be0da318f54b914" },
                    SideModels.Qwen3_4B,
                    // No VAE — Zeta-Chroma is pixel-space like Chroma Radiance.
                },
            },
            new CatalogEntry
            {
                // Deliberately points at the EDIT checkpoint, not a "base" one: verified 2026-07-21 that the
                // edit transformer runs cleanly as plain T2I when no reference image is passed (same base
                // architecture, edit conditioning slots simply unused) — a genuine Base/Turbo fp8 file also
                // exists at this repo (boogu_image_base_fp8_scaled.safetensors) if a cleaner separation is
                // ever wanted, but this is what was actually verified end-to-end here.
                Id = "boogu", Modality = img, DisplayName = "Boogu Image", Architecture = "single-stream DiT (Qwen3-VL-8B + Flux VAE)", Status = ok,
                CliDrivable = true, // `hartsy image --model-path <edit-fp8-scaled> -m boogu` verified end-to-end
                // 2026-07-21; the local file's sha256 didn't byte-match this repo's current upload (~18MB size
                // delta — likely a re-save/version bump upstream), so Sha256 is left unpinned pending a fresh
                // download + re-verify rather than pin a hash that would reject a legitimate current download.
                Assets = new ModelAsset[]
                {
                    new() { Repo = "Comfy-Org/Boogu-Image", RepoPath = "diffusion_models/boogu_image_edit_fp8_scaled.safetensors",
                        TargetSubdir = "Stable-Diffusion/Boogu", Role = "transformer" },
                    SideModels.Qwen3VL_8B,
                    SideModels.FluxAe,
                },
            },
            new CatalogEntry
            {
                Id = "lens", Modality = img, DisplayName = "Lens · Lens-Turbo", Architecture = "48-layer MoE DiT (Microsoft Lens)", Status = ok,
                CliDrivable = true, // `hartsy image -m lens` verified end-to-end 2026-07-21 (Turbo, 4 steps)
                Assets = new ModelAsset[]
                {
                    new() { Repo = "Comfy-Org/Lens", RepoPath = "diffusion_models/lens_turbo_bf16.safetensors",
                        TargetSubdir = "Stable-Diffusion/Lens", Role = "transformer",
                        Sha256 = "a9fc0e27261d9199d4e46e573a6b247f3cd94beec0241e61ae9eaee5ae9ef7c9" },
                    SideModels.LensGptOss20b,
                    SideModels.Flux2Vae,
                },
            },
            new CatalogEntry
            {
                Id = "krea2",
                Modality = img,
                DisplayName = "Krea 2 Turbo",
                Architecture = "Krea2 DiT (Qwen3-VL-4B + Qwen-Image VAE)",
                Status = ok,
                CliDrivable = true, // `hartsy image -m krea2` verified end-to-end 2026-07-20: transformer asset
                // resolved on disk, side models auto-resolved, coherent 1024x1024 photoreal fox output
                // Side models come straight from SideModels — the same SHA-256-pinned entries Krea2Recipe
                // downloads — so the catalog can never disagree with what the engine actually loads. Only the
                // transformer is spelled out here: it is the checkpoint itself and has no SideModels entry.
                Assets = new ModelAsset[]
                {
                    new() { Repo = "Comfy-Org/Krea-2", RepoPath = "diffusion_models/krea2_turbo_fp8_scaled.safetensors", TargetSubdir = "Stable-Diffusion/Krea2", Role = "transformer",
                        Sha256 = "eb4dd8c612cfd10f64f25b057e6e6bbcb5737c94a7372177e456dbf7579502f1" },
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
