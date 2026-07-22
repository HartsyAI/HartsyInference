using HartsyInference.Engine.Audio;

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
        const Modality vc = Modality.VoiceConvert;
        const Modality fx = Modality.Fx;
        const ModelStatus ok = ModelStatus.Verified;
        const ModelStatus vp = ModelStatus.ValidationPending;
        const ModelStatus st = ModelStatus.Structural;

        return new List<CatalogEntry>
        {
            // Text / LLM
            E("qwen2", txt, "Qwen2.5 (0.5B → 7B)", "Qwen2 dense transformer", ok, cli: true),
            new CatalogEntry
            {
                // `hartsy text -m qwen3` verified end-to-end 2026-07-22 (CLI catalog pass): coherent output on
                // both the 3060 (--low-vram-quant) and 4090, and --thinking/--no-thinking confirmed to produce
                // genuinely different output against this real checkpoint (see MODEL_STATUS_LLM.md).
                Id = "qwen3", Modality = txt, DisplayName = "Qwen3 (0.6B → 7B)", Architecture = "Qwen3 dense transformer",
                Status = ok, CliDrivable = true,
                Assets = new ModelAsset[]
                {
                    new() { Repo = "Qwen/Qwen3-4B-GGUF", RepoPath = "Qwen3-4B-Q4_K_M.gguf",
                        TargetSubdir = "LLM/qwen3", Role = "transformer",
                        Sha256 = "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5" },
                },
            },
            E("llama3", txt, "Llama-3.x", "Llama dense transformer", st, cli: true),
            E("mistral", txt, "Mistral (dense)", "Mistral dense transformer", st, cli: true),
            E("gguf", txt, "Quantized GGUF (Q4/Q8)", "config-driven, any GGUF LLM", ok, cli: true),

            // Text / LLM — dense families verified engine-internally per MODEL_STATUS_LLM.md, but not yet run
            // through `hartsy text` itself, so Status stays Structural (same convention as llama3/mistral above)
            // until a CLI pass confirms real output. No Assets: no HF GGUF repo for these is reliably documented
            // in this repo's checklists — see docs/Checklists/LLM_CLI_CATALOG_HANDOFF.md for what a GPU-equipped
            // session still needs to source and verify.
            E("gemma", txt, "Gemma 2 / Gemma 3 (text)", "GeGLU, √d embed scale, sandwich norm, dual-RoPE, attn/final logit soft-cap", st, cli: true),
            E("phi", txt, "Phi-3 / Phi-3.5-mini / Phi-4-mini", "fused QKV split, LongRoPE, partial rotary, gpt-4o tokenizer", st, cli: true),
            E("stablelm2", txt, "StableLM-2 (1.6B)", "partial rotary + QKV bias", st, cli: true),
            E("granite3", txt, "Granite-3", "embedding/attention/residual/logit scalar multipliers", st, cli: true),
            E("command-r", txt, "Cohere Command-R", "LayerNorm, parallel residual, interleaved RoPE, NoPE global layers, logit-scale", st, cli: true),
            E("olmoe", txt, "OLMoE (1B-7B-0924)", "MoE, whole-vector Q/K norm", st, cli: true),
            E("granite-moe", txt, "Granite-MoE", "scalar multipliers + MoE", st, cli: true),
            E("gemma4", txt, "Gemma-4 (E2B/E4B mobile)", "per-layer embeddings, per-layer head-dim, cross-layer KV donor sharing, weightless V-norm", st, cli: true),

            // Text / LLM — MoE / large dense: architecture, key-mapper, and slice/unit tests are built and pass
            // (see MODEL_STATUS_LLM.md "build-defer"), but no e2e run exists yet at any size — every one of these
            // checkpoints exceeds 12GB, so real verification needs a bigger GPU than this project has used so far.
            E("mixtral", txt, "Mixtral 8x7B", "llama+experts MoE, interleaved RoPE, renorm — 47B total params", st, cli: true),
            E("qwen3-moe", txt, "Qwen3-MoE (30B-A3B / 235B)", "per-head Q/K norm, no shared expert", st, cli: true),
            E("deepseek-v2-lite", txt, "DeepSeek-V2-Lite", "MLA + DeepSeek-MoE — OOMs a 12GB card at preload", st, cli: true),
            E("deepseek-v3", txt, "DeepSeek-V3 (671B) / Kimi-K2 (1T)", "MLA + MoE + sigmoid group-routing + q-LoRA query — likely needs multiple GPUs", st, cli: true),
            E("gpt-oss", txt, "GPT-OSS (20B / 120B)", "attention sinks, MoE, o200k tokenizer", st, cli: true),
            E("gemma4-moe", txt, "Gemma-4 (31B-dense / 26B-A4B-MoE)", "parallel dense+MoE branch, per-layer FFN width", st, cli: true),

            // Text / LLM — vision-language (VLM). Attach an image with `--image <path>`; all five are verified
            // e2e per MODEL_STATUS_LLM.md, but (like the dense families above) not yet through the CLI's new
            // --image flag specifically.
            E("llama32-vision", txt, "Llama-3.2-Vision-11B (mllama)", "gated cross-attention VLM, own 560px ViT (no token splice)", st, cli: true),
            E("gemma3-vision", txt, "Gemma-3-4B-vision", "SigLIP + avg-pool/RMSNorm/Linear projector", st, cli: true),
            E("smolvlm2", txt, "SmolVLM2-2.2B", "SigLIP + idefics3 pixel-shuffle projector", st, cli: true),
            E("llava15", txt, "LLaVA-1.5-7B", "CLIP ViT (CLS token, pre-LN, quick-GELU) + MLP projector", st, cli: true),
            E("qwen25-vl", txt, "Qwen2.5-VL (3B / 7B)", "own ViT, Conv3D patch embed, 2D-RoPE, window attention", st, cli: true),

            // Text / LLM — non-transformer / hybrid decoders (Ssm/*Model.cs, not GenericTransformer).
            E("mamba", txt, "Mamba-1 / Mamba-2, Falcon-Mamba", "selective state-space scan + causal Conv1d, no attention", st, cli: true),
            E("rwkv", txt, "RWKV-6 / RWKV-7", "WKV recurrence + data-dependent token-shift LoRA + GroupNorm", st, cli: true),
            E("qwen35", txt, "Qwen3.5 (Gated DeltaNet hybrid)", "every 4th layer is GQA+RoPE, the rest are Gated DeltaNet delta-rule recurrence", st, cli: true),

            new CatalogEntry
            {
                // NOT reachable via `hartsy text` today: TextService.LoadInto only routes a GGUF to the
                // decoder-only GgufLanguageModel path or (for SSM architectures) SsmLanguageModel — there is no
                // seq2seq branch that calls the standalone Seq2Seq/T5Model.cs loader, so a T5/FLAN-T5 GGUF would
                // load as (and fail as) a decoder-only transformer. T5Model itself is verified encoder+decoder
                // parity (cosine = 1.0 vs HF) per MODEL_STATUS_LLM.md, but that's exercised directly in the
                // Diffusion package's text-encoder tests, not through TextService/hartsy text. Listed for
                // discoverability; CliDrivable stays false until TextService gains a seq2seq generation path.
                Id = "t5", Modality = txt, DisplayName = "T5 / FLAN-T5", Architecture = "encoder-decoder, rel-pos bias, cross-attention, GeGLU",
                Status = st, CliDrivable = false,
            },

            // Text / LLM — architectures with a real IGgufKeyMapper (GgufKeyMapperRegistry) but no bring-up/e2e
            // run documented anywhere in this repo's checklists, unlike every family above. Included because the
            // mapping genuinely exists and a user may already have a matching GGUF; expect the first real run to
            // surface bugs the doc-verified families already had shaken out.
            E("glm4", txt, "GLM-4", "sandwich norm, fused gate/up projection (Glm4KeyMapper)", st, cli: true),
            E("gpt2", txt, "GPT-2 / BLOOM / GPT-NeoX", "absolute position embeddings, non-gated GELU (Gpt2KeyMapper)", st, cli: true),
            E("starcoder2", txt, "StarCoder2", "shares the llama-family key-mapper path (LlamaKeyMapper)", st, cli: true),

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
                // `hartsy image -m chroma` re-verified 2026-07-21: the terse prompt "an astronaut riding a horse"
                // reproducibly yields an off-prompt vintage/sepia rider-in-a-hat (weak compound-subject binding at
                // this family's default cfg, same class as AuraFlow) — needs an explicit compositional prompt
                // ("a photo of an astronaut sitting on top of a brown horse...") + a negative prompt + cfg~7 to
                // reliably bind both subjects. CliDrivable stays true because that combination IS coherent and
                // correct, not because the terse prompt works out of the box.
                CliDrivable = true,
                Assets = new ModelAsset[]
                {
                    // Switched from silveroxides/Chroma1-HD-fp8-scaled (extremely slow host, ~2MB/s) to the
                    // Comfy-Org repack of the same fp8-mixed weights — verified byte-different but functionally
                    // identical output on this file. Local filename kept as "-final" for continuity with the old
                    // pin.
                    new() { Repo = "Comfy-Org/Chroma1-HD_repackaged", RepoPath = "split_files/diffusion_models/Chroma1-HD-fp8mixed.safetensors",
                        TargetSubdir = "Stable-Diffusion/Chroma", TargetName = "Chroma1-HD-fp8mixed-final.safetensors", Role = "transformer",
                        Sha256 = "a2928ca6075f308f4d5e2182e2b96120fa8ad270ec6ea9b1b5c724c85c49a575" },
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
                CliDrivable = true, // `hartsy image -m hunyuan-image` verified end-to-end 2026-07-21 (1024x1024;
                // 2048x2048 still separately OOMs at VAE decode — see MODEL_STATUS_IMAGE.md). The former all-black
                // output was HunyuanImageTransformer's text stream overflowing F16 in the double-stream blocks
                // (fixed by keeping this model's block loop at F32 — see the Forward() comment).
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
                // Re-verified 2026-07-21: a bare {"high_level_description":"..."} (this comment's old example)
                // fails Ideogram4Dialect.Validate — it requires compositional_deconstruction.background (non-null)
                // and an elements array; the CLI feeds the prompt string verbatim (no in-process magic-prompt
                // expansion), so an incomplete JSON produces weak/off-schema conditioning. Full required shape:
                // {"high_level_description":"...","style_description":{"aesthetics":"...","lighting":"...",
                // "medium":"...","photo":"..."},"compositional_deconstruction":{"background":"...",
                // "elements":[{"type":"obj","desc":"..."},{"type":"obj","desc":"..."}]}} — confirmed correct output
                // with this full shape (clear astronaut in white spacesuit/helmet on a brown horse).
                CliDrivable = true,
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
                // Re-verified 2026-07-21: 1024x1024 (this family's native/documented default) reproducibly OOMs
                // on this 24GB 4090 — even in complete isolation with nothing else on the GPU, and even inside
                // the tiled-VAE-decode OOM-recovery path itself (CudaMemory.AllocateAsync inside VaeDecoder's
                // Conv2D). Contradicts the older "warm 61.5s, peak 23.8GB" bench note — that number leaves only
                // ~700MB headroom on a 24564MB card, which real-world CUDA context/allocator overhead eats into;
                // not reproducible here regardless of what else is running. 768x768 succeeds cleanly (clear
                // on-prompt astronaut-on-horse, no artifacts) — use --width 768 --height 768 until this gets a
                // real memory-budget pass (same class of issue as zeta-chroma's 1024 ceiling).
                CliDrivable = true,
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

            // Transcription — repos/files below are exactly what SttCatalog's descriptors (src/HartsyInference.Engine/
            // Audio/Stt/SttCatalog.cs) already resolve; Assets here only drive the CLI's pre-download confirm prompt
            // (see ModelAcquisition.EnsureAudioAssetsPresent) — the real fetch always goes through the engine's own
            // AudioModelCache regardless, so a missed sidecar file just downloads silently during generation.
            E("whisper", stt, "Whisper (tiny → large-v3)", "encoder-decoder", ok, cli: true),
            E("moonshine", stt, "Moonshine", "encoder-decoder", ok, cli: true),
            new CatalogEntry
            {
                Id = "distilwhisper", Modality = stt, DisplayName = "distil-whisper", Architecture = "encoder-decoder (distilled)", Status = ok,
                // Verified 2026-07-21: the BARE `-m distilwhisper` id fails — SttCatalog.ResolveDistilWhisperRepo's
                // no-match default is "distil-whisper/distil-large-v3.5", but WhisperPipeline.InferConfig's repo
                // switch only recognizes v2/v3/medium.en/small.en (no v3.5 case) → "Unknown Whisper repo". Use an
                // explicit variant that IS recognized, e.g. `-m distilwhisper:v3`.
                CliDrivable = true, // `hartsy transcribe -m distilwhisper:v3` (or :v2/:medium/:small) — SttCatalog "distilwhisper"; bare id currently broken, see above
                Assets = new ModelAsset[]
                {
                    new() { Repo = "distil-whisper/distil-large-v3", RepoPath = "model.safetensors", TargetSubdir = "Audio/DistilWhisper", Role = "encoder-decoder" },
                },
            },
            new CatalogEntry
            {
                Id = "moonshinestreaming", Modality = stt, DisplayName = "Moonshine (2nd-gen streaming)", Architecture = "sliding-window encoder-decoder", Status = ok,
                CliDrivable = true, // `hartsy transcribe -m moonshinestreaming` — SttCatalog "moonshinestreaming"; full-utterance batch, not true chunked streaming yet
                Assets = new ModelAsset[]
                {
                    new() { Repo = "UsefulSensors/moonshine-streaming-tiny", RepoPath = "model.safetensors", TargetSubdir = "Audio/MoonshineStreaming", Role = "encoder-decoder" },
                },
            },
            new CatalogEntry
            {
                Id = "kyutaistt", Modality = stt, DisplayName = "Kyutai STT (1B / 2.6B)", Architecture = "Helium LM + Mimi codec", Status = ok,
                CliDrivable = true, // `hartsy transcribe -m kyutaistt` — SttCatalog "kyutaistt"; input decoded at 24 kHz
                Assets = new ModelAsset[]
                {
                    new() { Repo = "kyutai/stt-1b-en_fr-trfs", RepoPath = "model.safetensors", TargetSubdir = "Audio/KyutaiStt", Role = "backbone + mimi codec" },
                    new() { Repo = "kyutai/stt-1b-en_fr", RepoPath = "tokenizer_en_fr_audio_8000.model", TargetSubdir = "Audio/KyutaiStt", Role = "tokenizer" },
                },
            },
            new CatalogEntry
            {
                Id = "whisperstreaming", Modality = stt, DisplayName = "Whisper Streaming (LocalAgreement-2)", Architecture = "encoder-decoder (streamed)", Status = ok,
                CliDrivable = true, // `hartsy transcribe -m whisperstreaming` — SttCatalog "whisperstreaming"; same weights as whisper, driven through the stabilizer
                Assets = new ModelAsset[]
                {
                    new() { Repo = "openai/whisper-base", RepoPath = "model.safetensors", TargetSubdir = "Audio/Whisper", Role = "encoder-decoder" },
                },
            },

            // Text-to-speech — repos/files below mirror TtsCatalog's descriptors (src/HartsyInference.Engine/Audio/
            // Tts/**); the gated-repo audit (memory audiolab-gated-repo-audit, 2026-07-16) already vetted every one
            // of these as non-gated or on a non-gated mirror. Clone-only models need `--reference <wav>` (some also
            // `--ref-text`); see SpeechCommand's help.
            E("piper", tts, "Piper (en_US-lessac-medium, …)", "VITS + espeak phonemes", ok, cli: true),
            new CatalogEntry
            {
                Id = "kokoro", Modality = tts, DisplayName = "Kokoro-82M", Architecture = "StyleTTS-family vocoder", Status = ok,
                CliDrivable = true, // `hartsy speak -m kokoro` — TtsCatalog "kokoro"; voice packs fetch per `--voice` (default af_heart)
                Assets = new ModelAsset[]
                {
                    new() { Repo = "Hartsy/kokoro-82m-safetensors", RepoPath = "kokoro-82m.safetensors", TargetSubdir = "Audio/Kokoro", Role = "transformer" },
                },
            },
            new CatalogEntry
            {
                Id = "bark", Modality = tts, DisplayName = "Bark", Architecture = "GPT-style TTS", Status = ok,
                CliDrivable = true, // `hartsy speak -m bark` — TtsCatalog "bark" (3-stage GPT cascade + EnCodec 24 kHz)
                Assets = new ModelAsset[]
                {
                    new() { Repo = "suno/bark", RepoPath = "model.safetensors", TargetSubdir = "Audio/Bark", Role = "transformer" },
                },
            },
            new CatalogEntry
            {
                Id = "styletts2", Modality = tts, DisplayName = "StyleTTS2", Architecture = "style-diffusion TTS", Status = ok,
                CliDrivable = true, // `hartsy speak -m styletts2 --reference <wav>` — TtsCatalog "styletts2"; clone from a reference clip
                Assets = new ModelAsset[]
                {
                    new() { Repo = "yl4579/StyleTTS2-LibriTTS", RepoPath = "Models/LibriTTS/epochs_2nd_00020.pth", TargetSubdir = "Audio/StyleTts2", Role = "transformer" },
                },
            },
            new CatalogEntry
            {
                Id = "sparktts", Modality = tts, DisplayName = "Spark-TTS", Architecture = "Qwen2.5-0.5B LM + BiCodec", Status = ok,
                CliDrivable = true, // `hartsy speak -m sparktts` — TtsCatalog key is "sparktts" (no hyphen; AudioModelSelector matches the catalog Id literally, case-insensitive only); controllable mode (gender/speed), no cloning yet
                Assets = new ModelAsset[]
                {
                    new() { Repo = "SparkAudio/Spark-TTS-0.5B", RepoPath = "model.safetensors", TargetSubdir = "Audio/SparkTts", Role = "transformer" },
                },
            },
            new CatalogEntry
            {
                Id = "cosyvoice", Modality = tts, DisplayName = "CosyVoice 2", Architecture = "Qwen LM + OT-CFM flow", Status = ok,
                // Verified 2026-07-21: `--reference` ALONE (no --ref-text) produced garbled, non-word-correct output
                // ("And a tall fall, tear, tape, tape." for a plain test sentence) — CosyVoiceModel accepts an empty
                // RefText without throwing, but zero-shot quality clearly depends on it. With --ref-text set to the
                // reference clip's real transcript, output was word-perfect. Always pass both for this model.
                CliDrivable = true, // `hartsy speak -m cosyvoice --reference <wav> --ref-text "<transcript>"` — TtsCatalog "cosyvoice"; zero-shot clone, ref-text effectively required for quality
                Assets = new ModelAsset[]
                {
                    new() { Repo = "FunAudioLLM/CosyVoice2-0.5B", RepoPath = "llm.pt", TargetSubdir = "Audio/CosyVoice", Role = "transformer" },
                    new() { Repo = "FunAudioLLM/CosyVoice2-0.5B", RepoPath = "flow.pt", TargetSubdir = "Audio/CosyVoice", Role = "flow decoder" },
                    new() { Repo = "FunAudioLLM/CosyVoice2-0.5B", RepoPath = "hift.pt", TargetSubdir = "Audio/CosyVoice", Role = "vocoder" },
                    new() { Repo = "ResembleAI/chatterbox", RepoPath = "s3gen.safetensors", TargetSubdir = "Audio/CosyVoice", Role = "s3 tokenizer + CAM++" },
                },
            },
            new CatalogEntry
            {
                Id = "vibevoice", Modality = tts, DisplayName = "VibeVoice", Architecture = "diffusion TTS", Status = ok,
                CliDrivable = true, // `hartsy speak -m vibevoice --reference <wav>` — re-verified 2026-07-21: a
                // 2026-07-21 pass had flagged this BROKEN ("[Hindi]"/non-English babble via Whisper on the same
                // jfk.wav reference), but re-testing did NOT reproduce it — 3 independent prompts (short pangram,
                // long multi-sentence, and a third unrelated paragraph) with the same reference all transcribed
                // Whisper word-perfect (only trivial "a/the", "morning/evening"-class mishears). `git log` on the
                // VibeVoice sources shows no change since the 07-17 perf pass except a namespace rename — no
                // regression found. Root cause of the original failure is unknown (not reproduced, so not
                // diagnosable); treat as a one-off unless it recurs.
                Assets = new ModelAsset[]
                {
                    new() { Repo = "microsoft/VibeVoice-1.5B", RepoPath = "model-00001-of-00003.safetensors", TargetSubdir = "Audio/VibeVoice", Role = "transformer" },
                    new() { Repo = "microsoft/VibeVoice-1.5B", RepoPath = "model-00002-of-00003.safetensors", TargetSubdir = "Audio/VibeVoice", Role = "transformer" },
                    new() { Repo = "microsoft/VibeVoice-1.5B", RepoPath = "model-00003-of-00003.safetensors", TargetSubdir = "Audio/VibeVoice", Role = "transformer" },
                },
            },
            new CatalogEntry
            {
                Id = "fishspeech", Modality = tts, DisplayName = "Fish-Speech 1.5", Architecture = "DualAR + tiktoken", Status = ok,
                CliDrivable = true, // `hartsy speak -m fishspeech` — TtsCatalog key is "fishspeech" (no hyphen); was "fish-speech" here before, which never resolved
                Assets = new ModelAsset[]
                {
                    new() { Repo = "fishaudio/fish-speech-1.5", RepoPath = "model.pth", TargetSubdir = "Audio/FishSpeech", Role = "transformer" },
                    new() { Repo = "fishaudio/fish-speech-1.5", RepoPath = "firefly-gan-vq-fsq-8x1024-21hz-generator.pth", TargetSubdir = "Audio/FishSpeech", Role = "codec" },
                },
            },
            new CatalogEntry
            {
                Id = "f5", Modality = tts, DisplayName = "F5-TTS", Architecture = "voice cloning, flow-matching DiT", Status = ok,
                CliDrivable = true, // `hartsy speak -m f5 --reference <wav> --ref-text "..."` — TtsCatalog key is "f5" (was "f5-tts" here before, which never resolved); clone-only, needs both
                Assets = new ModelAsset[]
                {
                    new() { Repo = "SWivid/F5-TTS", RepoPath = "F5TTS_Base/model_1200000.safetensors", TargetSubdir = "Audio/F5Tts", Role = "transformer" },
                },
            },
            new CatalogEntry
            {
                Id = "dia", Modality = tts, DisplayName = "Dia-1.6B (0626)", Architecture = "byte-level dialogue TTS + DAC codec", Status = ok,
                CliDrivable = true, // `hartsy speak -m dia "[S1] ... [S2] ..."` — TtsCatalog "dia"; needs the 0626 checkpoint (the original degenerates)
                Assets = new ModelAsset[]
                {
                    new() { Repo = "nari-labs/Dia-1.6B-0626", RepoPath = "pytorch_model.bin", TargetSubdir = "Audio/Dia", Role = "transformer" },
                    new() { Repo = "descript/descript-audio-codec", RepoPath = "weights.pth", TargetSubdir = "Audio/Dia", Role = "codec" },
                },
            },
            new CatalogEntry
            {
                Id = "orpheus", Modality = tts, DisplayName = "Orpheus", Architecture = "Llama-3.2-3B + SNAC 24 kHz", Status = ok,
                CliDrivable = true, // `hartsy speak -m orpheus` — TtsCatalog "orpheus"; non-gated mirror of the license-gated release
                Assets = new ModelAsset[]
                {
                    new() { Repo = "unsloth/orpheus-3b-0.1-ft", RepoPath = "model.safetensors", TargetSubdir = "Audio/Orpheus", Role = "transformer" },
                    new() { Repo = "hubertsiuzdak/snac_24khz", RepoPath = "model.safetensors", TargetSubdir = "Audio/Orpheus", Role = "codec" },
                },
            },
            new CatalogEntry
            {
                Id = "csm", Modality = tts, DisplayName = "CSM-1B (Sesame)", Architecture = "dual-transformer + Mimi 24 kHz", Status = ok,
                CliDrivable = true, // `hartsy speak -m csm` — TtsCatalog "csm"; fixed 2026-07-21: the prior mirror
                // (nielsr/csm-1b) ships torchtune-style keys ("attn.q_proj", "sa_norm.scale") that never matched
                // CsmModel.LoadWeights, throwing KeyNotFoundException on "backbone.norm.weight". Switched to the
                // unsloth/csm-1b mirror (HF transformers-export key layout) + CsmWeightRemap to split its two
                // combined tensors (audio_embeddings, codebooks_head) into the per-codebook slices the loader reads.
                // Also bundles its own 32-codebook Mimi (codec_model.* keys) — used instead of a separate
                // kyutai/mimi download, which ships only 8 codebooks (a mismatch against CSM's 32-codebook decoder).
                Assets = new ModelAsset[]
                {
                    new() { Repo = "unsloth/csm-1b", RepoPath = "model.safetensors", TargetSubdir = "Audio/Csm", Role = "transformer + codec" },
                },
            },
            new CatalogEntry
            {
                Id = "neutts", Modality = tts, DisplayName = "NeuTTS Air", Architecture = "Qwen2.5-0.5B LM + NeuCodec", Status = ok,
                CliDrivable = true, // `hartsy speak -m neutts [--reference <wav>]` — TtsCatalog "neutts"; clone is optional (falls back to the default voice)
                Assets = new ModelAsset[]
                {
                    new() { Repo = "neuphonic/neutts-air", RepoPath = "model.safetensors", TargetSubdir = "Audio/NeuTts", Role = "transformer" },
                    new() { Repo = "neuphonic/neucodec", RepoPath = "model.safetensors", TargetSubdir = "Audio/NeuTts", Role = "codec" },
                },
            },
            new CatalogEntry
            {
                Id = "qwen3tts", Modality = tts, DisplayName = "Qwen3-TTS", Architecture = "12 Hz talker + MTP + codec", Status = ok,
                // Verified end-to-end 2026-07-21: Qwen3TtsModel.ResolveRepo/ResolveMode read the WHOLE variant
                // string, so a bare `-m qwen3tts` (variant = "qwen3tts", no "CustomVoice"/"VoiceDesign" substring)
                // resolves to the Base checkpoint's voice_clone mode and REQUIRES --reference. For the preset-speaker
                // or instruct-text modes, pass the variant explicitly.
                CliDrivable = true, // `hartsy speak -m qwen3tts:1.7B-CustomVoice` (preset speaker) or
                // `-m qwen3tts:1.7B-VoiceDesign` (instruct text) or `-m qwen3tts --reference <wav>` (Base/voice_clone,
                // the bare-id default) — TtsCatalog key is "qwen3tts" (no hyphen)
                Assets = new ModelAsset[]
                {
                    new() { Repo = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", RepoPath = "model.safetensors", TargetSubdir = "Audio/Qwen3Tts", Role = "transformer" },
                    new() { Repo = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", RepoPath = "speech_tokenizer/model.safetensors", TargetSubdir = "Audio/Qwen3Tts", Role = "codec" },
                },
            },
            new CatalogEntry
            {
                Id = "chatterbox", Modality = tts, DisplayName = "Chatterbox", Architecture = "T3 LM + S3Gen (CosyVoice-derived)", Status = ok,
                CliDrivable = true, // `hartsy speak -m chatterbox [--reference <wav>] [--exaggeration N]` — TtsCatalog "chatterbox"; clone is optional
                Assets = new ModelAsset[]
                {
                    new() { Repo = "ResembleAI/chatterbox", RepoPath = "t3_cfg.safetensors", TargetSubdir = "Audio/Chatterbox", Role = "transformer" },
                    new() { Repo = "ResembleAI/chatterbox", RepoPath = "s3gen.safetensors", TargetSubdir = "Audio/Chatterbox", Role = "vocoder" },
                    new() { Repo = "ResembleAI/chatterbox", RepoPath = "conds.pt", TargetSubdir = "Audio/Chatterbox", Role = "default conditioning" },
                },
            },
            new CatalogEntry
            {
                Id = "kyutaitts", Modality = tts, DisplayName = "Kyutai TTS (1.6B en/fr)", Architecture = "DSM (Helium LM + Mimi)", Status = ok,
                CliDrivable = true, // `hartsy speak -m kyutaitts` — TtsCatalog key is "kyutaitts" (no hyphen); built-in voice embeds (kyutai/tts-voices), no --reference needed
                Assets = new ModelAsset[]
                {
                    new() { Repo = "kyutai/tts-1.6b-en_fr", RepoPath = "dsm_tts_1e68beda@240.safetensors", TargetSubdir = "Audio/KyutaiTts", Role = "transformer" },
                    new() { Repo = "kyutai/tts-1.6b-en_fr", RepoPath = "tokenizer-e351c8d8-checkpoint125.safetensors", TargetSubdir = "Audio/KyutaiTts", Role = "codec" },
                },
            },
            new CatalogEntry
            {
                Id = "melotts", Modality = tts, DisplayName = "MeloTTS (English-v3)", Architecture = "BERT + VITS", Status = ok,
                CliDrivable = true, // `hartsy speak -m melotts` — TtsCatalog "melotts"
                Assets = new ModelAsset[]
                {
                    new() { Repo = "myshell-ai/MeloTTS-English-v3", RepoPath = "checkpoint.pth", TargetSubdir = "Audio/MeloTts", Role = "transformer" },
                },
            },
            new CatalogEntry
            {
                Id = "pockettts", Modality = tts, DisplayName = "PocketTTS", Architecture = "voice-KV-primed streaming flow-LM", Status = ok,
                CliDrivable = true, // `hartsy speak -m pockettts --voice alba` — TtsCatalog "pockettts"; voice is a built-in NAME (not a file); non-gated mirror
                Assets = new ModelAsset[]
                {
                    new() { Repo = "kyutai/pocket-tts-without-voice-cloning", RepoPath = "languages/english/model.safetensors", TargetSubdir = "Audio/PocketTts", Role = "transformer" },
                },
            },
            new CatalogEntry
            {
                Id = "zonos", Modality = tts, DisplayName = "Zonos-v0.1", Architecture = "transformer + ResNet293 speaker encoder", Status = ok,
                CliDrivable = true, // `hartsy speak -m zonos --reference <wav>` — TtsCatalog "zonos"; clone-only
                Assets = new ModelAsset[]
                {
                    new() { Repo = "Zyphra/Zonos-v0.1-transformer", RepoPath = "model.safetensors", TargetSubdir = "Audio/Zonos", Role = "transformer" },
                    new() { Repo = "Zyphra/Zonos-v0.1-speaker-embedding", RepoPath = "ResNet293_SimAM_ASP_base.pt", TargetSubdir = "Audio/Zonos", Role = "speaker encoder" },
                    new() { Repo = "Zyphra/Zonos-v0.1-speaker-embedding", RepoPath = "ResNet293_SimAM_ASP_base_LDA-128.pt", TargetSubdir = "Audio/Zonos", Role = "speaker LDA" },
                    new() { Repo = "descript/descript-audio-codec", RepoPath = "weights.pth", TargetSubdir = "Audio/Zonos", Role = "codec" },
                },
            },
            new CatalogEntry
            {
                Id = "gptsovits", Modality = tts, DisplayName = "GPT-SoVITS v2", Architecture = "HuBERT + s1 GPT + s2 SoVITS", Status = ok,
                // Verified word-correct 2026-07-21 (whisper medium.en) with --reference + --ref-text; --ref-text is
                // REQUIRED (GptSoVitsModel throws InvalidOperationException without it — "needs the reference
                // transcript"), unlike CosyVoice where it's optional-but-recommended.
                CliDrivable = true, // `hartsy speak -m gptsovits --reference <wav> --ref-text "<transcript>"` — TtsCatalog key is "gptsovits" (no hyphen); clone-only, both required
                Assets = new ModelAsset[]
                {
                    new() { Repo = "lj1995/GPT-SoVITS", RepoPath = "gsv-v2final-pretrained/s2G2333k.pth", TargetSubdir = "Audio/GptSoVits", Role = "s2 SoVITS" },
                    new() { Repo = "lj1995/GPT-SoVITS", RepoPath = "gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt", TargetSubdir = "Audio/GptSoVits", Role = "s1 GPT" },
                    new() { Repo = "lj1995/GPT-SoVITS", RepoPath = "chinese-hubert-base/pytorch_model.bin", TargetSubdir = "Audio/GptSoVits", Role = "HuBERT" },
                },
            },
            new CatalogEntry
            {
                Id = "zipvoice", Modality = tts, DisplayName = "ZipVoice", Architecture = "Zipformer flow-matching + Vocos", Status = ok,
                CliDrivable = true, // `hartsy speak -m zipvoice --reference <wav> --ref-text "..."` — TtsCatalog "zipvoice"; clone-only, needs both.
                // Slow today (~11 min / 10s clip on the 3060, no GPU-residency pass yet) — see MODEL_STATUS_AUDIO.md.
                Assets = new ModelAsset[]
                {
                    new() { Repo = "k2-fsa/ZipVoice", RepoPath = "zipvoice/model.safetensors", TargetSubdir = "Audio/ZipVoice", Role = "transformer" },
                },
            },

            // Music / audio generation — repos/files mirror MusicCatalog's descriptors (src/HartsyInference.Engine/
            // Audio/Music/**).
            E("musicgen", mus, "MusicGen", "transformer + EnCodec", ok, cli: true),
            new CatalogEntry
            {
                Id = "audiogen", Modality = mus, DisplayName = "AudioGen", Architecture = "MusicGen-arch + T5-large", Status = ok,
                CliDrivable = true, // `hartsy music -m audiogen "a dog barking"` — MusicCatalog "audiogen"; 16 kHz sound effects, not melodic music
                Assets = new ModelAsset[]
                {
                    new() { Repo = "facebook/audiogen-medium", RepoPath = "state_dict.bin", TargetSubdir = "Audio/AudioGen", Role = "transformer" },
                    new() { Repo = "facebook/audiogen-medium", RepoPath = "compression_state_dict.bin", TargetSubdir = "Audio/AudioGen", Role = "codec" },
                    new() { Repo = "google-t5/t5-large", RepoPath = "pytorch_model.bin", TargetSubdir = "Audio/AudioGen", Role = "text encoder" },
                },
            },
            // "acestep"/"yue" (no hyphen) — MusicCatalog keys are AudioWeightsCatalog.AceStepId/YueId; the
            // pre-existing "ace-step" spelling here never resolved (AudioModelSelector matches the catalog Id
            // literally). Unlike musicgen/audiogen/stableaudio/heartmula (self-download via AudioModelCache), these
            // two are the Engine's "registry-backed local-checkpoint families": they resolve their weights via
            // AudioWeightsCatalog + the STANDARD ModelDownloader/ModelAsset machinery (same as image models), landing
            // under Models/audio/music/{acestep,yue}/ — reusing AudioWeightsCatalog.AssetsFor directly here (like
            // SideModels for image entries) keeps this catalog and the Engine's own resolution in lockstep instead of
            // duplicating repo/file/sha data. Default variant shown ("turbo"/"en-cot") is just the CLI's pre-download
            // preview for the bare id — `id:variant` still resolves any other registered variant at generation time.
            new CatalogEntry
            {
                Id = "acestep", Modality = mus, DisplayName = "ACE-Step", Architecture = "flow-matching DiT", Status = ok,
                CliDrivable = true, // `hartsy music -m acestep` (turbo default) or `-m acestep:sft`/`:base`/`:xl-turbo`/…
                Assets = AudioWeightsCatalog.AssetsFor(AudioWeightsCatalog.AceStepId, "turbo"),
            },
            new CatalogEntry
            {
                Id = "yue", Modality = mus, DisplayName = "YuE", Architecture = "dual-stage Llama", Status = ok,
                // Verified 2026-07-21: YueMusicModel.LoadAsync has NO auto-download fallback of its own (unlike
                // AceStepMusicModel, which calls AudioWeightsCatalog.EnsureAsync internally) — without this Assets
                // list + the CLI's download-confirm flow, YuE was manual-placement-only ("YuE checkpoint folder not
                // found... place the m-a-p/YuE-s1-7B-anneal-* folder there"). This Assets list is what makes
                // `hartsy music -m yue` self-serving for the first time.
                CliDrivable = true, // `hartsy music -m yue` (en-cot default, ~12.5 GB) or `-m yue:en-icl`/`:zh-cot`/`:zh-icl`
                Assets = AudioWeightsCatalog.AssetsFor(AudioWeightsCatalog.YueId, "en-cot"),
            },
            new CatalogEntry
            {
                Id = "stableaudio", Modality = mus, DisplayName = "Stable Audio Open Small", Architecture = "latent diffusion (Oobleck VAE)", Status = ok,
                CliDrivable = true, // `hartsy music -m stableaudio` — MusicCatalog key is "stableaudio" (no hyphen); Swarm-verified 11.89s stereo 44.1kHz in 2.85s gen (2026-07-20)
            },
            new CatalogEntry
            {
                Id = "heartmula", Modality = mus, DisplayName = "HeartMuLa (oss-3B)", Architecture = "CSM-LM + flow-match HeartCodec", Status = ok,
                CliDrivable = true, // `hartsy music -m heartmula` — MusicCatalog "heartmula"; 48 kHz, real-weight verified (see MODEL_STATUS_AUDIO.md)
            },

            // Voice conversion — VcCatalog (src/HartsyInference.Engine/Audio/Vc/**). Via `hartsy convert`.
            new CatalogEntry
            {
                Id = "openvoice", Modality = vc, DisplayName = "OpenVoice V2", Architecture = "tone-color transfer (Conv2d + GRU)", Status = ok,
                CliDrivable = true, // `hartsy convert <source.wav> -m openvoice --target <ref.wav>` — VcCatalog "openvoice"
                Assets = new ModelAsset[]
                {
                    new() { Repo = "myshell-ai/OpenVoiceV2", RepoPath = "converter/checkpoint.pth", TargetSubdir = "Audio/OpenVoice", Role = "transformer" },
                },
            },
            new CatalogEntry
            {
                Id = "rvc", Modality = vc, DisplayName = "RVC v2", Architecture = "ContentVec + YIN F0 + NSF-HiFiGAN", Status = ok,
                // No fixed weights repo: RVC re-voices toward a USER-TRAINED voice model placed at
                // Models/audio/clone/rvc/<name>.pth (or passed via --model-path) — there is nothing upstream to
                // catalog per se. The one auto-downloadable piece is the shared ContentVec content encoder, listed
                // here so the CLI can still offer to fetch it; VcCatalog.EnsureContentVecAsync converts it to
                // contentvec.safetensors on first use regardless of what's listed here.
                CliDrivable = true, // `hartsy convert <source.wav> -m rvc --model-path <voice.pth> [--pitch-shift N]` — VcCatalog "rvc"
                Assets = new ModelAsset[]
                {
                    new() { Repo = "lengyue233/content-vec-best", RepoPath = "pytorch_model.bin", TargetSubdir = "Audio/Rvc", Role = "content encoder (shared)" },
                },
            },

            // Audio effects (FX) — FxCatalog (src/HartsyInference.Engine/Audio/Fx/**). Via `hartsy fx separate|enhance`.
            new CatalogEntry
            {
                Id = "resemble-enhance", Modality = fx, DisplayName = "Resemble-Enhance", Architecture = "denoiser + LCFM enhancer + UnivNet", Status = vp,
                // Re-investigated 2026-07-21: the DeepSpeed-checkpoint angle was a scope red herring —
                // PytorchPickleLoader already parses enhancer_stage2/ds/G/default/mp_rank_00_model_states.pt fine
                // (909 real tensors, no reader change needed). The actual blocker: ResembleDenoiser/
                // ResembleIrmaeDecoder's assumed key layout (down/mid/up, conv1/norm1/conv2/norm2/downsample)
                // doesn't match the real checkpoint (encoder_blocks/middle_blocks/decoder_blocks, each pre_conv +
                // two PreactResBlocks of GroupNorm->GELU->Conv2d x2 — confirmed against the real
                // resemble_enhance/denoiser/unet.py + lcfm/irmae.py sources on GitHub). This is a genuine
                // forward-pass mismatch (module composition, not just names), so it needs those modules'
                // load/forward code rewritten to match — out of scope for this pass; deliberately left
                // ValidationPending/not-CliDrivable rather than half-fixing it. No Assets listed (nothing to
                // pre-fetch until the loader matches).
            },
            new CatalogEntry
            {
                Id = "demucs", Modality = fx, DisplayName = "Demucs (htdemucs)", Architecture = "hybrid transformer/conv separator", Status = ok,
                CliDrivable = true, // `hartsy fx separate <wav>` — fixed + verified 2026-07-21: not on HuggingFace
                // as a single-file checkpoint, so FxCatalog now auto-downloads the official Meta checkpoint
                // directly from dl.fbaipublicfiles.com (confirmed public, no auth). Verified real output: 4 stems
                // (drums/bass/other/vocals) on a real music clip, mutually distinct (pairwise sample corr
                // 0.007-0.14, i.e. not copies of each other or the mix) and non-silent. FxSeparateCommand always
                // forces the CPU backend (DemucsSpec's STFT/ISTFT has no CUDA/Vulkan implementation — CudaBackend
                // throws NotSupportedException) so the default invocation works without needing `-b cpu`.
                // `-m demucs:htdemucs_6s` (6 stems) is wired the same way but not individually run this pass.
                // `htdemucs_ft` stays --model-path-only: upstream ships it as a 4-checkpoint weight-averaged
                // ensemble, not a single 4-stem checkpoint.
            },

            // Vision
            new CatalogEntry
            {
                Id = "clip", Modality = vis, DisplayName = "CLIP ViT-L/14", Architecture = "ViT dual-tower embeddings", Status = ok,
                CliDrivable = true,
                Assets = new ModelAsset[]
                {
                    new() { Repo = "openai/clip-vit-large-patch14", RepoPath = "model.safetensors",
                        TargetSubdir = "Vision/Clip", Role = "transformer",
                        Sha256 = "a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb" },
                },
            },
            new CatalogEntry
            {
                Id = "siglip", Modality = vis, DisplayName = "SigLIP (base, patch16-224)", Architecture = "ViT dual-tower embeddings", Status = ok,
                CliDrivable = true,
                Assets = new ModelAsset[]
                {
                    new() { Repo = "google/siglip-base-patch16-224", RepoPath = "model.safetensors",
                        TargetSubdir = "Vision/Siglip", Role = "transformer",
                        Sha256 = "2c63cb7d1f2e95ba501893cbb8faeb4ea9a3af295498d35097126228659c2af8" },
                },
            },
            new CatalogEntry
            {
                Id = "dinov2", Modality = vis, DisplayName = "DINOv2 (small)", Architecture = "ViT dense features", Status = ok,
                CliDrivable = true,
                Assets = new ModelAsset[]
                {
                    new() { Repo = "facebook/dinov2-small", RepoPath = "model.safetensors",
                        TargetSubdir = "Vision/Dinov2", Role = "transformer",
                        Sha256 = "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1" },
                },
            },
            // YOLO8/11: raw Ultralytics .pt is a full pickled nn.Module object graph (not a flat state_dict) —
            // the engine's safe-subset pickle VM can't resolve it, and no pre-folded ungated safetensors mirror
            // was found on HF. CliDrivable stays false until either a pickle-VM extension or a found mirror
            // unblocks auto-download (see docs/Checklists/MODEL_STATUS_VISION.md).
            E("yolo8", vis, "YOLO8 (n → xl)", "object detection", vp),
            E("yolo11", vis, "YOLO11 (n → xl)", "object detection", vp),
            new CatalogEntry
            {
                Id = "rtdetr", Modality = vis, DisplayName = "RT-DETR (r18vd)", Architecture = "ResNet-18vd + hybrid encoder + deformable decoder", Status = ok,
                CliDrivable = true,
                Assets = new ModelAsset[]
                {
                    new() { Repo = "PekingU/rtdetr_r18vd", RepoPath = "model.safetensors",
                        TargetSubdir = "rtdetr", Role = "transformer", Sha256 = null },
                },
            },
            new CatalogEntry
            {
                Id = "grounding-dino", Modality = vis, DisplayName = "Grounding DINO (tiny)", Architecture = "Swin-T + BERT + deformable cross-modal encoder/decoder", Status = ok,
                CliDrivable = true,
                Assets = new ModelAsset[]
                {
                    new() { Repo = "IDEA-Research/grounding-dino-tiny", RepoPath = "model.safetensors",
                        TargetSubdir = "grounding-dino", Role = "transformer", Sha256 = null },
                    new() { Repo = "IDEA-Research/grounding-dino-tiny", RepoPath = "vocab.txt",
                        TargetSubdir = "grounding-dino", Role = "tokenizer", Sha256 = null },
                },
            },
            new CatalogEntry
            {
                Id = "clipseg", Modality = vis, DisplayName = "ClipSeg (rd64-refined)", Architecture = "CLIP + lightweight segmentation decoder", Status = vp,
                CliDrivable = true,
                Assets = new ModelAsset[]
                {
                    new() { Repo = "CIDAS/clipseg-rd64-refined", RepoPath = "model.safetensors",
                        TargetSubdir = "clipseg", Role = "transformer", Sha256 = null },
                },
            },
            new CatalogEntry
            {
                Id = "sam", Modality = vis, DisplayName = "SAM 2 (hiera-tiny)", Architecture = "Hiera windowed-attn encoder + two-way mask decoder", Status = ok,
                CliDrivable = true,
                Assets = new ModelAsset[]
                {
                    // Flat Lightning-style state dict ({"model": {...}}) — confirmed via a raw pickle opcode dump
                    // (only OrderedDict + torch._utils._rebuild_tensor_v2 + FloatStorage), directly loadable by
                    // PytorchPickleLoader with no offline conversion, unlike the Ultralytics YOLO checkpoints.
                    new() { Repo = "facebook/sam2-hiera-tiny", RepoPath = "sam2_hiera_tiny.pt",
                        TargetSubdir = "sam2", Role = "transformer", Sha256 = null },
                },
            },
            // "RetinaFace" was never actually built — the real, real-weight-verified face detector is a
            // YOLOv8-Face port (see MODEL_STATUS_VISION.md). Catalog id corrected to match; blocked on the same
            // pickle-graph issue as YOLO8/11, plus its known source (akanametov/yolo-face) is GitHub-only, not HF.
            E("yolov8-face", vis, "YOLOv8-Face + landmarks", "face detection + 5-pt landmarks", vp),
            E("arcface", vis, "ArcFace (IR-50, w600k_r50)", "face embedding", vp),
            new CatalogEntry
            {
                Id = "depth-anything", Modality = vis, DisplayName = "Depth-Anything-V2 (small)", Architecture = "DINOv2 backbone + DPT head", Status = ok,
                CliDrivable = true,
                Assets = new ModelAsset[]
                {
                    new() { Repo = "depth-anything/Depth-Anything-V2-Small", RepoPath = "depth_anything_v2_vits.pth",
                        TargetSubdir = "Vision/DepthAnything", Role = "transformer", Sha256 = null },
                },
            },
            new CatalogEntry
            {
                Id = "hed", Modality = vis, DisplayName = "HED (ControlNetHED soft-edge)", Architecture = "VGG16-style 5-block trunk", Status = ok,
                CliDrivable = true,
                Assets = new ModelAsset[]
                {
                    new() { Repo = "lllyasviel/Annotators", RepoPath = "ControlNetHED.pth",
                        TargetSubdir = "Vision/Annotators", Role = "transformer",
                        Sha256 = "5ca93762ffd68a29fee1af9d495bf6aab80ae86f08905fb35472a083a4c7a8fa" },
                },
            },
            new CatalogEntry
            {
                Id = "lineart", Modality = vis, DisplayName = "Lineart (sk_model realistic + sk_model2 coarse)", Architecture = "ResNet UNet generator", Status = ok,
                CliDrivable = true,
                Assets = new ModelAsset[]
                {
                    new() { Repo = "lllyasviel/Annotators", RepoPath = "sk_model.pth",
                        TargetSubdir = "Vision/Annotators", Role = "transformer",
                        Sha256 = "c686ced2a666b4850b4bb6ccf0748031c3eda9f822de73a34b8979970d90f0c6" },
                    new() { Repo = "lllyasviel/Annotators", RepoPath = "sk_model2.pth",
                        TargetSubdir = "Vision/Annotators", Role = "transformer-coarse",
                        Sha256 = "30a534781061f34e83bb9406b4335da4ff2616c95d22a585c1245aa8363e74e0" },
                },
            },
            new CatalogEntry
            {
                Id = "normalbae", Modality = vis, DisplayName = "NormalBAE (scannet NNET surface normals)", Architecture = "tf_efficientnet_b5_ap encoder + NNET decoder", Status = ok,
                CliDrivable = true,
                Assets = new ModelAsset[]
                {
                    new() { Repo = "lllyasviel/Annotators", RepoPath = "scannet.pt",
                        TargetSubdir = "Vision/Annotators", Role = "transformer",
                        Sha256 = "03dbf1600c51ee3d45c29f77b77bf1a3b7a24c3452dba62a4ae658f37330c209" },
                },
            },
            new CatalogEntry
            {
                Id = "upernet-seg", Modality = vis, DisplayName = "UperNet-Seg (ADE20K, ConvNeXt-Small)", Architecture = "ConvNeXt-Small + PSP/FPN head", Status = ok,
                CliDrivable = true,
                Assets = new ModelAsset[]
                {
                    new() { Repo = "openmmlab/upernet-convnext-small", RepoPath = "pytorch_model.bin",
                        TargetSubdir = "Vision/Annotators", TargetName = "upernet_convnext_small.bin", Role = "transformer",
                        Sha256 = "76c163aa531ab7edfb3a77bbcc039e340645aa0ffe2b0ffcfc68755f550c76ea" },
                },
            },

            // Video
            new CatalogEntry
            {
                Id = "ltx-video", Modality = vid, DisplayName = "LTX-Video", Architecture = "DiT + video VAE", Status = vp,
                CliDrivable = true,
                Assets = new ModelAsset[]
                {
                    // Single file bundles DiT + VAE; T5-XXL resolves as a side model (SideModels.T5XxlEnconly).
                    new() { Repo = "Lightricks/LTX-Video", RepoPath = "ltx-video-2b-v0.9.safetensors",
                        TargetSubdir = "Stable-Diffusion/LtxVideo", Role = "transformer",
                        Sha256 = "fb48c9fee3545631eeee6d039d45661a9ecc7a2eedf11cecd38ffca6eae0ae3b" },
                },
            },
            new CatalogEntry
            {
                Id = "wan", Modality = vid, DisplayName = "Wan 2.2 (T2V + I2V)", Architecture = "DiT + Wan VAE", Status = vp,
                CliDrivable = true,
                Assets = new ModelAsset[]
                {
                    // TI2V-5B; umT5-XXL + the Wan2.2 VAE resolve as side models inside WanVideoRecipe.
                    new() { Repo = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", RepoPath = "split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors",
                        TargetSubdir = "Stable-Diffusion/Wan", Role = "transformer",
                        Sha256 = "7057d12b745db48a79e449825f1ae26c75b14228148ea338fb94703452369555" },
                },
            },
            new CatalogEntry
            {
                Id = "lance-video", Modality = vid, DisplayName = "Lance (Video, T2V)", Architecture = "unified multimodal DiT", Status = vp,
                CliDrivable = true,
                Assets = new ModelAsset[]
                {
                    // Diffusers-style shard folder (ByteDance Lance_3B_Video); the Wan2.2 VAE resolves as a side
                    // model. Only the two files LanceVideoRecipe actually reads (the shard + tokenizer.json for the
                    // embedded byte-level BPE) — llm_config.json/vocab.json/merges.txt/generation_config.json ship
                    // in the repo but nothing in the loader path reads them.
                    new() { Repo = "bytedance-research/Lance", RepoPath = "Lance_3B_Video/model.safetensors",
                        TargetSubdir = "Stable-Diffusion/Lance/Lance_3B_Video", Role = "transformer",
                        Sha256 = "7f0550e1d1511b29a4740a67c1e18e176302a4ecb3177c8a5850ff5fe6447c25" },
                    new() { Repo = "bytedance-research/Lance", RepoPath = "Lance_3B_Video/tokenizer.json",
                        TargetSubdir = "Stable-Diffusion/Lance/Lance_3B_Video", Role = "tokenizer",
                        Sha256 = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539" },
                },
            },
            new CatalogEntry
            {
                Id = "kandinsky5-video", Modality = vid, DisplayName = "Kandinsky 5 Video", Architecture = "DiT (Qwen2.5-VL + CLIP)", Status = vp,
                CliDrivable = true,
                Assets = new ModelAsset[]
                {
                    // T2V-Lite-5s diffusers folder: transformer/ + a bundled vae/ that IS the shared HunyuanVideo 3D
                    // VAE in diffusers naming (Kandinsky5CheckpointConverter.LoadHunyuanVideoVae); Qwen2.5-VL-7B +
                    // CLIP-L resolve as side models inside Kandinsky5VideoRecipe.
                    new() { Repo = "kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers", RepoPath = "transformer/diffusion_pytorch_model.safetensors",
                        TargetSubdir = "Stable-Diffusion/Kandinsky5/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers/transformer", Role = "transformer", Sha256 = "9bd1cb1e67d07de19458b9ad288b906815411c68dad7910d042ceb66f61f9f44" },
                    new() { Repo = "kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers", RepoPath = "vae/diffusion_pytorch_model.safetensors",
                        TargetSubdir = "Stable-Diffusion/Kandinsky5/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers/vae", Role = "vae", Sha256 = "7c68a6295f9034a88225fbafb1f3258291a08d57a1fdb938233fa57b1b8f4883" },
                },
            },
            new CatalogEntry
            {
                Id = "hunyuan-video", Modality = vid, DisplayName = "HunyuanVideo 13B (T2V)", Architecture = "MMDiT + 3D causal VAE", Status = vp,
                CliDrivable = true,
                Assets = new ModelAsset[]
                {
                    // Comfy-Org repacked bf16 DiT; LLaVA-Llama-3-8B + CLIP-L + the 3D VAE resolve as side models.
                    new() { Repo = "Comfy-Org/HunyuanVideo_repackaged", RepoPath = "split_files/diffusion_models/hunyuan_video_t2v_720p_bf16.safetensors",
                        TargetSubdir = "Stable-Diffusion/HunyuanVideo", Role = "transformer", Sha256 = "c6ff2d107f0fec571fe276ad847468404ed01855c28c0be8859c3b311daec52a" },
                },
            },
            new CatalogEntry
            {
                Id = "ltx-2", Modality = vid, DisplayName = "LTX-2.3 (22B, video + audio)", Architecture = "dual-stream DiT + video/audio VAE", Status = vp,
                CliDrivable = true,
                Assets = new ModelAsset[]
                {
                    // Kijai's transformer-only split (not Lightricks' ~29GB bundled file) so this reuses the
                    // video/audio VAE + text-projection + Gemma-3-12B side models the engine already resolves —
                    // avoids re-downloading ~25GB of duplicated side-model weights.
                    new() { Repo = "Kijai/LTX2.3_comfy", RepoPath = "diffusion_models/ltx-2.3-22b-dev_transformer_only_fp8_scaled.safetensors",
                        TargetSubdir = "Stable-Diffusion/LtxVideo2", TargetName = "ltx-2.3-22b-dev-fp8.safetensors", Role = "transformer", Sha256 = null },
                },
            },
            // Cosmos-Predict1 Video2World — discrete-token autoregressive video continuation (T5-11B cross-attn +
            // DV8x16x16 tokenizer + AR backbone). Engine-only; run via the sample invocation in VideoCommand help.
            E("cosmos-predict1-5b-v2w", vid, "Cosmos-Predict1 5B Video2World", "AR discrete-token transformer", vp),
            E("cosmos-predict1-13b-v2w", vid, "Cosmos-Predict1 13B Video2World", "AR discrete-token transformer", vp),

            // 3D
            E("triposr", d3, "TripoSR", "triplane / NeRF", st, cli: true),
            E("hunyuan3d", d3, "Hunyuan3D-2 (Shape)", "Flux MMDiT + VecSet VAE", st, cli: true),
            E("trellis", d3, "TRELLIS (image-large)", "2-stage rectified flow + Gaussian splat", st, cli: true),

            // Interactive / world models
            E("hunyuan-gamecraft", act, "Hunyuan-GameCraft 1.0", "HunyuanVideo MM-DiT", vp),
            E("matrix-game-3", act, "Matrix-Game 3.0", "5B (+28B MoE) memory-augmented", vp),
            E("matrix-game-2", act, "Matrix-Game 2.0", "Wan2.1-lineage 1.8B", vp),
            E("oasis", act, "Oasis-500m", "axial-attention DiT", vp, cli: true),
            E("diamond", act, "DIAMOND (Atari-100k)", "EDM diffusion U-Net", vp, cli: true),
        };
    }
}
