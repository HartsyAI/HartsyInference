namespace HartsyInference.Engine;

/// <summary>Central registry of every text-encoder / VAE / shared side-model the per-architecture loaders may fetch on demand: one source of truth for canonical filename, download source, SHA-256, and target folder, so architectures that share a component reuse the same file instead of re-downloading.</summary>
public static class SideModels
{
    // ── Text encoders (folder = "Clip" — same convention as ComfyUI's text_encoders/) ──

    /// <summary>T5-XXL encoder-only fp8 (used by Flux.1 Dev, SD3 with T5, Chroma).</summary>
    public static readonly ModelAsset T5XxlEnconly = new ModelAsset
    {
        Repo = "mcmonkey/google_t5-v1_1-xxl_encoderonly",
        RepoPath = "t5xxl_fp8_e4m3fn.safetensors",
        TargetSubdir = "text_encoders",
        TargetName = "t5xxl_enconly.safetensors",
        Role = "text encoder",
        Sha256 = "7d330da4816157540d6bb7838bf63a0f02f573fc48ca4d8de34bb0cbfd514f09"
    };

    /// <summary>Pile-T5-XL — used by AuraFlow. Already bundled inside the AuraFlow checkpoint, so this is a fallback only; AuraFlowLoader doesn't auto-download it.</summary>
    public static readonly ModelAsset PileT5XlAuraFlow = new ModelAsset
    {
        Repo = "fal/AuraFlow-v0.2",
        RepoPath = "text_encoder/model.safetensors",
        TargetSubdir = "text_encoders",
        TargetName = "pile_t5xl_auraflow.safetensors",
        Role = "text encoder",
        Sha256 = "0a07449cf1141c0ec86e653c00465f6f0d79c6e58a2c60c8bcf4203d0e4ec4f6"
    };

    /// <summary>Pile-T5-XL SentencePiece tokenizer (32128 vocab, LLaMA-derived — same size as Google T5 v1.1 but
    /// different token-ID assignments). Required for AuraFlow conditioning: the embedded Google-T5 spiece
    /// denoises into a coherent image but not the prompted one (wrong token ids). Canonical EleutherAI source;
    /// byte-identical to the copy fal/AuraFlow-v0.3 ships under tokenizer/tokenizer.model.</summary>
    public static readonly ModelAsset PileT5XlSpiece = new ModelAsset
    {
        Repo = "EleutherAI/pile-t5-xl",
        RepoPath = "spiece.model",
        TargetSubdir = "Tokenizers/T5",
        TargetName = "pile_t5xl_spiece.model",
        Role = "tokenizer",
        Sha256 = "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347"
    };

    /// <summary>CLIP-L (SD1.5/SDXL/SD3/Flux text encoder).</summary>
    public static readonly ModelAsset ClipL = new ModelAsset
    {
        Repo = "stabilityai/stable-diffusion-xl-base-1.0",
        RepoPath = "text_encoder/model.fp16.safetensors",
        TargetSubdir = "text_encoders",
        TargetName = "clip_l.safetensors",
        Role = "text encoder",
        Sha256 = "660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd"
    };

    /// <summary>CLIP-G (SDXL / SD3 second text encoder).</summary>
    public static readonly ModelAsset ClipG = new ModelAsset
    {
        Repo = "stabilityai/stable-diffusion-xl-base-1.0",
        RepoPath = "text_encoder_2/model.fp16.safetensors",
        TargetSubdir = "text_encoders",
        TargetName = "clip_g.safetensors",
        Role = "text encoder",
        Sha256 = "ec310df2af79c318e24d20511b601a591ca8cd4f1fce1d8dff822a356bcdb1f4"
    };

    /// <summary>HiDream long-CLIP-L — standard SDXL CLIP-L architecture with HiDream-specific long-context fine-tuned weights (distinct from plain ClipL); canonical name matches Comfy's local name for file sharing.</summary>
    public static readonly ModelAsset HiDreamClipL = new ModelAsset
    {
        Repo = "Comfy-Org/HiDream-I1_ComfyUI",
        RepoPath = "split_files/text_encoders/clip_l_hidream.safetensors",
        TargetSubdir = "text_encoders",
        TargetName = "long_clip_l_hi_dream.safetensors",
        Role = "text encoder",
        Sha256 = "706fdb88e22e18177b207837c02f4b86a652abca0302821f2bfa24ac6aea4f71"
    };

    /// <summary>HiDream long-CLIP-G — standard SDXL CLIP-G architecture with HiDream-tuned weights.</summary>
    public static readonly ModelAsset HiDreamClipG = new ModelAsset
    {
        Repo = "Comfy-Org/HiDream-I1_ComfyUI",
        RepoPath = "split_files/text_encoders/clip_g_hidream.safetensors",
        TargetSubdir = "text_encoders",
        TargetName = "long_clip_g_hi_dream.safetensors",
        Role = "text encoder",
        Sha256 = "3771e70e36450e5199f30bad61a53faae85a2e02606974bcda0a6a573c0519d5"
    };

    /// <summary>Llama-3.1-8B-Instruct (fp8 scaled) — HiDream's fourth text encoder, run as a feature extractor harvesting hidden states from every layer; pairs with the embedded Llama-3.1 tokenizer.</summary>
    public static readonly ModelAsset Llama31_8B = new ModelAsset
    {
        Repo = "Comfy-Org/HiDream-I1_ComfyUI",
        RepoPath = "split_files/text_encoders/llama_3.1_8b_instruct_fp8_scaled.safetensors",
        TargetSubdir = "text_encoders",
        TargetName = "llama_3.1_8b_instruct_fp8_scaled.safetensors",
        Role = "text encoder",
        Sha256 = "9f86897bbeb933ef4fd06297740edb8dd962c94efcd92b373a11460c33765ea6"
    };

    /// <summary>Qwen2.5-VL-7B (fp8 scaled) — Qwen-Image's text encoder, run as a feature extractor; pairs with the embedded Qwen3 tokenizer (shared base BPE merges).</summary>
    public static readonly ModelAsset Qwen2_5_VL_7B = new ModelAsset
    {
        Repo = "Comfy-Org/Qwen-Image_ComfyUI",
        RepoPath = "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
        TargetSubdir = "text_encoders",
        TargetName = "qwen_2.5_vl_7b.safetensors",
        Role = "text encoder",
        Sha256 = "cb5636d852a0ea6a9075ab1bef496c0db7aef13c02350571e388aea959c5c0b4"
    };

    /// <summary>Qwen2.5-VL-3B (fp16) — OmniGen2's text encoder, run as a feature extractor (Comfy-Org repackage naming, this IS the 3B despite the generic filename); shares the Qwen3 tokenizer base BPE.</summary>
    public static readonly ModelAsset Qwen2_5_VL_3B = new ModelAsset
    {
        Repo = "Comfy-Org/Omnigen2_ComfyUI_repackaged",
        RepoPath = "split_files/text_encoders/qwen_2.5_vl_fp16.safetensors",
        TargetSubdir = "text_encoders",
        TargetName = "qwen_2.5_vl_fp16.safetensors",
        Role = "text encoder",
        Sha256 = "ba05dd266ad6a6aa90f7b2936e4e775d801fb233540585b43933647f8bc4fbc3"
    };

    /// <summary>Gemma-2-2B (fp16) — Lumina-Image-2.0's text encoder (hidden_states[-2] tap); Comfy-Org repackage naming for file sharing.</summary>
    public static readonly ModelAsset Gemma2_2B = new ModelAsset
    {
        Repo = "Comfy-Org/Lumina_Image_2.0_Repackaged",
        RepoPath = "split_files/text_encoders/gemma_2_2b_fp16.safetensors",
        TargetSubdir = "text_encoders",
        TargetName = "gemma_2_2b_fp16.safetensors",
        Role = "text encoder",
        Sha256 = "29761442862f8d064d3f854bb6fabf4379dcff511a7f6ba9405a00bd0f7e2dbd"
    };

    /// <summary>Qwen3-Embedding-0.6B — ACE-Step 1.5's style + lyric conditioner (1024-d states); hash unpinned (ships hash-less, downloader logs a no-verification warning until pinned).</summary>
    public static readonly ModelAsset Qwen3Embedding06B = new ModelAsset
    {
        Repo = "Qwen/Qwen3-Embedding-0.6B",
        RepoPath = "model.safetensors",
        TargetSubdir = "text_encoders",
        TargetName = "qwen3_embedding_0.6b.safetensors",
        Role = "text encoder",
        Sha256 = null
    };

    /// <summary>Qwen3-4B fp8-mixed — shared by Z-Image and Flux.2 Klein 4B (same file, both reuse it once downloaded).</summary>
    public static readonly ModelAsset Qwen3_4B = new ModelAsset
    {
        Repo = "Comfy-Org/z_image_turbo",
        RepoPath = "split_files/text_encoders/qwen_3_4b_fp8_mixed.safetensors",
        TargetSubdir = "text_encoders",
        TargetName = "qwen_3_4b.safetensors",
        Role = "text encoder",
        Sha256 = "72450b19758172c5a7273cf7de729d1c17e7f434a104a00167624cba94f68f15"
    };

    /// <summary>Qwen3-8B (fp4 mixed) for Flux.2 Klein 9B — Comfy's fp4-mixed quant; HartsyInference has no FP4 GEMM support yet so Flux2Loader refuses Klein 9B at runtime until FP4 lands or this points at an fp16/fp8 alternative.</summary>
    public static readonly ModelAsset Qwen3_8B_Fp4Mixed = new ModelAsset
    {
        Repo = "Comfy-Org/flux2-klein-9B",
        RepoPath = "split_files/text_encoders/qwen_3_8b_fp4mixed.safetensors",
        TargetSubdir = "text_encoders",
        TargetName = "qwen_3_8b.safetensors",
        Role = "text encoder",
        Sha256 = "bbf16f981d98e16d080c566134814c4e9f6aadd0d0e1383c60bc44ba939d760d"
    };

    /// <summary>Qwen3-VL-8B-Instruct (fp8_scaled) — Ideogram 4's sole text encoder (taps 13 decoder layers, rope θ=5e6); distinct from the base Qwen3-8B Flux.2 Klein uses, saved under Ideogram4/ to sit apart.</summary>
    public static readonly ModelAsset Qwen3VL_8B = new ModelAsset
    {
        Repo = "Comfy-Org/Ideogram-4",
        RepoPath = "text_encoders/qwen3vl_8b_fp8_scaled.safetensors",
        TargetSubdir = "text_encoders",
        TargetName = "Ideogram4/qwen3vl_8b_fp8_scaled.safetensors",
        Role = "text encoder",
        Sha256 = "4ba424cf62e51392e4d1a39933e803706f4e823c1065f36aaf149c6453f66bcd"
    };

    /// <summary>Qwen3-VL-4B (fp8_scaled) — Krea 2's text encoder (taps 12 decoder layers, rope θ=5e6); distinct from the base Qwen3-4B Z-Image / Flux.2 Klein 4B use, saved under Krea2/ to sit apart.</summary>
    public static readonly ModelAsset Qwen3VL_4B = new ModelAsset
    {
        Repo = "Comfy-Org/Krea-2",
        RepoPath = "text_encoders/qwen3vl_4b_fp8_scaled.safetensors",
        TargetSubdir = "text_encoders",
        TargetName = "Krea2/qwen3vl_4b_fp8_scaled.safetensors",
        Role = "text encoder",
        Sha256 = "54bd5144df0bbc25dd6ccadfcb826b521445a1b06ae5a42570bdd2974ca87094"
    };

    /// <summary>Mistral 3 Small (fp8) for Flux.2 Dev — MUST be the fp8 variant; the fp4_mixed file dequants with BOS-row activation outliers crushed, producing structured-noise images (was the Flux.2 Dev noise bug).</summary>
    public static readonly ModelAsset MistralSmallFlux2 = new ModelAsset
    {
        Repo = "Comfy-Org/flux2-dev",
        RepoPath = "split_files/text_encoders/mistral_3_small_flux2_fp8.safetensors",
        TargetSubdir = "text_encoders",
        TargetName = "mistral_3_small_flux2_fp8.safetensors",
        Role = "text encoder",
        Sha256 = "e3467b7d912a234fb929cdf215dc08efdb011810b44bc21081c4234cc75b370e"
    };

    /// <summary>Qwen2.5-VL-7B text stack for HunyuanImage 2.1 (fp8_scaled).</summary>
    public static readonly ModelAsset Qwen25Vl7BHunyuan = new ModelAsset
    {
        Repo = "Comfy-Org/HunyuanImage_2.1_ComfyUI",
        RepoPath = "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
        TargetSubdir = "text_encoders",
        TargetName = "qwen_2.5_vl_7b_fp8_scaled.safetensors",
        Role = "text encoder",
        Sha256 = "cb5636d852a0ea6a9075ab1bef496c0db7aef13c02350571e388aea959c5c0b4"
    };

    /// <summary>HunyuanImage 2.1 VAE (32× downscale, 64-channel latent).</summary>
    public static readonly ModelAsset HunyuanImageVae = new ModelAsset
    {
        Repo = "Comfy-Org/HunyuanImage_2.1_ComfyUI",
        RepoPath = "split_files/vae/hunyuan_image_2.1_vae_fp16.safetensors",
        TargetSubdir = "VAE",
        TargetName = "hunyuan_image_2.1_vae_fp16.safetensors",
        Role = "vae",
        Sha256 = "f2ae19863609206196b5e3a86bfd94f67bd3866f5042004e3994f07e3c93b2f9"
    };

    /// <summary>Ministral 3.3B for Ernie Image — HartsyInference has the encoder preset but no Ernie tokenizer yet.</summary>
    public static readonly ModelAsset Ministral_3_3B = new ModelAsset
    {
        Repo = "Comfy-Org/ERNIE-Image",
        RepoPath = "text_encoders/ministral-3-3b.safetensors",
        TargetSubdir = "text_encoders",
        TargetName = "ministral-3-3b.safetensors",
        Role = "text encoder",
        Sha256 = "49a750a128863854eac7d85e1a277a7b44bf6ec3646405b84686dfeeca3708ca"
    };

    /// <summary>GPT-OSS-20B (nvfp4) — Microsoft Lens' text encoder (MoE feature extractor, multi-layer tap); name/URL/hash match Comfy-Org/Lens for 13 GB file sharing, engine's NVFP4 codec dequants at load.</summary>
    public static readonly ModelAsset LensGptOss20b = new ModelAsset
    {
        Repo = "Comfy-Org/Lens",
        RepoPath = "text_encoders/gpt_oss_20b_nvfp4.safetensors",
        TargetSubdir = "text_encoders",
        TargetName = "gpt_oss_20b_nvfp4.safetensors",
        Role = "text encoder",
        Sha256 = "103d7759c720627e5ffdcb0d885595695085dad4201fa6a522a84d4b86335ca0"
    };

    /// <summary>UMT5-base — ACE-Step v1's style/genre text encoder (768-d hidden states); pairs with the embedded umT5 SentencePiece (256k vocab).</summary>
    public static readonly ModelAsset Umt5Base = new ModelAsset
    {
        Repo = "ACE-Step/ACE-Step-v1-3.5B",
        RepoPath = "umt5-base/model.safetensors",
        TargetSubdir = "text_encoders",
        TargetName = "umt5_base.safetensors",
        Role = "text encoder",
        Sha256 = "779cec0d210b2123e21d0a9cd8128f02b4d412627355028965a8be0b241cc3b6"
    };

    /// <summary>umT5-XXL (fp8 e4m3 scaled) — Wan-Video's text encoder; name/URL/hash match Comfy's GetUniMaxT5XXLModel() for file sharing, scale_weight tensors folded at load, pairs with the embedded umT5 SentencePiece (256k vocab, not base-T5 compatible).</summary>
    public static readonly ModelAsset Umt5Xxl = new ModelAsset
    {
        Repo = "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        RepoPath = "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        TargetSubdir = "text_encoders",
        TargetName = "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        Role = "text encoder",
        Sha256 = "c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68"
    };

    /// <summary>LLaVA-Llama-3-8B (fp8 scaled) — HunyuanVideo's primary text encoder (feature extractor, layer −3 through the diffusers prompt template); distinct from HiDream's <see cref="Llama31_8B"/> despite the shared Llama-3.1-8B backbone (different fine-tune). Comfy-Org's standard HunyuanVideo repack.</summary>
    public static readonly ModelAsset LlavaLlama3 = new ModelAsset
    {
        Repo = "Comfy-Org/HunyuanVideo_repackaged",
        RepoPath = "split_files/text_encoders/llava_llama3_fp8_scaled.safetensors",
        TargetSubdir = "text_encoders",
        Role = "text encoder",
        Sha256 = "2f0c3ad255c282cead3f078753af37d19099cafcfc8265bbbd511f133e7af250"
    };

    /// <summary>Gemma-3-12B-it (fp8 scaled) — LTX-2's text encoder when the checkpoint doesn't bundle the text tower; loaded raw (fp8-resident, run with CacheWeightCasts=off for 24 GB cards), Gemma tokenizer is a separate file.</summary>
    public static readonly ModelAsset GemmaLtx2 = new ModelAsset
    {
        Repo = "Comfy-Org/ltx-2",
        RepoPath = "split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors",
        TargetSubdir = "text_encoders",
        TargetName = "gemma_3_12B_it_fp8_scaled.safetensors",
        Role = "text encoder",
        Sha256 = "60216ce97c01c3a8753c2dfd0a89fc76e16fbe446d1a32ef8f1b528ac8bae466"
    };

    // ── VAEs (folder = "VAE") ──

    /// <summary>Qwen3-0.6B Base — Anima's text encoder; saved as canonical qwen_3_600m.safetensors to match Comfy's GetQwen3_600mModel() filename for file reuse.</summary>
    public static readonly ModelAsset Qwen3_0_6B = new ModelAsset
    {
        Repo = "circlestone-labs/Anima",
        RepoPath = "split_files/text_encoders/qwen_3_06b_base.safetensors",
        TargetSubdir = "text_encoders",
        TargetName = "qwen_3_600m.safetensors",
        Role = "text encoder",
        Sha256 = "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba"
    };

    /// <summary>Flux.1 VAE — shared by Flux.1, Z-Image, and Chroma.</summary>
    public static readonly ModelAsset FluxAe = new ModelAsset
    {
        Repo = "mcmonkey/swarm-vaes",
        RepoPath = "flux_ae.safetensors",
        TargetSubdir = "VAE",
        TargetName = "Flux/ae.safetensors",
        Role = "vae",
        Sha256 = "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38"
    };

    /// <summary>Flux.2 VAE (32-channel latent + BatchNorm stats) — distinct from Flux.1's ae, shared by Flux.2 and Ideogram 4; name/URL/hash match SwarmUI core's flux2-vae for cross-backend reuse.</summary>
    public static readonly ModelAsset Flux2Vae = new ModelAsset
    {
        Repo = "Comfy-Org/flux2-dev",
        RepoPath = "split_files/vae/flux2-vae.safetensors",
        TargetSubdir = "VAE",
        TargetName = "Flux/flux2-vae.safetensors",
        Role = "vae",
        Sha256 = "d64f3a68e1cc4f9f4e29b6e0da38a0204fe9a49f2d4053f0ec1fa1ca02f9c4b5"
    };

    /// <summary>Qwen-Image VAE — 16-channel autoencoder used by Anima and Qwen Image; hash matches SwarmUI's qwen-image-vae for file reuse.</summary>
    public static readonly ModelAsset QwenImageVae = new ModelAsset
    {
        Repo = "Comfy-Org/Qwen-Image_ComfyUI",
        RepoPath = "split_files/vae/qwen_image_vae.safetensors",
        TargetSubdir = "VAE",
        TargetName = "QwenImage/qwen_image_vae.safetensors",
        Role = "vae",
        Sha256 = "a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f"
    };

    /// <summary>Mage-Flow VAE — bespoke 128-channel /16 one-step-diffusion codec (MageVAE). Decoder only is used at
    /// inference; the <c>student.*</c> / <c>y_embedder.encoder|bottleneck</c> encoder side is skipped at load. Official
    /// microsoft/Mage-Flow repo ships bf16; no sha pinned yet.</summary>
    public static readonly ModelAsset MageVae = new ModelAsset
    {
        Repo = "microsoft/Mage-Flow",
        RepoPath = "vae/diffusion_pytorch_model.safetensors",
        TargetSubdir = "VAE",
        TargetName = "MageFlow/mage_vae.safetensors",
        Role = "vae",
    };

    /// <summary>SD3 / SD3.5 VAE — 16-channel autoencoder shared by every SD3.x variant; already diffusers-keyed, so it stages through <see cref="Features.LoaderVaeUtils.LoadFluxVaeF32"/> unchanged. Same mcmonkey/swarm-vaes repo as <see cref="FluxAe"/>, so a SwarmUI install shares the file.</summary>
    public static readonly ModelAsset Sd35Vae = new ModelAsset
    {
        Repo = "mcmonkey/swarm-vaes",
        RepoPath = "sd35_vae.safetensors",
        TargetSubdir = "VAE",
        TargetName = "SD3/sd35_vae.safetensors",
        Role = "vae",
        Sha256 = "6ad8546282f0f74d6a1184585f1c9fe6f1509f38f284e7c4f7ed578554209859"
    };

    // ── Diffusion transformers (folder = "Stable-Diffusion" — companion DiT weights) ──

    /// <summary>Ideogram 4's unconditional transformer — the second 9.3B DiT (identical architecture to the conditional one) for the asymmetric-CFG negative pass; auto-resolves the companion, fp8_scaled, saved under Ideogram4/.</summary>
    public static readonly ModelAsset Ideogram4Unconditional = new ModelAsset
    {
        Repo = "Comfy-Org/Ideogram-4",
        RepoPath = "diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors",
        TargetSubdir = "Stable-Diffusion",
        TargetName = "Ideogram4/ideogram4_unconditional_fp8_scaled.safetensors",
        Role = "transformer",
        Sha256 = "9b359007dae162cca7591d00868feea733eb7c56e56e3a214a4d5a9a2a07cd60"
    };

    /// <summary>ACE-Step v1 Music-DCAE (Sana-style mel autoencoder, decode side at inference); saved under VAE/AceStep/ as the audio analog of a VAE.</summary>
    public static readonly ModelAsset AceStepDcae = new ModelAsset
    {
        Repo = "ACE-Step/ACE-Step-v1-3.5B",
        RepoPath = "music_dcae_f8c8/diffusion_pytorch_model.safetensors",
        TargetSubdir = "VAE",
        TargetName = "AceStep/music_dcae_f8c8.safetensors",
        Role = "vae",
        Sha256 = "2b0cb469307ac50659d1880db2a99bae47d0df335cbb36853964662d4b80e8ee"
    };

    /// <summary>ACE-Step v1 ADaMoS HiFi-GAN vocoder (mel → 44.1 kHz waveform).</summary>
    public static readonly ModelAsset AceStepVocoder = new ModelAsset
    {
        Repo = "ACE-Step/ACE-Step-v1-3.5B",
        RepoPath = "music_vocoder/diffusion_pytorch_model.safetensors",
        TargetSubdir = "VAE",
        TargetName = "AceStep/music_vocoder.safetensors",
        Role = "vae",
        Sha256 = "c92c9b46e28ab7b37b777780cf4308ad7ddac869636bb77aa61599358c4bc1c0"
    };

    /// <summary>Wan2.2 VAE — 48-channel 3D video autoencoder used by Wan2.2 TI2V-5B; path/URL/hash match SwarmUI core's wan22-vae for cross-backend reuse.</summary>
    public static readonly ModelAsset Wan22Vae = new ModelAsset
    {
        Repo = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        RepoPath = "split_files/vae/wan2.2_vae.safetensors",
        TargetSubdir = "VAE",
        TargetName = "Wan/wan2.2_vae.safetensors",
        Role = "vae",
        Sha256 = "e40321bd36b9709991dae2530eb4ac303dd168276980d3e9bc4b6e2b75fed156"
    };

    /// <summary>Wan2.1 16-channel video VAE (8× spatial / 4× temporal) — shared by Wan2.1 1.3B/14B and Wan2.2 A14B; hash unpinned (TODO pin once verified).</summary>
    public static readonly ModelAsset Wan21Vae = new ModelAsset
    {
        Repo = "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        RepoPath = "split_files/vae/wan_2.1_vae.safetensors",
        TargetSubdir = "VAE",
        TargetName = "Wan/wan_2.1_vae.safetensors",
        Role = "vae",
        Sha256 = null
    };

    /// <summary>Wav2Vec2 speech feature extractor (chinese-base, hidden 768) for Wan2.2-S2V's audio front-end; URL/hash best-effort and UNVERIFIED (S2V validation-pending), loader refuses cleanly on failure.</summary>
    public static readonly ModelAsset Wav2Vec2Base = new ModelAsset
    {
        Repo = "TencentGameMate/chinese-wav2vec2-base",
        RepoPath = "model.safetensors",
        TargetSubdir = "text_encoders",
        TargetName = "Wav2Vec2/chinese-wav2vec2-base.safetensors",
        Role = "text encoder",
        Sha256 = null
    };

    /// <summary>wav2vec2-large (hidden 1024, 24 layers) for S2V checkpoints expecting 1024-dim audio features — the Comfy-Org Wan 2.2 repackage the S2V parity harness validated against; hash unpinned.</summary>
    public static readonly ModelAsset Wav2Vec2Large = new ModelAsset
    {
        Repo = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        RepoPath = "split_files/audio_encoders/wav2vec2_large_english_fp16.safetensors",
        TargetSubdir = "text_encoders",
        TargetName = "Wav2Vec2/wav2vec2_large_english_fp16.safetensors",
        Role = "text encoder",
        Sha256 = null
    };

    /// <summary>ACE-Step 1.5 Oobleck audio VAE (48 kHz stereo ↔ 64-ch 25 Hz latents); path/URL/hash match SwarmUI core's ace-step-15-vae for cross-backend reuse.</summary>
    public static readonly ModelAsset AceStep15Vae = new ModelAsset
    {
        Repo = "Comfy-Org/ace_step_1.5_ComfyUI_files",
        RepoPath = "split_files/vae/ace_1.5_vae.safetensors",
        TargetSubdir = "VAE",
        TargetName = "AceStep/ace_1.5_vae.safetensors",
        Role = "vae",
        Sha256 = "6de92e3a862acd287e08b024ac90f0783a8635451b728721a33ff03565bcb2bb"
    };

    /// <summary>HunyuanVideo 3D causal VAE (bf16) — Comfy-Org's standard HunyuanVideo repack; shared architecture with Kandinsky-5 Video's bundled diffusers VAE.</summary>
    public static readonly ModelAsset HunyuanVideoVae3D = new ModelAsset
    {
        Repo = "Comfy-Org/HunyuanVideo_repackaged",
        RepoPath = "split_files/vae/hunyuan_video_vae_bf16.safetensors",
        TargetSubdir = "Stable-Diffusion/HunyuanVideo",
        Role = "vae",
        Sha256 = "e8f8553275406d84ccf22e7a47601650d8f98bdb8aa9ccfdd6506b57a9701aed"
    };

    /// <summary>LTX-2.3 video VAE (bf16) — split out of the DiT file; path/URL/hash match SwarmUI core's ltx2-3-video-vae, auto-resolved when the DiT carries no bundled VAE.</summary>
    public static readonly ModelAsset Ltx23VideoVae = new ModelAsset
    {
        Repo = "Kijai/LTX2.3_comfy",
        RepoPath = "vae/LTX23_video_vae_bf16.safetensors",
        TargetSubdir = "VAE",
        TargetName = "LTX-2/LTX23_video_vae_bf16.safetensors",
        Role = "vae",
        Sha256 = "01ea62d09bc139f95c5dee7b5c062ad6a3e6cd8be910a1983ac02e7eb5b8ee3b"
    };

    /// <summary>LTX-2.3 audio VAE (bf16) — bundles the audio VAE and the BigVGAN vocoder the LTX-2 audio path needs; matches core's ltx2-3-audio-vae.</summary>
    public static readonly ModelAsset Ltx23AudioVae = new ModelAsset
    {
        Repo = "Kijai/LTX2.3_comfy",
        RepoPath = "vae/LTX23_audio_vae_bf16.safetensors",
        TargetSubdir = "VAE",
        TargetName = "LTX-2/LTX23_audio_vae_bf16.safetensors",
        Role = "vae",
        Sha256 = "5bc10fa4adecf99dda132d916e23048cbd56797702c5fa50eb5d2079048a38c3"
    };

    /// <summary>LTX-2.3 text-embedding-projection (bf16) — the connector weights split out of the DiT file; matches Comfy's GetLTX23TextProjectionClip() file.</summary>
    public static readonly ModelAsset Ltx23TextProjection = new ModelAsset
    {
        Repo = "Kijai/LTX2.3_comfy",
        RepoPath = "text_encoders/ltx-2.3_text_projection_bf16.safetensors",
        TargetSubdir = "text_encoders",
        TargetName = "LTX-2/ltx-2.3_text_projection_bf16.safetensors",
        Role = "text encoder",
        Sha256 = "911d59bb4cb7708179c9a0045ea0fe41212ecfb77aed3a02702b7c0a8274911f"
    };

    /// <summary>LTX-2.5 learned x2 spatial latent upsampler for the distilled two-stage flow (~950 MB BF16).</summary>
    public static readonly ModelAsset Ltx25LatentUpsampler = new ModelAsset
    {
        Repo = "Lightricks/LTX-2.5",
        RepoPath = "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
        TargetSubdir = "latent_upscale_models",
        TargetName = "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
        Role = "latent upsampler",
        Sha256 = "eb5a71fe4068ee87ccdb1c3aa635e547ca76bd2d30ae20ae889f2c325c0677e8"
    };

    // ── TAESD preview decoders (folder = "VAE", subfolder "Taesd/") ──

    /// <summary>TAESD SD 1.5 preview decoder (madebyollin, ~10 MB); hash intentionally empty (upstream publishes no canonical sha256, loader fails loudly on truncation).</summary>
    public static readonly ModelAsset TaesdSd15 = new ModelAsset
    {
        Repo = "madebyollin/taesd",
        RepoPath = "taesd_decoder.safetensors",
        TargetSubdir = "VAE",
        TargetName = "Taesd/sd_decoder.safetensors",
        Role = "vae",
        Sha256 = null
    };

    /// <summary>TAESD SDXL preview decoder (madebyollin, ~10 MB); hash intentionally empty (upstream publishes no canonical sha256).</summary>
    public static readonly ModelAsset TaesdSdxl = new ModelAsset
    {
        Repo = "madebyollin/taesdxl",
        RepoPath = "taesdxl_decoder.safetensors",
        TargetSubdir = "VAE",
        TargetName = "Taesd/sdxl_decoder.safetensors",
        Role = "vae",
        Sha256 = null
    };

    /// <summary>TAESD SD3 preview decoder (madebyollin, ~10 MB); hash intentionally empty (upstream publishes no canonical sha256).</summary>
    public static readonly ModelAsset TaesdSd3 = new ModelAsset
    {
        Repo = "madebyollin/taesd3",
        RepoPath = "taesd3_decoder.safetensors",
        TargetSubdir = "VAE",
        TargetName = "Taesd/sd3_decoder.safetensors",
        Role = "vae",
        Sha256 = null
    };

    /// <summary>TAESD Flux.1 preview decoder (madebyollin, ~10 MB); hash intentionally empty (upstream publishes no canonical sha256).</summary>
    public static readonly ModelAsset TaesdFlux = new ModelAsset
    {
        Repo = "madebyollin/taef1",
        RepoPath = "taef1_decoder.safetensors",
        TargetSubdir = "VAE",
        TargetName = "Taesd/flux_decoder.safetensors",
        Role = "vae",
        Sha256 = null
    };

    // ── CLIP-Vision (folder = "ClipVision") ──

    /// <summary>CLIP-ViT-H/14 image encoder used by IP-Adapter SDXL standard + Plus; auto-downloaded when IP-Adapter is enabled without an explicit ClipVisionModel, canonical version from the h94/IP-Adapter repo.</summary>
    public static readonly ModelAsset ClipVisionH14 = new ModelAsset
    {
        Repo = "h94/IP-Adapter",
        RepoPath = "models/image_encoder/model.safetensors",
        TargetSubdir = "ClipVision",
        TargetName = "clip-vision-h-14.safetensors",
        Role = "clip vision",
        Sha256 = "6ca9667da1ca9e0b0f75e46bb030f7e011f44f86cbfb8d5a36590fcd7507b030"
    };

    /// <summary>SigLIP so400m/14 384 vision tower (Comfy-Org packaging) — the image encoder FLUX.1 Redux consumes (729 × 1152 patch tokens); auto-downloaded when a style model is enabled without an explicit ClipVisionModel.</summary>
    public static readonly ModelAsset SigclipVision384 = new ModelAsset
    {
        Repo = "Comfy-Org/sigclip_vision_384",
        RepoPath = "sigclip_vision_patch14_384.safetensors",
        TargetSubdir = "ClipVision",
        TargetName = "sigclip_vision_patch14_384.safetensors",
        Role = "clip vision",
        Sha256 = "1fee501deabac72f0ed17610307d7131e3e9d1e838d0363aa3c2b97a6e03fb33"
    };
}
