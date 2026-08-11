using HartsyInference.Engine.Audio;

namespace HartsyInference.Engine.Registry;

/// <summary>Static, data-driven catalog of the models the engine can drive, mirroring the README support table.
/// Shared single source of truth: backs the CLI's <c>hartsy list</c>/REPL/shell-completion as well as the HTTP
/// API's <c>/admin/catalog</c>.</summary>
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
        const Modality emb = Modality.Embedding;
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
            // `hartsy text -m llama3/mistral` verified end-to-end 2026-07-22 (CLI catalog pass): both coherent
            // and correct on the 3060 (--low-vram-quant).
            new CatalogEntry
            {
                Id = "llama3", Modality = txt, DisplayName = "Llama-3.x", Architecture = "Llama dense transformer",
                Status = ok, CliDrivable = true,
                Assets = new ModelAsset[]
                {
                    new() { Repo = "hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF", RepoPath = "llama-3.2-1b-instruct-q4_k_m.gguf",
                        TargetSubdir = "LLM/llama3", Role = "transformer",
                        Sha256 = "1d0e9419ec4e12aef73ccf4ffd122703e94c48344a96bc7c5f0f2772c2152ce3" },
                },
            },
            new CatalogEntry
            {
                Id = "mistral", Modality = txt, DisplayName = "Mistral (dense)", Architecture = "Mistral dense transformer",
                Status = ok, CliDrivable = true,
                Assets = new ModelAsset[]
                {
                    new() { Repo = "bartowski/Mistral-7B-Instruct-v0.3-GGUF", RepoPath = "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
                        TargetSubdir = "LLM/mistral", Role = "transformer",
                        Sha256 = "1270d22c0fbb3d092fb725d4d96c457b7b687a5f5a715abe1e818da303e562b6" },
                },
            },
            E("gguf", txt, "Quantized GGUF (Q4/Q8)", "config-driven, any GGUF LLM", ok, cli: true),
            // Qwen3.5/3.6 Gated-DeltaNet hybrid (SSM path, Qwen35Model). `qwen35` = dense 0.8B, verified 2026-07-27
            // for BOTH text and vision (its mmproj sidecar downloads alongside → `hartsy text -m qwen35 -i img.png`
            // captions/OCRs correctly). See MODEL_STATUS_LLM.md + docs/Design/TRENDING_MODELS_PHASED_PLAN.md.
            new CatalogEntry
            {
                Id = "qwen35", Modality = txt, DisplayName = "Qwen3.5-VL (0.8B, dense hybrid)",
                Architecture = "Gated-DeltaNet + full-attn hybrid (+Qwen3-VL vision)", Status = ok, CliDrivable = true,
                Assets = new ModelAsset[]
                {
                    new() { Repo = "unsloth/Qwen3.5-0.8B-GGUF", RepoPath = "Qwen3.5-0.8B-Q4_K_M.gguf",
                        TargetSubdir = "LLM/qwen35", Role = "transformer",
                        Sha256 = "bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517" },
                    new() { Repo = "unsloth/Qwen3.5-0.8B-GGUF", RepoPath = "mmproj-F16.gguf",
                        TargetSubdir = "LLM/qwen35", Role = "mmproj" },
                },
            },
            // `qwen35moe` = the 35B-A3B MoE tier (256 experts, top-8 softmax + shared expert; same hybrid attention).
            // Implemented 2026-07-27; code + dense/VL regressions pass, but the smallest quant (~10.7GB IQ2_XXS) OOMs
            // the dev box — Status ValidationPending until a real-weight run on a larger GPU (see the GPU harness).
            new CatalogEntry
            {
                Id = "qwen35moe", Modality = txt, DisplayName = "Qwen3.5-VL-MoE (35B-A3B)",
                Architecture = "hybrid attn + 256-expert MoE FFN + shared expert (+vision)", Status = vp, CliDrivable = true,
                Assets = new ModelAsset[]
                {
                    new() { Repo = "unsloth/Qwen3.5-35B-A3B-GGUF", RepoPath = "Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf",
                        TargetSubdir = "LLM/qwen35moe", Role = "transformer" },
                    new() { Repo = "unsloth/Qwen3.5-35B-A3B-GGUF", RepoPath = "mmproj-F16.gguf",
                        TargetSubdir = "LLM/qwen35moe", Role = "mmproj" },
                },
            },

            // Text / LLM — dense families. `hartsy text -m <id>` verified end-to-end 2026-07-22 (CLI catalog
            // pass) on the 3060 (--low-vram-quant) unless noted; see MODEL_STATUS_LLM.md for the exact
            // prompt/output per model and PARITY_VERIFICATION.md for the full session record.
            new CatalogEntry
            {
                Id = "gemma", Modality = txt, DisplayName = "Gemma 2 / Gemma 3 (text)",
                Architecture = "GeGLU, √d embed scale, sandwich norm, dual-RoPE, attn/final logit soft-cap",
                Status = ok, CliDrivable = true,
                Assets = new ModelAsset[]
                {
                    new() { Repo = "ggml-org/gemma-3-4b-it-GGUF", RepoPath = "gemma-3-4b-it-Q4_K_M.gguf",
                        TargetSubdir = "LLM/gemma", Role = "transformer",
                        Sha256 = "882e8d2db44dc554fb0ea5077cb7e4bc49e7342a1f0da57901c0802ea21a0863" },
                },
            },
            new CatalogEntry
            {
                Id = "phi", Modality = txt, DisplayName = "Phi-3 / Phi-3.5-mini / Phi-4-mini",
                Architecture = "fused QKV split, LongRoPE, partial rotary, gpt-4o tokenizer",
                Status = ok, CliDrivable = true,
                Assets = new ModelAsset[]
                {
                    new() { Repo = "microsoft/Phi-3-mini-4k-instruct-gguf", RepoPath = "Phi-3-mini-4k-instruct-q4.gguf",
                        TargetSubdir = "LLM/phi", Role = "transformer",
                        Sha256 = "8a83c7fb9049a9b2e92266fa7ad04933bb53aa1e85136b7b30f1b8000ff2edef" },
                },
            },
            new CatalogEntry
            {
                // Factual prompt ("capital of Italy" -> "Rome") correct and coherent; a separate arithmetic
                // prompt got a rambling wrong answer — a real 1.6B-model capability limit (confirmed the
                // template/generation pipeline itself is correct via the factual prompt), not an engine bug.
                Id = "stablelm2", Modality = txt, DisplayName = "StableLM-2 (1.6B)", Architecture = "partial rotary + QKV bias",
                Status = ok, CliDrivable = true,
                Assets = new ModelAsset[]
                {
                    // The base (non-chat) repo has no usable chat template for this CLI's single-turn prompting —
                    // use the chat-tuned repack.
                    new() { Repo = "brittlewis12/stablelm-2-1_6b-chat-GGUF", RepoPath = "stablelm-2-1_6b-chat.Q4_K_M.gguf",
                        TargetSubdir = "LLM/stablelm2", Role = "transformer",
                        Sha256 = "a465c9a5a8e0afdef7f29e6dd0066dce7d8aeea484c3a9e11cf21478df4f435d" },
                },
            },
            new CatalogEntry
            {
                Id = "granite3", Modality = txt, DisplayName = "Granite-3",
                Architecture = "embedding/attention/residual/logit scalar multipliers", Status = ok, CliDrivable = true,
                Assets = new ModelAsset[]
                {
                    new() { Repo = "bartowski/granite-3.0-2b-instruct-GGUF", RepoPath = "granite-3.0-2b-instruct-Q4_K_M.gguf",
                        TargetSubdir = "LLM/granite3", Role = "transformer",
                        Sha256 = "06e5f1fdb176a2d79a2f33febfbbd521935fc0c00573326e91439b7d21e89df0" },
                },
            },
            // NOTE: antares-1b (fine-tune of IBM Granite-4.0-1B) was trialed here and REVERTED. Its GGUF reports
            // general.architecture="granite" with purely dense tensors (attn_q/k/v/output + ffn_gate/up/down, no
            // ssm_*/expert tensors), so it loads through the granite path — but generation is GARBAGE on both the
            // quant-resident and full-precision paths. Granite-4.0 diverges from the Granite-3-validated path in
            // something subtler (most likely tokenizer.ggml.pre pre-tokenizer and/or rope config). Needs real parity
            // work before it can be catalogued. See docs/Design/TRENDING_MODELS_PHASED_PLAN.md (Phase 0).
            // command-r: a real, ungated Q3_K_S source was confirmed and downloads/parses fine, but generation
            // genuinely OOM'd on this box — kernel-killed at ~48.7GB resident RAM (confirmed via dmesg), above
            // even the loader's own 2.5x-file-size headroom estimate (~40GB) for this 15.9GB file; real peak was
            // closer to 3.1x. This is a 35B dense model — Status stays Structural (never actually completed a
            // generation) pending a host with >64GB RAM. Assets included since the source is real and correct.
            new CatalogEntry
            {
                Id = "command-r", Modality = txt, DisplayName = "Cohere Command-R",
                Architecture = "LayerNorm, parallel residual, interleaved RoPE, NoPE global layers, logit-scale",
                Status = st, CliDrivable = true,
                Assets = new ModelAsset[]
                {
                    new() { Repo = "bartowski/c4ai-command-r-v01-GGUF", RepoPath = "c4ai-command-r-v01-Q3_K_S.gguf",
                        TargetSubdir = "LLM/command-r", Role = "transformer",
                        Sha256 = "3d039ff1205bec9294d70d4e3cfedac50c0cad5738e5429a1772a40573c21917" },
                },
            },
            new CatalogEntry
            {
                Id = "olmoe", Modality = txt, DisplayName = "OLMoE (1B-7B-0924)", Architecture = "MoE, whole-vector Q/K norm",
                Status = ok, CliDrivable = true,
                Assets = new ModelAsset[]
                {
                    new() { Repo = "allenai/OLMoE-1B-7B-0924-Instruct-GGUF", RepoPath = "olmoe-1b-7b-0924-instruct-q4_k_m.gguf",
                        TargetSubdir = "LLM/olmoe", Role = "transformer",
                        Sha256 = "8c310f1435a1222338fd2d3d974975be9cd908180b644bab0c2a94da1ac32f3f" },
                },
            },
            new CatalogEntry
            {
                Id = "granite-moe", Modality = txt, DisplayName = "Granite-MoE", Architecture = "scalar multipliers + MoE",
                Status = ok, CliDrivable = true,
                Assets = new ModelAsset[]
                {
                    new() { Repo = "bartowski/granite-3.0-1b-a400m-instruct-GGUF", RepoPath = "granite-3.0-1b-a400m-instruct-Q4_K_M.gguf",
                        TargetSubdir = "LLM/granite-moe", Role = "transformer",
                        Sha256 = "074f09e13484e54e73c93830d34e9fa9917a6319fb8bae762a22594b9b4da0dc" },
                },
            },
            new CatalogEntry
            {
                // Correct output but notably slow (~15s) for its size with GPU sm% near 0% throughout the run —
                // inconclusive whether that's decode-phase host-glue in the PLE (per-layer-embedding) gather path
                // or just one-time load/setup overhead dominating a very short (~1 token) completion; worth a
                // longer-generation re-check if Gemma-4 perf becomes a priority. See MODEL_STATUS_LLM.md.
                Id = "gemma4", Modality = txt, DisplayName = "Gemma-4 (E2B/E4B mobile)",
                Architecture = "per-layer embeddings, per-layer head-dim, cross-layer KV donor sharing, weightless V-norm",
                Status = ok, CliDrivable = true,
                Assets = new ModelAsset[]
                {
                    new() { Repo = "unsloth/gemma-4-E2B-it-GGUF", RepoPath = "gemma-4-E2B-it-Q4_K_M.gguf",
                        TargetSubdir = "LLM/gemma4", Role = "transformer",
                        Sha256 = "740185b21d22ceb83a11c3aa62ad5842ef32c70f6096d756bbee85a1e4ec34b8" },
                },
            },

            // Text / LLM — MoE / large dense: real ungated sources exist for all six (HF-API-confirmed
            // 2026-07-22), but none reach a coherent CLI-verified generation on this box's hardware (24GB 4090 /
            // 62GB RAM) — each hit a genuinely distinct blocker, not a uniform "too big". No Assets on any of
            // these: pointing `hartsy text -m <id>` at a source that's confirmed to fail here would be worse than
            // no default. Full detail + exact repro commands in MODEL_STATUS_LLM.md. TODO: re-verify all six once
            // a larger-VRAM/RAM box (rented GPU) is available.
            E("mixtral", txt, "Mixtral 8x7B", "llama+experts MoE, interleaved RoPE, renorm — 47B total params. " +
                "TODO(rented GPU): real Q4_K_M source confirmed (TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF) but its " +
                "~26GB VRAM footprint exceeds this 24GB 4090; not downloaded (disk budget spent on the other five).", st, cli: true),
            E("qwen3-moe", txt, "Qwen3-MoE (30B-A3B / 235B)", "per-head Q/K norm, no shared expert. " +
                "TODO(rented GPU): official Qwen/Qwen3-30B-A3B-GGUF Q4_K_M (18.6GB) downloaded and loaded, but " +
                "CUDA_ERROR_OUT_OF_MEMORY during weight preload on the 24GB 4090 even with --low-vram-quant — " +
                "confirmed doesn't fit this card at Q4.", st, cli: true),
            E("deepseek-v2-lite", txt, "DeepSeek-V2-Lite", "MLA + DeepSeek-MoE. " +
                "TODO(needs debugging, not just a bigger GPU): mradermacher/DeepSeek-V2-Lite-GGUF Q4_K_M (10.4GB) " +
                "loads and generates on the 4090 without crashing, but output is incoherent garbage " +
                "(\"\\\" ! +—————— »\\n...\\n- = 20.\\n##\") — a real MLA-attention-path bug, not root-caused.", st, cli: true),
            E("deepseek-v3", txt, "DeepSeek-V3 (671B) / Kimi-K2 (1T)", "MLA + MoE + sigmoid group-routing + q-LoRA query. " +
                "TODO(rented multi-GPU): real ungated GGUF sources confirmed to exist (bullerwins/DeepSeek-V3-GGUF, " +
                "unsloth/DeepSeek-V3.1-GGUF) but 671B/1T params is not attemptable on any single-box config this " +
                "project has access to at any quant; not downloaded.", st, cli: true),
            E("gpt-oss", txt, "GPT-OSS (20B / 120B)", "attention sinks, MoE, o200k tokenizer. " +
                "MXFP4 decode support ADDED 2026-07-22 (DType.MXFP4 + Codec_MXFP4, verified bit-for-bit against " +
                "upstream ggml source; routes through the existing CPU-dequant fallback, no new CUDA kernel needed) " +
                "— the 'Unsupported GGUF tensor type: 39' load crash is fixed. Still BLOCKED on this box: every " +
                "available public GGUF (native ggml-org/gpt-oss-20b-GGUF MXFP4 release and unsloth's \"Q4_K_M\" " +
                "repack alike, which keeps the MoE expert tensors in MXFP4 regardless of the label) exceeds this " +
                "hardware's VRAM at any quant — a genuine size limit now, not an engine gap. See MODEL_STATUS_LLM.md.", st, cli: true),
            E("gemma4-moe", txt, "Gemma-4 (31B-dense / 26B-A4B-MoE)", "parallel dense+MoE branch, per-layer FFN width. " +
                "Root-caused + fixed 2026-07-22: unsloth/gemma-4-26B-A4B-it-GGUF Q3_K_M (12.7GB) fuses the MoE " +
                "gate+up expert projections into one ffn_gate_up_exps tensor (llama.cpp LLM_TENSOR_FFN_GATE_UP_EXPS) " +
                "instead of separate ffn_gate_exps/ffn_up_exps, which Gemma4KeyMapper had no case for at all — " +
                "confirmed via a partial HTTP range-fetch of the real checkpoint's header, not guessed. Fixed: " +
                "Gemma4KeyMapper maps the fused tensor, GgufLanguageModel.SplitStackedExperts gained a fused-split " +
                "path (gate=first half/up=second half of each expert's row-block, matches ggml build_moe_ffn's view " +
                "split), regression-tested. The original KeyNotFoundException is gone. Still BLOCKED: exceeds this " +
                "box's VRAM (12.7GB > 3060, tight on the 4090 alongside everything else), AND the same checkpoint " +
                "carries a ffn_down_exps.scale per-expert post-matmul correction (ggml's build_lora_mm_id w_s) that " +
                "MoeFeedForward has no support for yet — deliberately left unmapped rather than half-wired; expect " +
                "numerics to still be off by a per-expert scale factor even once VRAM allows a run. See " +
                "MODEL_STATUS_LLM.md.", st, cli: true),

            // Text / LLM — vision-language (VLM). Attach an image with `--image <path>`. `hartsy text -m <id> -i
            // <img>` verified 2026-07-22 against tests/HartsyInference.Vision.Tests/TestData/bus.png (a real photo
            // of a person and an electric "cero emisiones" bus at a street crossing) on the 4090. 4 of 5 produce
            // genuinely grounded (not hallucinated) descriptions — see MODEL_STATUS_LLM.md for exact output per
            // model and how each was judged.
            new CatalogEntry
            {
                // PARTIAL: correctly identifies bus/street/people/trees (the real subjects) but gets the bus color
                // wrong (reports white; it's blue) and does not reliably stop at content end — confirmed the
                // model's own eos token (128009) IS registered in StopIds (not a stop-token wiring bug), the
                // checkpoint itself sometimes just doesn't emit it within a normal token budget and free-runs into
                // a hallucinated follow-up turn with leaked special tokens. Lower --max-tokens (~35) truncates
                // cleanly. Status stays Structural pending a cleaner-terminating checkpoint or generation-side fix.
                Id = "llama32-vision", Modality = txt, DisplayName = "Llama-3.2-Vision-11B (mllama)",
                Architecture = "gated cross-attention VLM, own 560px ViT (no token splice)", Status = st, CliDrivable = true,
                Assets = new ModelAsset[]
                {
                    new() { Repo = "leafspark/Llama-3.2-11B-Vision-Instruct-GGUF", RepoPath = "Llama-3.2-11B-Vision-Instruct.Q4_K_M.gguf",
                        TargetSubdir = "LLM/llama32-vision", Role = "transformer",
                        Sha256 = "652e85aa1e14c9087a4ccc3ab516fb794cbcf152f8b4b8d3c0b828da4ada62d9" },
                    new() { Repo = "leafspark/Llama-3.2-11B-Vision-Instruct-GGUF", RepoPath = "Llama-3.2-11B-Vision-Instruct-mmproj.f16.gguf",
                        TargetSubdir = "LLM/llama32-vision", Role = "mmproj",
                        Sha256 = "622429e8d31810962dd984bc98559e706db2fb1d40e99cb073beb7148d909d73" },
                },
            },
            // gemma3-vision: FAIL, confirmed real and deeply diagnosed, narrowed further 2026-07-22 but still NOT
            // resolved. Consistently and confidently describes an unrelated street/market scene (originally
            // "Barcelona", now "Madrid" after the fix below — a real change, not the same failure) instead of
            // bus.png's actual content. A from-scratch PyTorch reference replay of the vision tower + projector
            // (patch-embed -> 27 ViT blocks -> post-LN -> avg-pool -> RMSNorm -> linear) against this exact
            // checkpoint's real weights matched the C# SiglipVlmEncoder at cosine similarity 0.99996-1.0 at every
            // stage — the vision math itself is numerically correct, ruling that out. Cross-checked against
            // smolvlm2/llava15/qwen25-vl (different VLM families, same --image pipeline, same bus.png) which all
            // produce genuinely grounded output — rules out a systemic --image bug.
            // FOUND + FIXED 2026-07-22 (real bug, confirmed against HF transformers source, not guessed):
            // MultimodalGenerator.Generate applied the Gemma-3 √hidden embedding normalizer to the SPLICED IMAGE
            // embeddings too. HF's Gemma3TextScaledWordEmbedding bakes that scale into the TOKEN embedding lookup
            // only (`super().forward(input_ids) * embed_scale`); Gemma3Model.forward calls that scaled lookup
            // THEN `inputs_embeds.masked_scatter(image_mask, image_features)`, overwriting the image positions
            // with the RAW (unscaled) vision-projector output — so the image embeddings were being amplified by
            // ~50.6x (sqrt(2560)) beyond what the decoder was ever trained on. Every other VLM family has
            // EmbeddingScale=1.0 so this bug could only ever affect gemma3 — consistent with it being the only
            // family that failed. Fixed: image embeddings now spliced in raw. Re-verified live against bus.png —
            // STILL HALLUCINATES (different wrong content, so the fix is real, just not sufficient alone).
            // FOUND, NOT FIXED (real new-engine-capability work, out of scope for this pass): Gemma-3's real
            // attention mask is NOT pure-causal across a multimodal sequence — HF's
            // get_block_sequence_ids_for_mask/create_masks_for_vision_model give image tokens BIDIRECTIONAL
            // attention to every token in the same contiguous image block (OR'd with the ordinary causal mask
            // for everything else); GenericTransformer's decode path only ever passes a boolean `causal: true`
            // to FlashAttention, with no mechanism for a mixed causal/blockwise mask at all. This is a strong,
            // evidence-based lead for the remaining hallucination (each image token currently can't attend to
            // LATER image tokens in the LLM's own cross-sequence attention, on top of whatever's inside the
            // vision tower's own self-contained attention), but implementing it needs a real masked-attention
            // capability this engine's LLM decode path doesn't have yet — not a quick fix. No Assets — don't
            // want `hartsy text -m gemma3-vision` defaulting to a source still confirmed to hallucinate.
            E("gemma3-vision", txt, "Gemma-3-4B-vision", "SigLIP + avg-pool/RMSNorm/Linear projector — FAIL, see comment above and MODEL_STATUS_LLM.md", st, cli: true),
            new CatalogEntry
            {
                Id = "smolvlm2", Modality = txt, DisplayName = "SmolVLM2-2.2B",
                Architecture = "SigLIP + idefics3 pixel-shuffle projector", Status = ok, CliDrivable = true,
                Assets = new ModelAsset[]
                {
                    new() { Repo = "ggml-org/SmolVLM2-2.2B-Instruct-GGUF", RepoPath = "SmolVLM2-2.2B-Instruct-Q4_K_M.gguf",
                        TargetSubdir = "LLM/smolvlm2", Role = "transformer",
                        Sha256 = "0cf76814555b8665149075b74ab6b5c1d428ea1d3d01c1918c12012e8d7c9f58" },
                    new() { Repo = "ggml-org/SmolVLM2-2.2B-Instruct-GGUF", RepoPath = "mmproj-SmolVLM2-2.2B-Instruct-f16.gguf",
                        TargetSubdir = "LLM/smolvlm2", Role = "mmproj",
                        Sha256 = "db9a3a1648cab1ebc3af4a2b0c8145dd8faebf6f7dd7b16e7dc1842229f14ac4" },
                },
            },
            new CatalogEntry
            {
                Id = "llava15", Modality = txt, DisplayName = "LLaVA-1.5-7B",
                Architecture = "CLIP ViT (CLS token, pre-LN, quick-GELU) + MLP projector", Status = ok, CliDrivable = true,
                Assets = new ModelAsset[]
                {
                    new() { Repo = "second-state/Llava-v1.5-7B-GGUF", RepoPath = "llava-v1.5-7b-Q4_K_M.gguf",
                        TargetSubdir = "LLM/llava15", Role = "transformer",
                        Sha256 = "2687b20ac8b7a23f6c70296d5b1e7f908fef2ce4769ecdebd1bb9503528a75bf" },
                    new() { Repo = "second-state/Llava-v1.5-7B-GGUF", RepoPath = "llava-v1.5-7b-mmproj-model-f16.gguf",
                        TargetSubdir = "LLM/llava15", Role = "mmproj",
                        Sha256 = "50da4e5b0a011615f77686f9b02613571e65d23083c225e107c08c3b1775d9b1" },
                },
            },
            new CatalogEntry
            {
                Id = "llava16", Modality = txt, DisplayName = "LLaVA-NeXT (1.6) Vicuna-7B",
                Architecture = "CLIP ViT (same tower as LLaVA-1.5) + MLP projector, anyres dynamic tiling " +
                    "(select_best_resolution -> resize+pad -> tile) with pack_image_features merge " +
                    "(reshape/unpad/image_newline, base tile first)", Status = ok, CliDrivable = true,
                Assets = new ModelAsset[]
                {
                    new() { Repo = "cjpais/llava-v1.6-vicuna-7b-gguf", RepoPath = "llava-v1.6-vicuna-7b.Q4_K_M.gguf",
                        TargetSubdir = "LLM/llava16", Role = "transformer",
                        Sha256 = "bc3360111db647026db732a528aa5c4dbcb1525e12a58f08acf7edbc46cd488c" },
                    new() { Repo = "cjpais/llava-v1.6-vicuna-7b-gguf", RepoPath = "mmproj-model-f16.gguf",
                        TargetSubdir = "LLM/llava16", Role = "mmproj",
                        Sha256 = "d1b9981bda94fb0bdddad5ff60f84bb226ec29ade04a6474e0168185d659357d" },
                },
            },
            new CatalogEntry
            {
                // Best VLM result of the pass: correctly read the actual "cero emisiones" text painted on the
                // bus in the test photo, not just the broad scene.
                Id = "qwen25-vl", Modality = txt, DisplayName = "Qwen2.5-VL (3B / 7B)",
                Architecture = "own ViT, Conv3D patch embed, 2D-RoPE, window attention", Status = ok, CliDrivable = true,
                Assets = new ModelAsset[]
                {
                    new() { Repo = "unsloth/Qwen2.5-VL-7B-Instruct-GGUF", RepoPath = "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
                        TargetSubdir = "LLM/qwen25-vl", Role = "transformer",
                        Sha256 = "d16776dcd9a28d42758c2958ed3a752aabf20a305252cd64ff2be72b4a78c503" },
                    new() { Repo = "unsloth/Qwen2.5-VL-7B-Instruct-GGUF", RepoPath = "mmproj-F16.gguf",
                        TargetSubdir = "LLM/qwen25-vl", Role = "mmproj",
                        Sha256 = "987dd0733033fb5dd9b124d1ca926ae865572e432384eee7618b2eec3e735e17" },
                },
            },

            // Text / LLM — non-transformer / hybrid decoders (Ssm/*Model.cs, not GenericTransformer). All three
            // dispatch through SsmGenerationPipeline, not TextGenerationPipeline — confirmed working 2026-07-22.
            new CatalogEntry
            {
                // Correct, coherent output, stops cleanly — but ~0.8 tok/s (26.6s for ~20 tokens) with GPU sm%
                // staying at 2-14% the ENTIRE run, unlike the transformer families' periodic 30%+ decode bursts.
                // Root cause (confirmed by direct code read, not inference): Mamba1Model/Mamba2Model run their
                // causal Conv1d AND the actual selective-scan recurrence host-side EVERY layer, by design (own
                // doc comment: "the recurrence is inherently sequential... run host-side") — GPU handles only the
                // big linear projections. This is the real "host glue, not using the GPU" pattern to flag: a
                // legitimate architectural tradeoff (a fused parallel-scan CUDA kernel is substantial dedicated
                // engineering, same class of justified exception as diffusion's CPU-side interleaved RoPE), not
                // an oversight bug — but a genuine, measured perf cost worth a future dedicated kernel if Mamba
                // throughput matters. See MODEL_STATUS_LLM.md / PARITY_VERIFICATION.md for the numbers.
                Id = "mamba", Modality = txt, DisplayName = "Mamba-1 / Mamba-2, Falcon-Mamba",
                Architecture = "selective state-space scan + causal Conv1d, no attention", Status = ok, CliDrivable = true,
                Assets = new ModelAsset[]
                {
                    new() { Repo = "dranger003/mamba-2.8b-hf-GGUF", RepoPath = "ggml-mamba-2.8b-hf-q4_k.gguf",
                        TargetSubdir = "LLM/mamba", Role = "transformer",
                        Sha256 = "0fb59e6e6c46503dda44e983a17fb84f7707d9b32e9f704c38a0696cc516324c" },
                },
            },
            // rwkv: REAL BUG FOUND + FIXED (GgufLanguageModel.cs BuildTemplate — some llama.cpp GGUF converters,
            // RWKV-World among them, store a bare format-name sentinel like "rwkv-world" in tokenizer.chat_template
            // instead of real Jinja source; JinjaChatTemplate "compiled" it as literal text with nothing to
            // substitute, so every render silently produced that same constant regardless of the actual prompt —
            // confirmed via 3 different prompts all giving byte-identical output before the fix). Fixed by
            // detecting templates with no {{/{% syntax and routing straight to ChatML. Prompt-sensitivity is
            // restored and verified, but the downloadable RWKV-6-World checkpoint is a base multilingual model,
            // not ChatML-instruction-tuned, so its answers are still often wrong against a ChatML-formatted
            // prompt — Status stays Structural (the engine bug is fixed; a real end-to-end pass needs an
            // instruction-tuned RWKV GGUF with its own real chat template, not this one). Same shared-recurrence
            // host-glue tradeoff as mamba applies here too (RwkvModel/Rwkv7Model doc comments).
            E("rwkv", txt, "RWKV-6 / RWKV-7", "WKV recurrence + data-dependent token-shift LoRA + GroupNorm — chat-template " +
                "sentinel bug found+fixed (GgufLanguageModel.BuildTemplate); needs an instruct-tuned checkpoint for a full pass", st, cli: true),
            new CatalogEntry
            {
                // PARTIAL: the Gated-DeltaNet/attention hybrid forward pass itself is verified correct (coherent,
                // correct answer in ALL of --thinking/--no-thinking/neither, all three routed through a ChatML
                // fallback). --thinking specifically against THIS model's real Jinja template remains unverified:
                // fixing a real 3-part-slice parser gap (messages[::-1], see JinjaExpr.cs SliceExpr) exposed a
                // SEPARATE, deeper "Unsupported Jinja call expression" failure in the template's tool-calling
                // filter chain (args_value | tojson | safe if args_value is mapping ... ), not yet root-caused.
                // A new render-time ChatML fallback (JinjaChatTemplate.cs Encode) means this degrades gracefully
                // (logged [WRN], not a crash) rather than failing generation outright.
                Id = "qwen35", Modality = txt, DisplayName = "Qwen3.5 (Gated DeltaNet hybrid)",
                Architecture = "every 4th layer is GQA+RoPE, the rest are Gated DeltaNet delta-rule recurrence",
                Status = st, CliDrivable = true,
                Assets = new ModelAsset[]
                {
                    new() { Repo = "unsloth/Qwen3.5-0.8B-GGUF", RepoPath = "Qwen3.5-0.8B-Q4_K_M.gguf",
                        TargetSubdir = "LLM/qwen35", Role = "transformer",
                        Sha256 = "bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517" },
                },
            },

            new CatalogEntry
            {
                // NOT reachable via `hartsy text` today: TextService.LoadInto only routes a GGUF to the
                // decoder-only GgufLanguageModel path or (for SSM architectures) SsmLanguageModel — there is no
                // seq2seq branch that calls the standalone Seq2Seq/T5Model.cs loader, so a T5/FLAN-T5 GGUF would
                // load as (and fail as) a decoder-only transformer. T5Model itself is verified encoder+decoder
                // parity (cosine = 1.0 vs HF) per MODEL_STATUS_LLM.md, but that's exercised directly in the
                // Diffusion package's text-encoder tests, not through TextService/hartsy text. Listed for
                // discoverability; CliDrivable stays false until TextService gains a seq2seq generation path.
                // Gap reconfirmed 2026-07-22 by direct grep of TextService.cs: zero Seq2Seq/T5Model references.
                Id = "t5", Modality = txt, DisplayName = "T5 / FLAN-T5", Architecture = "encoder-decoder, rel-pos bias, cross-attention, GeGLU",
                Status = st, CliDrivable = false,
            },

            // Text / LLM — architectures with a real IGgufKeyMapper (GgufKeyMapperRegistry) but no bring-up/e2e
            // run documented anywhere in this repo's checklists, unlike every family above. `gpt2`/`starcoder2`
            // are now CLI-verified 2026-07-22; `glm4` surfaced exactly the class of bug this comment predicted.
            new CatalogEntry
            {
                // TWO real, distinct bugs found. (1) FIXED 2026-07-22: legraphista/glm-4-9b-chat-GGUF declares
                // general.architecture=chatglm — the OLDER llama.cpp lineage, genuinely different from glm4 per
                // Glm4KeyMapper's own doc comment. No IGgufKeyMapper is registered for chatglm, so it fell back to
                // key-heuristic detection, which produced wrong tensor byte offsets and crashed with
                // ArgumentOutOfRangeException in GgufLoader.GetTensor. GgufModelLoader.Load now catches this class
                // of failure (any declared-but-unregistered architecture, not chatglm-specific) and throws a clear
                // UnsupportedModelException naming the declared arch instead — a wrong-architecture GGUF is still
                // unusable, but no longer an opaque crash. (2) PARTIALLY FIXED 2026-07-22 (follow-up session): the
                // CORRECT-architecture checkpoint's incoherent output was ONE real bug — a missing partial-rotary
                // guard in the interleaved/GPT-J RoPE path (ApplyRopeInterleaved / lm_rope_interleaved_f32); glm4
                // rotates only the first headDim/2 dims per HF/llama.cpp source but the kernel rotated the FULL
                // headDim. Fixed (kernel + launcher + CudaBackend + IBackend default + both GenericTransformer
                // call sites; regression test in RopeInterleavedPartialRotaryTests.cs) and confirmed via real
                // generation: open-ended prompts (a binary-search explanation, a 250+ token story) are now fully
                // coherent. BUT a SECOND, still-open bug remains: any prompt requiring exact retrieval of a
                // number from the user's own prompt fails ("What is 9 times 9?", "I have 12 apples...", "List
                // 1 through 5" all produce confused non-answers that drop the actual digits), while
                // llama-cpp-python running the byte-identical GGUF answers all three correctly — so this is a
                // real engine bug, not a checkpoint/quantization limitation. Not yet root-caused; leading
                // suspects are the extreme GQA ratio (2 kv-heads/32 q-heads) or the RoPE pairing convention
                // itself. See MODEL_STATUS_LLM.md for the full investigation and what's been ruled out.
                Id = "glm4", Modality = txt, DisplayName = "GLM-4", Architecture = "sandwich norm, fused gate/up projection (Glm4KeyMapper) — FAIL, see comment above",
                Status = st, CliDrivable = true,
            },
            new CatalogEntry
            {
                Id = "gpt2", Modality = txt, DisplayName = "GPT-2 / BLOOM / GPT-NeoX",
                Architecture = "absolute position embeddings, non-gated GELU (Gpt2KeyMapper)", Status = ok, CliDrivable = true,
                Assets = new ModelAsset[]
                {
                    new() { Repo = "RichardErkhov/openai-community_-_gpt2-medium-gguf", RepoPath = "gpt2-medium.Q4_K_M.gguf",
                        TargetSubdir = "LLM/gpt2", Role = "transformer",
                        Sha256 = "0dc1490576067f1683dc257b517e4a56a99816638fe8575161b00fa2753e6382" },
                },
            },
            new CatalogEntry
            {
                Id = "starcoder2", Modality = txt, DisplayName = "StarCoder2",
                Architecture = "shares the llama-family key-mapper path (LlamaKeyMapper)", Status = ok, CliDrivable = true,
                Assets = new ModelAsset[]
                {
                    new() { Repo = "second-state/StarCoder2-3B-GGUF", RepoPath = "starcoder2-3b-Q4_K_M.gguf",
                        TargetSubdir = "LLM/starcoder2", Role = "transformer",
                        Sha256 = "d8fb39287a463549b80d97473b0a7595c3a5a6da3ae2604ca33906a1a43f7175" },
                },
            },

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
            // Mage-Flow (Microsoft, 4B NR-MMDiT). Reuses the Qwen-Image DiT (MageFlow preset) + Qwen3-VL-4B encoder;
            // bespoke 128-ch one-step MageVAE. Status ValidationPending — built + compiles but only GPU-verifiable
            // (~8.5B bf16; run an fp8/GGUF DiT via --model-path on a 12GB card). Official transformer is bf16.
            new CatalogEntry
            {
                Id = "mage-flow",
                Modality = img,
                DisplayName = "Mage-Flow (T2I)",
                Architecture = "NR-MMDiT dual-stream (Qwen3-VL-4B + MageVAE)",
                Status = vp,
                CliDrivable = true,
                Assets = new ModelAsset[]
                {
                    new() { Repo = "microsoft/Mage-Flow", RepoPath = "transformer/diffusion_pytorch_model.safetensors",
                        TargetSubdir = "Stable-Diffusion/MageFlow", Role = "transformer" },
                    SideModels.Qwen3VL_4B,
                    SideModels.MageVae,
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
                // Bare `-m distilwhisper` resolves to distil-large-v3.5 (same shape as v3 — the .5 is a
                // longer-trained release), recognized by WhisperPipeline.InferConfig since 2026-07-24.
                CliDrivable = true, // `hartsy transcribe -m distilwhisper` (or :v3/:v2/:medium/:small) — SttCatalog "distilwhisper"
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
                Id = "vibevoice:realtime", Modality = tts, DisplayName = "VibeVoice-Realtime-0.5B", Architecture = "split-LM diffusion TTS", Status = ok,
                CliDrivable = true, // `hartsy speak -m vibevoice:realtime --reference <wav>` — single-speaker, low-latency
                // streaming variant, architecturally distinct from VibeVoice 1.5B/7B (split 4-layer text encoder +
                // 20-layer TTS backbone, binary EOS classifier, no lm_head at all). Zero-shot voice cloning from a
                // reference clip is a HartsyInference-side adaptation, not upstream's own mechanism — see
                // VibeVoiceStreamingPipeline's class doc.
                Assets = new ModelAsset[]
                {
                    new() { Repo = "microsoft/VibeVoice-Realtime-0.5B", RepoPath = "model.safetensors", TargetSubdir = "Audio/VibeVoiceRealtime", Role = "transformer" },
                    new() { Repo = "microsoft/VibeVoice-Realtime-0.5B", RepoPath = "config.json", TargetSubdir = "Audio/VibeVoiceRealtime", Role = "config" },
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
                        TargetSubdir = "rtdetr", Role = "transformer",
                        Sha256 = "fe87a5a30f5daf298d10794c7682a63b6107986f97d6a770ba948d89e4340093" },
                },
            },
            new CatalogEntry
            {
                Id = "grounding-dino", Modality = vis, DisplayName = "Grounding DINO (tiny)", Architecture = "Swin-T + BERT + deformable cross-modal encoder/decoder", Status = ok,
                CliDrivable = true,
                Assets = new ModelAsset[]
                {
                    new() { Repo = "IDEA-Research/grounding-dino-tiny", RepoPath = "model.safetensors",
                        TargetSubdir = "grounding-dino", Role = "transformer",
                        Sha256 = "1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3" },
                    new() { Repo = "IDEA-Research/grounding-dino-tiny", RepoPath = "vocab.txt",
                        TargetSubdir = "grounding-dino", Role = "tokenizer",
                        Sha256 = "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3" },
                },
            },
            new CatalogEntry
            {
                Id = "clipseg", Modality = vis, DisplayName = "ClipSeg (rd64-refined)", Architecture = "CLIP + lightweight segmentation decoder", Status = ok,
                CliDrivable = true,
                Assets = new ModelAsset[]
                {
                    new() { Repo = "CIDAS/clipseg-rd64-refined", RepoPath = "model.safetensors",
                        TargetSubdir = "clipseg", Role = "transformer",
                        Sha256 = "d00ca85d6b859f9d07b7cfb8ef26fe9771cb275b34c9368f2ecf603139307f55" },
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
                        TargetSubdir = "sam2", Role = "transformer",
                        Sha256 = "65b50056e05bcb13694174f51bb6da89c894b57b75ccdf0ba6352c597c5d1125" },
                },
            },
            // "RetinaFace" was never actually built — the real, real-weight-verified face detector is a
            // YOLOv8-Face port (see MODEL_STATUS_VISION.md). Catalog id corrected to match; blocked on the same
            // pickle-graph issue as YOLO8/11 (every HF mirror checked 2026-07-22 — jaredthejelly/, arnabdhar/,
            // etc. — ships only the raw Ultralytics .pt, never a folded safetensors).
            E("yolov8-face", vis, "YOLOv8-Face + landmarks", "face detection + 5-pt landmarks", vp),
            // ArcFace: upstream (InsightFace buffalo_l) ships ONNX only — no HF mirror hosts a safetensors/flat
            // state_dict export either (checked 2026-07-22). The engine's loaders read safetensors and flat
            // pickle state_dicts, not ONNX graphs, so this needs an ONNX reader, not a pickle-VM extension.
            // Architecture is already real-weight parity-verified (corr 1.0) via the project's own offline
            // convert_arcface_onnx.py — that conversion just isn't something the Assets auto-download system
            // can perform, only a one-time local step (see MODEL_STATUS_VISION.md).
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
            // Gated on HF (requires accepting terms); the user's account already has access (used real-weight
            // parity-verified 2026-07-01, corr 1.00000000 vs upstream — see MODEL_STATUS_VISION.md), so the
            // existing HF token downloads it directly, no ungated repack needed.
            new CatalogEntry
            {
                Id = "rmbg", Modality = vis, DisplayName = "RMBG-1.4 (BriaRMBG / ISNet background removal)", Architecture = "U²-Net-style nested-U RSU encoder/decoder", Status = ok,
                CliDrivable = true,
                Assets = new ModelAsset[]
                {
                    new() { Repo = "briaai/RMBG-1.4", RepoPath = "model.safetensors",
                        TargetSubdir = "Vision/Rmbg", Role = "transformer",
                        Sha256 = "46ef7fe46f2ae284d8f1aaa24bfa5fca5ef25a34e2c7caa890a0029eb100e87f" },
                },
            },

            // Restoration (SeedVR2). numz/SeedVR2_comfyUI ships the original state-dict keys verbatim, so
            // the converter loads its fp16 files directly. The pos/neg embeddings exist upstream only as
            // torch-pickle .pt, hence the Hartsy-hosted converted copy (until published, place it under
            // Models/Video/SeedVr2/).
            new CatalogEntry
            {
                Id = "seedvr2-3b", Modality = Modality.Restore, DisplayName = "SeedVR2-3B (video/image restoration)",
                Architecture = "NaDiT (windowed MM-DiT) + s8c16t4 causal video VAE, one-step", Status = vp,
                CliDrivable = true,
                Assets = new ModelAsset[]
                {
                    new() { Repo = "numz/SeedVR2_comfyUI", RepoPath = "seedvr2_ema_3b_fp16.safetensors",
                        TargetSubdir = "Video/SeedVr2", Role = "transformer",
                        Sha256 = "2fd0e03a3dad24e07086750360727ca437de4ecd456f769856e960ae93e2b304" },
                    new() { Repo = "numz/SeedVR2_comfyUI", RepoPath = "ema_vae_fp16.safetensors",
                        TargetSubdir = "Video/SeedVr2", Role = "vae",
                        Sha256 = "20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1" },
                    new() { Repo = "HartsyAI/SeedVR2-safetensors", RepoPath = "seedvr2_embeddings.safetensors",
                        TargetSubdir = "Video/SeedVr2", Role = "embeddings" },
                },
            },
            new CatalogEntry
            {
                Id = "seedvr2-7b", Modality = Modality.Restore, DisplayName = "SeedVR2-7B (video/image restoration)",
                Architecture = "NaDiT (windowed MM-DiT) + s8c16t4 causal video VAE, one-step", Status = vp,
                CliDrivable = true,
                Assets = new ModelAsset[]
                {
                    new() { Repo = "numz/SeedVR2_comfyUI", RepoPath = "seedvr2_ema_7b_fp16.safetensors",
                        TargetSubdir = "Video/SeedVr2", Role = "transformer" },
                    new() { Repo = "numz/SeedVR2_comfyUI", RepoPath = "ema_vae_fp16.safetensors",
                        TargetSubdir = "Video/SeedVr2", Role = "vae",
                        Sha256 = "20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1" },
                    new() { Repo = "HartsyAI/SeedVR2-safetensors", RepoPath = "seedvr2_embeddings.safetensors",
                        TargetSubdir = "Video/SeedVr2", Role = "embeddings" },
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
                        Sha256 = "456f901338bd9eadbded3828b819109a9b68e8a525ca5cf8d0049a69fcfeca1e" },
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
                        TargetSubdir = "Stable-Diffusion/LtxVideo2", TargetName = "ltx-2.3-22b-dev-fp8.safetensors", Role = "transformer",
                        Sha256 = "f0b8f92f8a33daf7e7f508878db6d6f25f9c8bc7b90fcb5650fb402a9d9e9f13" },
                },
            },
            // Cosmos-Predict1 Video2World — discrete-token autoregressive video continuation (T5-11B cross-attn +
            // DV8x16x16 tokenizer + AR backbone). Engine-only; run via the sample invocation in VideoCommand help.
            E("cosmos-predict1-5b-v2w", vid, "Cosmos-Predict1 5B Video2World", "AR discrete-token transformer", vp),
            E("cosmos-predict1-13b-v2w", vid, "Cosmos-Predict1 13B Video2World", "AR discrete-token transformer", vp),
            new CatalogEntry
            {
                // The fp8 DiT is the production build: fully resident on a 24 GB card, against ~90 s/step for the
                // 66 GB bf16. LoRA is the one thing it cannot do — a merge has to rewrite weights in full precision —
                // so point --model-path at 'minimax_h3_fl2va_pruned_bf16.safetensors' (40 GB, same repo) for that.
                // Ref2VA is a SEPARATE checkpoint in this repo ('minimax_h3_ref2va_*'), not a mode of this one.
                Id = "minimax-h3", Modality = vid, DisplayName = "MiniMax-H3 (omni, video + stereo audio)",
                Architecture = "H3-Omni packed-token DiT + ViT3D video VAE + DAC/BigVGAN audio VAE", Status = vp,
                CliDrivable = true,
                Assets = new ModelAsset[]
                {
                    new() { Repo = "Comfy-Org/MiniMax-H3", RepoPath = "diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
                        TargetSubdir = "Stable-Diffusion/MiniMaxH3/flat/diffusion_models", Role = "transformer",
                        Sha256 = "12944c1f7791637e7de12208aef04da82bd26b95271b1b47d817364315ade993" },
                    new() { Repo = "Comfy-Org/MiniMax-H3", RepoPath = "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                        TargetSubdir = "Stable-Diffusion/MiniMaxH3/flat/text_encoders", Role = "text encoder",
                        Sha256 = "35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6" },
                    new() { Repo = "Comfy-Org/MiniMax-H3", RepoPath = "vae/minimax_h3_video_vae_fp16.safetensors",
                        TargetSubdir = "Stable-Diffusion/MiniMaxH3/flat/vae/MiniMaxH3", Role = "vae",
                        Sha256 = "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522" },
                    // Hash from the repo's own LFS metadata, unlike the three above: the copy staged on the dev box is
                    // the vendor MiniMaxAI original (a byte-different container of the same weights), so this one file
                    // has not been verified locally end-to-end.
                    new() { Repo = "Comfy-Org/MiniMax-H3", RepoPath = "vae/minimax_h3_audio_vae_fp32.safetensors",
                        TargetSubdir = "Stable-Diffusion/MiniMaxH3/flat/vae/MiniMaxH3", Role = "vae",
                        Sha256 = "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48" },
                },
            },

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

            // Text embeddings — decoder-LLM-backed only (DecoderEmbeddingModel wraps the same verified
            // GgufLanguageModel/GenericTransformer GGUF pipeline chat models use). Not CLI-drivable yet: reachable
            // only via POST /v1/native/embeddings and the OpenAI-compat /v1/embeddings. BERT-family encoders
            // (bge/gte/nomic) are a separate, later scope — see EmbeddingService's class doc.
            new CatalogEntry
            {
                // Verified 2026-07-22 against a real transformers.AutoModel reference on the exact same token
                // ids: full 1024-dim cosine similarity = 1.000000 (see DecoderEmbeddingRealCheckpointTests).
                // f16, NOT the Q8_0 quant this repo also ships: Q8_0 hits a real, pre-existing gap where the CPU
                // backend's generic Linear/CastTo path can't dequantize block-quantized weights on its own
                // (throws "requires a dedicated dequantizer. Use GgufDequantizer instead.") — a broader Engine
                // limitation affecting quantized-GGUF inference on the CPU backend generally, not something
                // introduced by or specific to embeddings. Out of scope to fix here; f16 sidesteps it entirely.
                Id = "qwen3-embedding", Modality = emb, DisplayName = "Qwen3-Embedding-0.6B", Architecture = "Qwen3 decoder + last-token pooling",
                Status = ok, CliDrivable = false,
                Assets = new ModelAsset[]
                {
                    new() { Repo = "Qwen/Qwen3-Embedding-0.6B-GGUF", RepoPath = "Qwen3-Embedding-0.6B-f16.gguf",
                        TargetSubdir = "Embedding/qwen3-embedding", Role = "transformer",
                        Sha256 = "421a27e58d165478cc7acb984a688c2aa41404968b0203e7cd743ece44c54340" },
                },
            },
        };
    }
}
