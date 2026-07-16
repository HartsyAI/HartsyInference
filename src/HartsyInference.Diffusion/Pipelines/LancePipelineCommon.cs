using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>Shared sequence-packing / sampling helpers for the Lance image and video pipelines, replicating the upstream T2I/T2V inference path (<c>validation_gen</c> + <c>ValidationDataset</c> + Qwen2.5-VL <c>get_rope_index</c>). Covers the chat-templated packed sequence (text splits causal, one bidirectional noise block carrying the VAE tokens), the 3-D M-RoPE positions, the sparse attention mask, the frozen latent-position-table indices, the logit-normal-shifted timestep grid, and the interval-gated global-renorm CFG combine.</summary>
public static unsafe class LancePipelineCommon
{
    /// <summary>Additive-mask value for blocked attention (finite so F16 SDPA paths stay NaN-free).</summary>
    private const float MaskBlocked = -1e9f;

    /// <summary>Logit-normal (SD3-style) shifted timestep grid from 1→0, length steps+1.</summary>
    public static float[] BuildShiftedTimesteps(int steps, float shift)
    {
        float[] t = new float[steps + 1];
        for (int i = 0; i <= steps; i++)
        {
            float lin = 1.0f - (float)i / steps;
            t[i] = shift * lin / (1.0f + (shift - 1.0f) * lin);
        }
        return t;
    }

    /// <summary>Builds the packed generation sequence for one prompt: <c>prefix | caption | mid | &lt;|vision_start|&gt; | pads×nVae | &lt;|vision_end|&gt; [| &lt;|im_end|&gt;]</c>. Text (incl. the vision sentinels) is the und stream; the pads are the gen stream. Position ids replicate Qwen2.5-VL <c>get_rope_index</c> (text 1-D sequential; vision block 3-D anchored at the first pad index, temporal step <see cref="LanceConfig.TokensPerSecond"/>; trailing text resumes at max+1). The attention mask replicates <c>create_sparse_mask</c>: causal text, bidirectional noise block, and nothing outside the block ever attends the noise tokens.</summary>
    public static LanceSequence BuildGenSequence(LancePromptTemplate template, int[] captionTokens,
        LanceConfig config, int gridT, int gridH, int gridW)
    {
        if (gridH > config.MaxLatentSize || gridW > config.MaxLatentSize)
            throw new ArgumentException($"Latent grid {gridH}x{gridW} exceeds max_latent_size {config.MaxLatentSize}.");
        int nVae = gridT * gridH * gridW;
        int[] prefix = template.PrefixTokens, mid = template.MidTokens;
        int nPre = prefix.Length + captionTokens.Length + mid.Length + 1;   // + <|vision_start|>
        int nTail = template.TrailingEos ? 2 : 1;                            // <|vision_end|> [+ <|im_end|>]
        int seq = nPre + nVae + nTail;
        int nText = nPre + nTail;

        int[] textIds = new int[nText];
        int[] undIdx = new int[nText];
        int cursor = 0;
        foreach (int id in prefix) { textIds[cursor] = id; undIdx[cursor] = cursor; cursor++; }
        foreach (int id in captionTokens) { textIds[cursor] = id; undIdx[cursor] = cursor; cursor++; }
        foreach (int id in mid) { textIds[cursor] = id; undIdx[cursor] = cursor; cursor++; }
        textIds[cursor] = config.VisionStartTokenId; undIdx[cursor] = cursor; cursor++;
        textIds[cursor] = config.VisionEndTokenId; undIdx[cursor] = nPre + nVae;
        if (template.TrailingEos)
        {
            cursor++;
            textIds[cursor] = config.EosTokenId; undIdx[cursor] = nPre + nVae + 1;
        }

        int[] genIdx = new int[nVae];
        for (int i = 0; i < nVae; i++) genIdx[i] = nPre + i;

        // ── get_rope_index replica ──
        Tensor pos = new Tensor(new TensorShape(seq, 3), DType.F32);
        float* p = (float*)pos.DataPointer;
        for (int i = 0; i < nPre; i++) { p[i * 3 + 0] = i; p[i * 3 + 1] = i; p[i * 3 + 2] = i; }
        int step = config.TokensPerSecond;
        int idx = 0;
        for (int ti = 0; ti < gridT; ti++)
            for (int hi = 0; hi < gridH; hi++)
                for (int wi = 0; wi < gridW; wi++)
                {
                    long off = (long)(nPre + idx) * 3;
                    p[off + 0] = nPre + ti * step;
                    p[off + 1] = nPre + hi;
                    p[off + 2] = nPre + wi;
                    idx++;
                }
        int blockMax = nPre + Math.Max((gridT - 1) * step, Math.Max(gridH, gridW) - 1);
        for (int i = 0; i < nTail; i++)
        {
            long off = (long)(nPre + nVae + i) * 3;
            float v = blockMax + 1 + i;
            p[off + 0] = v; p[off + 1] = v; p[off + 2] = v;
        }

        // ── create_sparse_mask replica: allow = (q ≥ kv AND kv not noise) OR (q, kv both noise).
        // The noise split spans <|vision_start|> + pads + <|vision_end|> (upstream split_lens put both
        // sentinels inside the bidirectional block, hidden from every token outside it). ──
        Tensor mask = new Tensor(new TensorShape(1, 1, seq, seq), DType.F32);
        float* m = (float*)mask.DataPointer;
        int noiseStart = nPre - 1, noiseEnd = nPre + nVae + 1;   // [start, end)
        for (int q = 0; q < seq; q++)
        {
            bool qNoise = q >= noiseStart && q < noiseEnd;
            long row = (long)q * seq;
            for (int kv = 0; kv < seq; kv++)
            {
                bool kvNoise = kv >= noiseStart && kv < noiseEnd;
                bool allow = (q >= kv && !kvNoise) || (qNoise && kvNoise);
                m[row + kv] = allow ? 0f : MaskBlocked;
            }
        }

        return new LanceSequence
        {
            TextTokenIds = textIds,
            UndIdx = undIdx,
            GenIdx = genIdx,
            PositionIds = pos,
            AttentionMask = mask,
        };
    }

    /// <summary>Row indices into the checkpoint's frozen <c>latent_pos_embed</c> table for a (t,h,w) latent grid: <c>t·max² + h·max + w</c> (upstream <c>get_flattened_position_ids_extrapolate_video</c>).</summary>
    public static int[] BuildLatentPositionIds(int gridT, int gridH, int gridW, int maxLatentSize)
    {
        int[] ids = new int[gridT * gridH * gridW];
        int idx = 0;
        for (int ti = 0; ti < gridT; ti++)
            for (int hi = 0; hi < gridH; hi++)
                for (int wi = 0; wi < gridW; wi++)
                    ids[idx++] = ti * maxLatentSize * maxLatentSize + hi * maxLatentSize + wi;
        return ids;
    }

    /// <summary>Folds interval-gated CFG with global renorm into <paramref name="cond"/> in place (upstream <c>validation_gen</c>): <c>v = u + cfg·(c − u)</c>, then <c>v *= clamp(‖c‖/‖v‖, renormMin, 1)</c> — the renorm pulls the amplified velocity back to the conditional's magnitude.</summary>
    public static void CfgRenormGlobalInPlace(Tensor cond, Tensor uncond, float cfg, float renormMin)
    {
        long n = cond.Shape.ElementCount;
        float* c = (float*)cond.DataPointer;
        float* u = (float*)uncond.DataPointer;
        double normCondSq = 0;
        for (long i = 0; i < n; i++) normCondSq += (double)c[i] * c[i];
        double normCfgSq = 0;
        for (long i = 0; i < n; i++)
        {
            c[i] = u[i] + cfg * (c[i] - u[i]);
            normCfgSq += (double)c[i] * c[i];
        }
        float scale = (float)Math.Clamp(Math.Sqrt(normCondSq) / (Math.Sqrt(normCfgSq) + 1e-8), renormMin, 1.0);
        for (long i = 0; i < n; i++) c[i] *= scale;
    }

    /// <summary>In-place Euler step <c>z −= v·dt</c>.</summary>
    public static void EulerStep(Tensor latents, Tensor velocity, float dt)
    {
        long n = latents.Shape.ElementCount;
        float* z = (float*)latents.DataPointer;
        float* v = (float*)velocity.DataPointer;
        for (long i = 0; i < n; i++) z[i] -= v[i] * dt;
    }

    /// <summary>In-place 2-way text-CFG Euler step: <c>z -= (uncond + cfg·(cond − uncond))·dt</c>. Shared by the Wan/LTX/Hunyuan video pipelines.</summary>
    public static void EulerCfgStep(Tensor latents, Tensor cond, Tensor uncond, float cfg, float dt)
    {
        long n = latents.Shape.ElementCount;
        float* z = (float*)latents.DataPointer;
        float* c = (float*)cond.DataPointer;
        float* u = (float*)uncond.DataPointer;
        for (long i = 0; i < n; i++)
        {
            float v = u[i] + cfg * (c[i] - u[i]);
            z[i] -= v * dt;
        }
    }

    /// <summary>Folds 2-way CFG into <paramref name="cond"/> in place: <c>cond ← uncond + cfg·(cond − uncond)</c>.
    /// Use this to produce the single combined velocity that a stateful scheduler (e.g.
    /// <c>FlowUniPCMultistepScheduler</c>) steps with, instead of <see cref="EulerCfgStep"/> which combines and steps
    /// in one pass.</summary>
    public static void CfgCombineInPlace(Tensor cond, Tensor uncond, float cfg)
    {
        long n = cond.Shape.ElementCount;
        float* c = (float*)cond.DataPointer;
        float* u = (float*)uncond.DataPointer;
        for (long i = 0; i < n; i++) c[i] = u[i] + cfg * (c[i] - u[i]);
    }

    /// <summary>CFG with guidance-renormalization (Lin et al. 2023, "Common Diffusion Noise Schedules…"). Folds
    /// <c>v_cfg = uncond + cfg·(cond − uncond)</c>, then rescales <c>v_cfg</c> so its mean+std match the raw
    /// <b>conditional</b> prediction (which is on-distribution), blended by <paramref name="rescale"/> in [0,1].
    /// Corrects the mean/std inflation that high CFG induces — the DC drift that turns fp8-quantized DiTs' output dark
    /// at cfg≥5. <paramref name="rescale"/>=0 is byte-identical to plain <see cref="CfgCombineInPlace"/> (so fp16
    /// models with the flag off are unchanged); ~0.7 tames the drift while preserving the guidance direction.</summary>
    public static void CfgCombineRenormInPlace(Tensor cond, Tensor uncond, float cfg, float rescale)
    {
        if (rescale <= 0f) { CfgCombineInPlace(cond, uncond, cfg); return; }
        long n = cond.Shape.ElementCount;
        float* c = (float*)cond.DataPointer;
        float* u = (float*)uncond.DataPointer;

        // Stats of the raw conditional prediction (the on-distribution target) BEFORE we overwrite cond.
        double sumC = 0; for (long i = 0; i < n; i++) sumC += c[i];
        double meanC = sumC / n;
        double varC = 0; for (long i = 0; i < n; i++) { double d = c[i] - meanC; varC += d * d; }
        double stdC = Math.Sqrt(varC / n);

        // Fold CFG in place + accumulate its stats.
        double sumCfg = 0; for (long i = 0; i < n; i++) { c[i] = u[i] + cfg * (c[i] - u[i]); sumCfg += c[i]; }
        double meanCfg = sumCfg / n;
        double varCfg = 0; for (long i = 0; i < n; i++) { double d = c[i] - meanCfg; varCfg += d * d; }
        double stdCfg = Math.Sqrt(varCfg / n);

        // Rescale v_cfg to the conditional's mean+std, then blend back by `rescale`.
        float factor = (float)(stdC / Math.Max(stdCfg, 1e-8));
        float mC = (float)meanC, mCfg = (float)meanCfg, phi = rescale;
        for (long i = 0; i < n; i++)
        {
            float matched = (c[i] - mCfg) * factor + mC;
            c[i] = phi * matched + (1f - phi) * c[i];
        }
    }

    /// <summary>Channel-last <c>[T,H,W,C]</c> → <c>[1,C,T,H,W]</c> for the VAE decode handoff.</summary>
    public static Tensor ChannelLastToBcthw(Tensor cl)
    {
        int t = (int)cl.Shape[0], h = (int)cl.Shape[1], w = (int)cl.Shape[2], c = (int)cl.Shape[3];
        Tensor outT = new Tensor(new TensorShape([1L, c, t, h, w]), DType.F32);
        float* s = (float*)cl.DataPointer;
        float* d = (float*)outT.DataPointer;
        for (int ti = 0; ti < t; ti++)
            for (int hi = 0; hi < h; hi++)
                for (int wi = 0; wi < w; wi++)
                    for (int ci = 0; ci < c; ci++)
                    {
                        long src = (((long)ti * h + hi) * w + wi) * c + ci;
                        long dst = (((long)ci * t + ti) * h + hi) * w + wi;
                        d[dst] = s[src];
                    }
        return outT;
    }

    /// <summary>Inverse of <see cref="ChannelLastToBcthw"/>: <c>[1,C,T,H,W]</c> → channel-last <c>[T,H,W,C]</c>. Used by img2img to feed the VAE-encoded source into <c>LanceLatentPatch.Patchify</c>.</summary>
    public static Tensor BcthwToChannelLast(Tensor bcthw)
    {
        int c = (int)bcthw.Shape[1], t = (int)bcthw.Shape[2], h = (int)bcthw.Shape[3], w = (int)bcthw.Shape[4];
        Tensor outT = new Tensor(new TensorShape([(long)t, h, w, c]), DType.F32);
        float* s = (float*)bcthw.DataPointer;
        float* d = (float*)outT.DataPointer;
        for (int ti = 0; ti < t; ti++)
            for (int hi = 0; hi < h; hi++)
                for (int wi = 0; wi < w; wi++)
                    for (int ci = 0; ci < c; ci++)
                    {
                        long src = (((long)ci * t + ti) * h + hi) * w + wi;
                        long dst = (((long)ti * h + hi) * w + wi) * c + ci;
                        d[dst] = s[src];
                    }
        return outT;
    }
}

/// <summary>One packed Lance generation sequence: role-partitioned token layout + M-RoPE positions + sparse attention mask. Dispose to free the position/mask tensors.</summary>
public sealed class LanceSequence : IDisposable
{
    /// <summary>Text token ids (incl. vision sentinels), aligned with <see cref="UndIdx"/>.</summary>
    public required int[] TextTokenIds { get; init; }

    /// <summary>Sequence positions of the und (text) tokens.</summary>
    public required int[] UndIdx { get; init; }

    /// <summary>Sequence positions of the gen (VAE pad) tokens.</summary>
    public required int[] GenIdx { get; init; }

    /// <summary>M-RoPE positions <c>[seq, 3]</c>.</summary>
    public required Tensor PositionIds { get; init; }

    /// <summary>Additive attention mask <c>[1,1,seq,seq]</c>.</summary>
    public required Tensor AttentionMask { get; init; }

    public void Dispose()
    {
        PositionIds.Dispose();
        AttentionMask.Dispose();
    }
}

/// <summary>Pre-tokenized ChatML scaffold around the user caption, matching upstream <c>render_qwenvl_prompt</c> for T2I/T2V: <c>&lt;|im_start|&gt;system\n{sys}&lt;|im_end|&gt;\n&lt;|im_start|&gt;user\n{caption}&lt;|im_end|&gt;\n&lt;|im_start|&gt;assistant\n&lt;vision block&gt;&lt;|im_end|&gt;</c>. Built once per tokenizer via <see cref="Create"/>; <see cref="Bare"/> is the upstream <c>text_template=false</c> layout for tests.</summary>
public sealed record LancePromptTemplate
{
    /// <summary>Tokens before the caption: <c>&lt;|im_start|&gt;system\n{sys}&lt;|im_end|&gt;\n&lt;|im_start|&gt;user\n</c>.</summary>
    public required int[] PrefixTokens { get; init; }

    /// <summary>Tokens between the caption and the vision block: <c>&lt;|im_end|&gt;\n&lt;|im_start|&gt;assistant\n</c>.</summary>
    public required int[] MidTokens { get; init; }

    /// <summary>Whether an <c>&lt;|im_end|&gt;</c> follows the vision block (true for the chat template; false for the bare layout).</summary>
    public required bool TrailingEos { get; init; }

    /// <summary>Builds the chat-templated scaffold with the upstream task system prompt. <paramref name="encode"/> must be a plain-text (no special literals) byte-level-exact BPE encoder.</summary>
    public static LancePromptTemplate Create(Func<string, int[]> encode, LanceConfig config, bool video)
    {
        string visionType = video ? "video" : "image";
        string system = video
            ? $"Describe the {visionType} by detailing the color, quantity, visible text, shape, size, texture, spatial relationships and motion/camera movements of the objects and background:"
            : $"Describe the {visionType} by detailing the color, quantity, text, shape, size, texture, spatial relationships of the objects and background:";
        int[] newline = encode("\n");
        int[] prefix = [config.ImStartTokenId, .. encode("system\n" + system), config.EosTokenId, .. newline,
            config.ImStartTokenId, .. encode("user\n")];
        int[] mid = [config.EosTokenId, .. newline, config.ImStartTokenId, .. encode("assistant\n")];
        return new LancePromptTemplate { PrefixTokens = prefix, MidTokens = mid, TrailingEos = true };
    }

    /// <summary>Upstream <c>text_template=false</c> layout: <c>&lt;bos&gt;{caption}&lt;|im_end|&gt;&lt;vision block&gt;</c> with no trailing <c>&lt;|im_end|&gt;</c>. No tokenizer needed — useful for synthetic tests.</summary>
    public static LancePromptTemplate Bare(LanceConfig config) => new()
    {
        PrefixTokens = [config.BosTokenId],
        MidTokens = [config.EosTokenId],
        TrailingEos = false,
    };
}
