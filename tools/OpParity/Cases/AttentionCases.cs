using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Tensors;

namespace HartsyInference.OpParity.Cases;

/// <summary>The legacy attention entry points as the models call them today: every parameter combination the
/// Diffusion, LLM and Audio call sites use, so a change to the attention path is checked against what runs.</summary>
public static class AttentionCases
{
    public const string Family = "attention";

    private static readonly string[] All = ["cpu", "cuda", "vulkan"];
    private static readonly string[] Gpu = ["cuda", "vulkan"];

    /// <summary>Arithmetic order differs per kernel row, so a tolerance; F32 operands, scores bounded by the seeded inputs.</summary>
    private static readonly Acceptance F32Tol = Acceptance.Within(2e-4, 1e-3, "attention kernels accumulate in different orders; F32 operands");

    public static IReadOnlyList<OpCase> Cases()
    {
        List<OpCase> cases = [];
        // ScaledDotProductAttention: head-major, equal heads. Mask present or absent; F32 and F16-allowed.
        foreach ((string name, bool mask, bool allowF16) in new[] { ("sdpa_hm_nomask", false, false), ("sdpa_hm_mask", true, false), ("sdpa_hm_mask_f16ok", true, true) })
        {
            cases.Add(new(Family, name + "_b2h4s32d64", seed => Sdpa(seed, 2, 4, 32, 48, 64, mask), (be, t) =>
                be.ScaledDotProductAttention(t.Outputs["out"], t.Inputs["q"], t.Inputs["k"], t.Inputs["v"], mask ? t.Inputs["mask"] : null, 1f / MathF.Sqrt(64), allowF16),
                _ => true, All, allowF16 ? Acceptance.Within(2e-2, 2e-2, "F16 permitted: GPU rows may run half precision") : F32Tol));
        }
        // Token-major SDPA (CUDA/Vulkan rows where they exist).
        cases.Add(new(Family, "sdpa_tm_b1h8s32d64", seed => SdpaTokenMajor(seed, 1, 8, 8, 32, 32, 64), (be, t) =>
            be.ScaledDotProductAttentionTokenMajor(t.Outputs["out"], t.Inputs["q"], t.Inputs["k"], t.Inputs["v"], null, 8, 64, 1f / 8f, false),
            be => be.SupportsTokenMajorAttention, Gpu, F32Tol,
            t =>
            {
                AttentionCall call = AttentionCall.FromTokenMajor(t.Outputs["out"], t.Inputs["q"], t.Inputs["k"], t.Inputs["v"], null, 8, 8, 64, 1f / 8f, false, 1);
                AttentionHostReference.Run(in call);
            }));
        cases.Add(new(Family, "sdpa_tm_gqa_b1h8kv2s32d64", seed => SdpaTokenMajor(seed, 1, 8, 2, 32, 32, 64), (be, t) =>
            be.ScaledDotProductAttentionTokenMajor(t.Outputs["out"], t.Inputs["q"], t.Inputs["k"], t.Inputs["v"], null, 8, 2, 64, 1f / 8f, false),
            be => be.SupportsTokenMajorGqaAttention, Gpu, F32Tol,
            t =>
            {
                AttentionCall call = AttentionCall.FromTokenMajor(t.Outputs["out"], t.Inputs["q"], t.Inputs["k"], t.Inputs["v"], null, 8, 2, 64, 1f / 8f, false, 1);
                AttentionHostReference.Run(in call);
            }));
        // FlashAttention: the LLM / vision-encoder entry. Causal and bidirectional, GQA, qOffset, softcap, sink, window, ALiBi.
        foreach ((string name, bool causal, int hq, int hkv, int sq, int skv, int qOff, float softcap, bool sink, int window, bool alibi) in new[]
        {
            ("flash_bidir_mha", false, 8, 8, 32, 32, 0, 0f, false, 0, false),            // Siglip, Mllama, Bert
            ("flash_causal_gqa", true, 8, 2, 32, 32, 0, 0f, false, 0, false),             // MiniMax H3 text encoder, decoder prefill
            ("flash_causal_gqa_qoffset", true, 8, 2, 8, 40, 32, 0f, false, 0, false),     // decode-step continuation
            ("flash_causal_softcap", true, 4, 4, 32, 32, 0, 50f, false, 0, false),        // Gemma-style logit softcap
            ("flash_causal_sink", true, 4, 4, 32, 32, 0, 0f, true, 0, false),             // GPT-OSS sinks
            ("flash_causal_window", true, 4, 4, 32, 32, 0, 0f, false, 16, false),         // sliding window locals
            ("flash_bidir_alibi", false, 4, 4, 32, 32, 0, 0f, false, 0, true),            // jina-bert ALiBi
            ("flash_causal_softcap_alibi", true, 4, 4, 32, 32, 0, 30f, false, 0, true),   // the ALiBi/softcap order case
        })
        {
            // The legacy CPU reference applied ALiBi before the softcap; the contract (§3) caps first. That case is judged
            // against the contract reference, with the old output documented as wrong by design.
            Acceptance rule = softcap > 0f && alibi
                ? Acceptance.AgainstHost(2e-4, 1e-3, "the legacy reference applied ALiBi before the softcap; the contract caps first, so the contract reference is the oracle")
                : F32Tol;
            cases.Add(new(Family, name, seed => Flash(seed, 1, hq, hkv, sq, skv, 64, sink, alibi), (be, t) =>
                be.FlashAttention(t.Outputs["out"], t.Inputs["q"], t.Inputs["k"], t.Inputs["v"], kvLen: skv, kvGroup: hq / hkv, causal: causal, qOffset: qOff,
                    scale: 1f / 8f, softcap: softcap, sink: sink ? t.Inputs["sink"] : null, slidingWindow: window, alibiSlopes: alibi ? t.Inputs["alibi"] : null),
                _ => true, All, rule,
                t => AttentionHostReference.Run(AttentionCall.FromFlashAttention(t.Outputs["out"], t.Inputs["q"], t.Inputs["k"], t.Inputs["v"], skv, causal, qOff,
                    1f / 8f, softcap, sink ? t.Inputs["sink"] : null, window, alibi ? t.Inputs["alibi"] : null))));
        }
        // FlashAttentionDev: positions read from a device buffer. Only backends that allocate one serve it.
        cases.Add(new(Family, "flashdev_causal_gqa", seed => Flash(seed, 1, 8, 2, 8, 40, 64, false, false), (be, t) =>
            {
                ulong pos = be.AllocDevicePos();
                if (pos == 0) throw new NotSupportedException("no device positions");
                try
                {
                    be.WriteDevicePos(pos, 40, 32);
                    be.FlashAttentionDev(t.Outputs["out"], t.Inputs["q"], t.Inputs["k"], t.Inputs["v"], kvLen: 40, kvGroup: 4, causal: true, qOffset: 32, scale: 1f / 8f, pos);
                }
                finally { be.FreeDevicePos(pos); }
            },
            HasDevicePositions, Gpu, F32Tol,
            // The host reference takes host positions: the device buffer holds the same kvLen and qOffset.
            t =>
            {
                AttentionCall call = AttentionCall.FromFlashAttention(t.Outputs["out"], t.Inputs["q"], t.Inputs["k"], t.Inputs["v"],
                    kvLen: 40, causal: true, qOffset: 32, scale: 1f / 8f, softcap: 0f, sink: null, slidingWindow: 0, alibiSlopes: null);
                AttentionHostReference.Run(in call);
            }));
        return cases;
    }

    /// <summary>A backend serves device positions when it hands out a buffer for them; 0 means it does not.</summary>
    private static bool HasDevicePositions(IBackend be)
    {
        ulong handle = be.AllocDevicePos();
        if (handle == 0) return false;
        be.FreeDevicePos(handle);
        return true;
    }

    private static OpTensors Sdpa(int seed, int b, int h, int sq, int skv, int d, bool mask)
    {
        OpTensors t = new();
        t.Inputs["q"] = Seeded.F32(new TensorShape(b, h, sq, d), seed, 1);
        t.Inputs["k"] = Seeded.F32(new TensorShape(b, h, skv, d), seed, 2);
        t.Inputs["v"] = Seeded.F32(new TensorShape(b, h, skv, d), seed, 3);
        if (mask) t.Inputs["mask"] = CausalBias(b, h, sq, skv);
        t.Outputs["out"] = Seeded.Zeros(new TensorShape(b, h, sq, d));
        return t;
    }

    private static OpTensors SdpaTokenMajor(int seed, int b, int hq, int hkv, int sq, int skv, int d)
    {
        OpTensors t = new();
        t.Inputs["q"] = Seeded.F32(new TensorShape(b, sq, hq, d), seed, 1);
        t.Inputs["k"] = Seeded.F32(new TensorShape(b, skv, hkv, d), seed, 2);
        t.Inputs["v"] = Seeded.F32(new TensorShape(b, skv, hkv, d), seed, 3);
        t.Outputs["out"] = Seeded.Zeros(new TensorShape(b, sq, hq, d));
        return t;
    }

    private static OpTensors Flash(int seed, int b, int hq, int hkv, int sq, int skv, int d, bool sink, bool alibi)
    {
        OpTensors t = new();
        t.Inputs["q"] = Seeded.F32(new TensorShape(b, hq, sq, d), seed, 1);
        t.Inputs["k"] = Seeded.F32(new TensorShape(b, hkv, skv, d), seed, 2);
        t.Inputs["v"] = Seeded.F32(new TensorShape(b, hkv, skv, d), seed, 3);
        if (sink) t.Inputs["sink"] = Seeded.F32(new TensorShape(hq), seed, 4, 2f);
        if (alibi)
        {
            Tensor slopes = Seeded.Zeros(new TensorShape(hq));
            Span<float> s = slopes.AsSpan<float>();
            for (int i = 0; i < hq; i++) s[i] = MathF.Pow(2f, -8f * (i + 1) / hq);
            t.Inputs["alibi"] = slopes;
        }
        t.Outputs["out"] = Seeded.Zeros(new TensorShape(b, hq, sq, d));
        return t;
    }

    /// <summary>An additive causal bias: 0 where key ≤ query, -1e9 above the diagonal (finite, so the output stays finite).</summary>
    private static Tensor CausalBias(int b, int h, int sq, int skv)
    {
        Tensor mask = Seeded.Zeros(new TensorShape(b, h, sq, skv));
        Span<float> m = mask.AsSpan<float>();
        int offset = skv - sq;
        for (int bh = 0; bh < b * h; bh++)
            for (int i = 0; i < sq; i++)
                for (int j = 0; j < skv; j++)
                    m[(bh * sq + i) * skv + j] = j > i + offset ? -1e9f : 0f;
        return mask;
    }
}
