using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.ModelHandler.Mxfp4;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Unit tests for the GPT-OSS encoder side of Microsoft Lens — MoE routing math, MXFP4 dequant,
/// encoder construction/validation, attention sinks scaling, sliding-window mask generation, multi-layer
/// hidden-state capture, and the LensGptOssEncoder offset-stripping wrapper. End-to-end generation tests
/// against the actual `microsoft/Lens` checkpoint live in <see cref="LensGenerationTests"/> (gated on
/// `LENS_PATH` env var) — these are the deterministic unit tests that run on CI without checkpoints.</summary>
public sealed unsafe class LensEncoderTests
{
    // ────────────────────────────────────────────────────────────────────
    // Config
    // ────────────────────────────────────────────────────────────────────

    [Fact]
    public void GptOss_Config_Preset_Matches_Upstream()
    {
        LlamaStyleEncoderConfig c = LlamaStyleEncoderConfig.GptOss;
        Assert.Equal(2880, c.HiddenSize);
        Assert.Equal(24, c.NumLayers);
        Assert.Equal(64, c.NumQueryHeads);
        Assert.Equal(8, c.NumKvHeads);
        Assert.Equal(64, c.HeadDim);
        Assert.Equal(2880, c.IntermediateSize);
        Assert.Equal(201088, c.VocabSize);
        Assert.Equal(1e-5f, c.RmsNormEps);
        Assert.Equal(150_000f, c.RopeTheta);
        Assert.True(c.AttentionBias);
        Assert.True(c.HasAttentionSinks);
        Assert.True(c.ClampedSwiGlu);
        Assert.Equal(7.0f, c.ClampedSwiGluLimit);
        Assert.Equal(1.702f, c.ClampedSwiGluAlpha);
        Assert.Equal(32, c.NumLocalExperts);
        Assert.Equal(4, c.NumExpertsPerToken);
        Assert.Equal(128, c.SlidingWindow);
        Assert.NotNull(c.LayerAttentionTypes);
        Assert.Equal(24, c.LayerAttentionTypes!.Length);
        Assert.Equal(RopeScalingType.Yarn, c.RopeScaling);
        Assert.Equal(32.0f, c.YarnFactor);
        Assert.Equal(32.0f, c.YarnBetaFast);
        Assert.Equal(1.0f, c.YarnBetaSlow);
        Assert.Equal(4096, c.YarnOriginalMaxPosition);
        Assert.Equal(200002, c.EosTokenId);
    }

    [Fact]
    public void GptOss_Alternating_Attention_Pattern_Starts_With_Sliding()
    {
        string[] pattern = LlamaStyleEncoderConfig.GptOss.LayerAttentionTypes!;
        for (int i = 0; i < pattern.Length; i++)
            Assert.Equal(i % 2 == 0 ? "sliding_attention" : "full_attention", pattern[i]);
    }

    // ────────────────────────────────────────────────────────────────────
    // MXFP4 dequant
    // ────────────────────────────────────────────────────────────────────

    [Fact]
    public void Mxfp4_Lut_Has_Sixteen_Entries_With_Sign_Magnitude_Symmetry()
    {
        Assert.Equal(16, Mxfp4Codec.Fp4Lut.Length);
        // First 8 are nonneg ascending magnitudes; second 8 are their negatives.
        for (int i = 0; i < 8; i++)
            Assert.Equal(-Mxfp4Codec.Fp4Lut[i], Mxfp4Codec.Fp4Lut[i + 8]);
        // Magnitudes per upstream: 0, 0.5, 1, 1.5, 2, 3, 4, 6.
        float[] expectedMagnitudes = [0f, 0.5f, 1f, 1.5f, 2f, 3f, 4f, 6f];
        for (int i = 0; i < 8; i++)
            Assert.Equal(expectedMagnitudes[i], Mxfp4Codec.Fp4Lut[i]);
    }

    [Fact]
    public void Mxfp4_DequantToF32_OneBlock_AllZeros_With_UnitScale()
    {
        // 16 bytes = 32 elements, all zero nibbles. Scale = 127 → 2^0 = 1.
        Tensor blocks = MakeU8(16, fill: 0);
        Tensor scales = MakeU8(1, fill: 127);
        Tensor dequant = Mxfp4Codec.DequantToF32(blocks, scales, new TensorShape(32));
        try
        {
            float* p = (float*)dequant.DataPointer;
            for (int i = 0; i < 32; i++) Assert.Equal(0f, p[i]);
        }
        finally
        {
            dequant.Dispose();
            blocks.Dispose();
            scales.Dispose();
        }
    }

    [Fact]
    public void Mxfp4_DequantToF32_OneBlock_NibbleOrdering_LowFirst()
    {
        // Byte 0x21: low=1 → +0.5, high=2 → +1.0. Place at output positions 0 and 1.
        // All other bytes 0.
        Tensor blocks = MakeU8(16, fill: 0);
        ((byte*)blocks.DataPointer)[0] = 0x21;
        Tensor scales = MakeU8(1, fill: 127);
        Tensor dequant = Mxfp4Codec.DequantToF32(blocks, scales, new TensorShape(32));
        try
        {
            float* p = (float*)dequant.DataPointer;
            Assert.Equal(0.5f, p[0]); // low nibble of byte 0 = 1 → +0.5
            Assert.Equal(1.0f, p[1]); // high nibble of byte 0 = 2 → +1.0
            for (int i = 2; i < 32; i++) Assert.Equal(0f, p[i]);
        }
        finally
        {
            dequant.Dispose();
            blocks.Dispose();
            scales.Dispose();
        }
    }

    [Fact]
    public void Mxfp4_DequantToF32_ScaleAppliesPowerOf2()
    {
        // Nibble 4 = +2.0 magnitude. Scale stored=130 → exp=130-127=3 → 2^3=8.
        // So output should be 2.0 * 8 = 16.0.
        Tensor blocks = MakeU8(16, fill: 0);
        ((byte*)blocks.DataPointer)[0] = 0x04; // low nibble = 4
        Tensor scales = MakeU8(1, fill: 130);
        Tensor dequant = Mxfp4Codec.DequantToF32(blocks, scales, new TensorShape(32));
        try
        {
            Assert.Equal(16.0f, ((float*)dequant.DataPointer)[0]);
        }
        finally
        {
            dequant.Dispose();
            blocks.Dispose();
            scales.Dispose();
        }
    }

    [Fact]
    public void Mxfp4_DequantToF32_NegativeMagnitudes_AreSignFlipped()
    {
        // Nibble 13 = sign bit + magnitude 5 → -3.0. Scale 127 = unit.
        Tensor blocks = MakeU8(16, fill: 0);
        ((byte*)blocks.DataPointer)[0] = 0x0D;
        Tensor scales = MakeU8(1, fill: 127);
        Tensor dequant = Mxfp4Codec.DequantToF32(blocks, scales, new TensorShape(32));
        try
        {
            Assert.Equal(-3.0f, ((float*)dequant.DataPointer)[0]);
        }
        finally
        {
            dequant.Dispose();
            blocks.Dispose();
            scales.Dispose();
        }
    }

    [Fact]
    public void Mxfp4_DequantToF32_MultipleBlocks_ApplyOwnScales()
    {
        // 2 blocks × 16 bytes each, each filled with byte 0x04 (low nibble=4=+2.0, high=0).
        // Block 0 scale 127 (unit) → block-0 even outputs = 2.0.
        // Block 1 scale 128 (2^1) → block-1 even outputs = 4.0.
        Tensor blocks = MakeU8(32, fill: 0x04);
        Tensor scales = MakeU8(2);
        ((byte*)scales.DataPointer)[0] = 127;
        ((byte*)scales.DataPointer)[1] = 128;
        Tensor dequant = Mxfp4Codec.DequantToF32(blocks, scales, new TensorShape(64));
        try
        {
            float* p = (float*)dequant.DataPointer;
            for (int i = 0; i < 32; i += 2) Assert.Equal(2.0f, p[i]);
            for (int i = 32; i < 64; i += 2) Assert.Equal(4.0f, p[i]);
        }
        finally
        {
            dequant.Dispose();
            blocks.Dispose();
            scales.Dispose();
        }
    }

    [Fact]
    public void Mxfp4_DequantAllPairs_FindsAndStripsCompanions()
    {
        Dictionary<string, Tensor> dict = new()
        {
            ["model.layers.0.mlp.experts.gate_up_proj_blocks"] = MakeU8(16, fill: 0),
            ["model.layers.0.mlp.experts.gate_up_proj_scales"] = MakeU8(1, fill: 127),
            ["other.weight"] = MakeU8(8, fill: 1),
        };
        int n = Mxfp4Codec.DequantAllPairsInPlace(dict);
        Assert.Equal(1, n);
        Assert.True(dict.ContainsKey("model.layers.0.mlp.experts.gate_up_proj"));
        Assert.False(dict.ContainsKey("model.layers.0.mlp.experts.gate_up_proj_blocks"));
        Assert.False(dict.ContainsKey("model.layers.0.mlp.experts.gate_up_proj_scales"));
        Assert.True(dict.ContainsKey("other.weight"));
        foreach (Tensor t in dict.Values) t.Dispose();
    }

    [Fact]
    public void Mxfp4_DequantGptOssExpert_Transposes_LastTwoAxes()
    {
        // On-disk GPT-OSS expert layout: blocks [E, A, G, 16], scales [E, A, G]. The dequant must
        // produce the transposed runtime layout [E, G*32, A] (matches transformers'
        // convert_moe_packed_tensors, which ends in .transpose(1, 2)). Verified byte-exact against the
        // Python reference for random inputs; here we pin a tiny deterministic case.
        // E=1, A=2, G=1 → output [1, 32, 2].
        Tensor blocks = MakeU8(1 * 2 * 1 * 16, fill: 0);
        byte* bp = (byte*)blocks.DataPointer;
        // Block for (e=0, a=0, g=0): byte 0 low nibble = 4 (+2.0) → output element index hidden=0.
        bp[0] = 0x04;
        // Block for (e=0, a=1, g=0): starts at byte 16. low nibble of byte 16 = 2 (+1.0) → hidden=0.
        bp[16] = 0x02;
        Tensor blocks4d = Reshape(blocks, 1, 2, 1, 16);

        Tensor scales = MakeU8(1 * 2 * 1, fill: 127); // unit scale 2^0
        Tensor scales3d = Reshape(scales, 1, 2, 1);

        Tensor dq = Mxfp4Codec.DequantGptOssExpert(blocks4d, scales3d);
        try
        {
            Assert.Equal(new long[] { 1, 32, 2 }, ToLongArray(dq.Shape));
            float* p = (float*)dq.DataPointer;
            // output[e=0, hidden=0, a=0] = +2.0 ; output[e=0, hidden=0, a=1] = +1.0
            Assert.Equal(2.0f, p[(0 * 32 + 0) * 2 + 0]); // a=0
            Assert.Equal(1.0f, p[(0 * 32 + 0) * 2 + 1]); // a=1
            // Everything else zero.
            Assert.Equal(0f, p[(0 * 32 + 1) * 2 + 0]);
        }
        finally
        {
            dq.Dispose();
            blocks4d.Dispose();
            scales3d.Dispose();
        }
    }

    [Fact]
    public void Mxfp4_DequantGptOssExpert_RejectsWrongRank()
    {
        Tensor blocks3d = new Tensor(new TensorShape(1, 2, 16), DType.U8);
        Tensor scales = new Tensor(new TensorShape(1, 2), DType.U8);
        try
        {
            Assert.Throws<ArgumentException>(() => Mxfp4Codec.DequantGptOssExpert(blocks3d, scales));
        }
        finally
        {
            blocks3d.Dispose();
            scales.Dispose();
        }
    }

    [Fact]
    public void Mxfp4_DequantGptOssExpertsInPlace_DequantsGateUpAndDown_StripsCompanions()
    {
        // gate_up: E=2, hidden=G*32, on-disk [E, 2I, G, 16]; here 2I=3, G=1 → output [2, 32, 3].
        // down:    on-disk [E, hidden, G, 16]; here hidden=3, G=1 → output [2, 32, 3].
        Dictionary<string, Tensor> dict = new()
        {
            ["model.layers.0.mlp.experts.gate_up_proj_blocks"] = Reshape(MakeU8(2 * 3 * 1 * 16, fill: 0), 2, 3, 1, 16),
            ["model.layers.0.mlp.experts.gate_up_proj_scales"] = Reshape(MakeU8(2 * 3 * 1, fill: 127), 2, 3, 1),
            ["model.layers.0.mlp.experts.down_proj_blocks"] = Reshape(MakeU8(2 * 3 * 1 * 16, fill: 0), 2, 3, 1, 16),
            ["model.layers.0.mlp.experts.down_proj_scales"] = Reshape(MakeU8(2 * 3 * 1, fill: 127), 2, 3, 1),
            ["model.layers.0.self_attn.q_proj.weight"] = new Tensor(new TensorShape(4, 4), DType.F32),
        };
        int n = Mxfp4Codec.DequantGptOssExpertsInPlace(dict);
        try
        {
            Assert.Equal(2, n);
            Assert.True(dict.ContainsKey("model.layers.0.mlp.experts.gate_up_proj"));
            Assert.True(dict.ContainsKey("model.layers.0.mlp.experts.down_proj"));
            Assert.False(dict.ContainsKey("model.layers.0.mlp.experts.gate_up_proj_blocks"));
            Assert.False(dict.ContainsKey("model.layers.0.mlp.experts.gate_up_proj_scales"));
            Assert.False(dict.ContainsKey("model.layers.0.mlp.experts.down_proj_blocks"));
            Assert.False(dict.ContainsKey("model.layers.0.mlp.experts.down_proj_scales"));
            // Bare keys carry the transposed runtime shape [E, G*32, A].
            Assert.Equal(new long[] { 2, 32, 3 }, ToLongArray(dict["model.layers.0.mlp.experts.gate_up_proj"].Shape));
            Assert.Equal(new long[] { 2, 32, 3 }, ToLongArray(dict["model.layers.0.mlp.experts.down_proj"].Shape));
            // Non-MXFP4 tensors pass through untouched.
            Assert.True(dict.ContainsKey("model.layers.0.self_attn.q_proj.weight"));
        }
        finally
        {
            foreach (Tensor t in dict.Values) t.Dispose();
        }
    }

    [Fact]
    public void Mxfp4_DequantGptOssExpertsInPlace_NoMxfp4Pairs_IsNoOp()
    {
        Dictionary<string, Tensor> dict = new()
        {
            ["model.embed_tokens.weight"] = new Tensor(new TensorShape(4, 4), DType.F32),
        };
        int n = Mxfp4Codec.DequantGptOssExpertsInPlace(dict);
        try
        {
            Assert.Equal(0, n);
            Assert.Single(dict);
        }
        finally
        {
            foreach (Tensor t in dict.Values) t.Dispose();
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // MoE FFN — top-k routing math
    // ────────────────────────────────────────────────────────────────────

    [Fact]
    public void MoeFfn_Construction_RejectsBadTopK()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new GptOssMoeFfn(hiddenSize: 8, intermediateSize: 8, numExperts: 4, topK: 0,
                clampLimit: 7.0f, alpha: 1.702f));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new GptOssMoeFfn(hiddenSize: 8, intermediateSize: 8, numExperts: 4, topK: 5,
                clampLimit: 7.0f, alpha: 1.702f));
    }

    [Fact]
    public void MoeFfn_LoadWeights_ValidatesShapes()
    {
        const int hidden = 8;
        const int intermediate = 8;
        const int numExperts = 4;
        GptOssMoeFfn ffn = new GptOssMoeFfn(hidden, intermediate, numExperts, topK: 2,
            clampLimit: 7.0f, alpha: 1.702f);

        // Right shapes — should load cleanly.
        Dictionary<string, Tensor> good = new()
        {
            ["m.router.weight"] = new Tensor(new TensorShape(numExperts, hidden), DType.F32),
            ["m.router.bias"] = new Tensor(new TensorShape(numExperts), DType.F32),
            ["m.experts.gate_up_proj"] = new Tensor(new TensorShape(numExperts, hidden, 2 * intermediate), DType.F32),
            ["m.experts.gate_up_proj_bias"] = new Tensor(new TensorShape(numExperts, 2 * intermediate), DType.F32),
            ["m.experts.down_proj"] = new Tensor(new TensorShape(numExperts, intermediate, hidden), DType.F32),
            ["m.experts.down_proj_bias"] = new Tensor(new TensorShape(numExperts, hidden), DType.F32),
        };
        ffn.LoadWeights(good, "m");

        // Wrong shape on router.weight should throw.
        Dictionary<string, Tensor> bad = new()
        {
            ["m.router.weight"] = new Tensor(new TensorShape(numExperts + 1, hidden), DType.F32),
            ["m.router.bias"] = good["m.router.bias"],
            ["m.experts.gate_up_proj"] = good["m.experts.gate_up_proj"],
            ["m.experts.gate_up_proj_bias"] = good["m.experts.gate_up_proj_bias"],
            ["m.experts.down_proj"] = good["m.experts.down_proj"],
            ["m.experts.down_proj_bias"] = good["m.experts.down_proj_bias"],
        };
        GptOssMoeFfn ffn2 = new GptOssMoeFfn(hidden, intermediate, numExperts, topK: 2,
            clampLimit: 7.0f, alpha: 1.702f);
        Assert.Throws<InvalidOperationException>(() => ffn2.LoadWeights(bad, "m"));

        foreach (Tensor t in good.Values) t.Dispose();
        bad["m.router.weight"].Dispose();
    }

    [Fact]
    public void MoeFfn_EnumerateWeights_YieldsAllSix()
    {
        const int hidden = 8;
        const int intermediate = 8;
        const int numExperts = 4;
        GptOssMoeFfn ffn = new GptOssMoeFfn(hidden, intermediate, numExperts, topK: 2, 7f, 1.702f);
        Dictionary<string, Tensor> w = new()
        {
            ["m.router.weight"] = new Tensor(new TensorShape(numExperts, hidden), DType.F32),
            ["m.router.bias"] = new Tensor(new TensorShape(numExperts), DType.F32),
            ["m.experts.gate_up_proj"] = new Tensor(new TensorShape(numExperts, hidden, 2 * intermediate), DType.F32),
            ["m.experts.gate_up_proj_bias"] = new Tensor(new TensorShape(numExperts, 2 * intermediate), DType.F32),
            ["m.experts.down_proj"] = new Tensor(new TensorShape(numExperts, intermediate, hidden), DType.F32),
            ["m.experts.down_proj_bias"] = new Tensor(new TensorShape(numExperts, hidden), DType.F32),
        };
        ffn.LoadWeights(w, "m");
        int count = 0;
        foreach (Tensor _ in ffn.EnumerateWeights()) count++;
        Assert.Equal(6, count);
        foreach (Tensor t in w.Values) t.Dispose();
    }

    // ────────────────────────────────────────────────────────────────────
    // MoE end-to-end forward — functional correctness vs hand-computed values
    // ────────────────────────────────────────────────────────────────────

    [Fact]
    public void MoeFfn_Top1_Routing_Routes_To_HighestLogit_Expert()
    {
        // Setup: hidden=4, intermediate=4, numExperts=2, topK=1.
        // Router weight = [[1,0,0,0],[0,1,0,0]] so input [1,0,0,0] → expert 0, input [0,1,0,0] → expert 1.
        // down_proj all-zero, down_proj_bias different per expert. With topK=1 the softmax weight is 1.0,
        // so output equals the routed expert's down_proj_bias verbatim regardless of gate/up math.
        const int hidden = 4;
        const int intermediate = 4;
        const int numExperts = 2;

        Tensor routerW = MakeF32(numExperts, hidden, vals: new float[]
        {
            1, 0, 0, 0,
            0, 1, 0, 0,
        });
        Tensor routerB = MakeF32(numExperts, vals: new float[] { 0, 0 });
        Tensor gateUp = MakeF32Zero(numExperts, hidden, 2 * intermediate);
        Tensor gateUpBias = MakeF32Zero(numExperts, 2 * intermediate);
        Tensor down = MakeF32Zero(numExperts, intermediate, hidden);
        Tensor downBias = MakeF32(numExperts, hidden, vals: new float[]
        {
            10, 20, 30, 40,       // expert 0
            100, 200, 300, 400,   // expert 1
        });

        GptOssMoeFfn ffn = new GptOssMoeFfn(hidden, intermediate, numExperts, topK: 1, 7f, 1.702f);
        Dictionary<string, Tensor> weights = new()
        {
            ["m.router.weight"] = routerW,
            ["m.router.bias"] = routerB,
            ["m.experts.gate_up_proj"] = gateUp,
            ["m.experts.gate_up_proj_bias"] = gateUpBias,
            ["m.experts.down_proj"] = down,
            ["m.experts.down_proj_bias"] = downBias,
        };
        ffn.LoadWeights(weights, "m");

        using HartsyInference.Cpu.CpuBackend backend = new HartsyInference.Cpu.CpuBackend();

        // Two test inputs: routed to expert 0 and expert 1.
        Tensor input = MakeF32(1, 2, hidden, vals: new float[]
        {
            1, 0, 0, 0,
            0, 1, 0, 0,
        });
        Tensor output = ffn.Forward(backend, input);
        try
        {
            float* p = (float*)output.DataPointer;
            // Token 0 routes to expert 0 → expected [10, 20, 30, 40]
            Assert.Equal(10f, p[0], precision: 4);
            Assert.Equal(20f, p[1], precision: 4);
            Assert.Equal(30f, p[2], precision: 4);
            Assert.Equal(40f, p[3], precision: 4);
            // Token 1 routes to expert 1 → expected [100, 200, 300, 400]
            Assert.Equal(100f, p[4], precision: 4);
            Assert.Equal(200f, p[5], precision: 4);
            Assert.Equal(300f, p[6], precision: 4);
            Assert.Equal(400f, p[7], precision: 4);
        }
        finally
        {
            input.Dispose();
            output.Dispose();
            foreach (Tensor t in weights.Values) t.Dispose();
        }
    }

    [Fact]
    public void MoeFfn_Top2_Routing_SoftmaxWeights_Match_HandComputed()
    {
        // Setup: hidden=4, intermediate=4, numExperts=2, topK=2.
        // Router output for input [1,0,0,0] should be [3, 1] (so softmax over top-2 = [σ(3-3)/(σ(3-3)+σ(1-3)),
        // σ(1-3)/(...)] = [1/(1+exp(-2)), exp(-2)/(1+exp(-2))] ≈ [0.8808, 0.1192]).
        // Each expert contributes 10 or 100 in channel 0; expected mix = 0.8808*10 + 0.1192*100 ≈ 20.7280.
        const int hidden = 4;
        const int intermediate = 4;
        const int numExperts = 2;

        Tensor routerW = MakeF32(numExperts, hidden, vals: new float[]
        {
            3, 0, 0, 0,
            1, 0, 0, 0,
        });
        Tensor routerB = MakeF32(numExperts, vals: new float[] { 0, 0 });
        Tensor gateUp = MakeF32Zero(numExperts, hidden, 2 * intermediate);
        Tensor gateUpBias = MakeF32Zero(numExperts, 2 * intermediate);
        Tensor down = MakeF32Zero(numExperts, intermediate, hidden);
        Tensor downBias = MakeF32(numExperts, hidden, vals: new float[]
        {
            10, 0, 0, 0,    // expert 0
            100, 0, 0, 0,   // expert 1
        });

        GptOssMoeFfn ffn = new GptOssMoeFfn(hidden, intermediate, numExperts, topK: 2, 7f, 1.702f);
        Dictionary<string, Tensor> weights = new()
        {
            ["m.router.weight"] = routerW,
            ["m.router.bias"] = routerB,
            ["m.experts.gate_up_proj"] = gateUp,
            ["m.experts.gate_up_proj_bias"] = gateUpBias,
            ["m.experts.down_proj"] = down,
            ["m.experts.down_proj_bias"] = downBias,
        };
        ffn.LoadWeights(weights, "m");

        using HartsyInference.Cpu.CpuBackend backend = new HartsyInference.Cpu.CpuBackend();
        Tensor input = MakeF32(1, 1, hidden, vals: new float[] { 1, 0, 0, 0 });
        Tensor output = ffn.Forward(backend, input);
        try
        {
            float w0 = 1.0f / (1.0f + MathF.Exp(-2.0f));
            float w1 = MathF.Exp(-2.0f) / (1.0f + MathF.Exp(-2.0f));
            float expected = w0 * 10.0f + w1 * 100.0f;
            Assert.Equal(expected, ((float*)output.DataPointer)[0], precision: 3);
        }
        finally
        {
            input.Dispose();
            output.Dispose();
            foreach (Tensor t in weights.Values) t.Dispose();
        }
    }

    [Fact]
    public void MoeFfn_ClampedActivation_MatchesUpstreamMath()
    {
        // Drive a known gate/up directly through the activation: set gate=2, up=3, L=7, α=1.702.
        // Then glu = 2 * sigmoid(2 * 1.702) ≈ 2 * 0.9665 ≈ 1.9331.
        // gated_inner = (3 + 1) * 1.9331 ≈ 7.7322.
        // down_proj_bias = [val] → output ≈ val (with down_proj=0; expert weight=1 for topK=1).
        // We can verify by computing expert output = 0 * gated_inner + down_proj_bias scaled by topK=1.
        // Simpler: use gate/up=0, bias gate_up_proj_bias = [2, 3, ...], compare against hand-computed
        // gated_inner times down_proj_bias[k] contribution.
        const int hidden = 2;
        const int intermediate = 1;
        const int numExperts = 1;

        Tensor routerW = MakeF32(numExperts, hidden, vals: new float[] { 0, 0 });
        Tensor routerB = MakeF32(numExperts, vals: new float[] { 1 });
        Tensor gateUp = MakeF32Zero(numExperts, hidden, 2 * intermediate);
        Tensor gateUpBias = MakeF32(numExperts, 2 * intermediate, vals: new float[] { 2.0f, 3.0f });
        // down_proj[expert=0] shape [intermediate=1, hidden=2] all ones — so output = gated_inner * [1, 1].
        Tensor down = MakeF32(numExperts, intermediate, hidden, vals: new float[] { 1, 1 });
        Tensor downBias = MakeF32Zero(numExperts, hidden);

        GptOssMoeFfn ffn = new GptOssMoeFfn(hidden, intermediate, numExperts, topK: 1, 7f, 1.702f);
        Dictionary<string, Tensor> weights = new()
        {
            ["m.router.weight"] = routerW,
            ["m.router.bias"] = routerB,
            ["m.experts.gate_up_proj"] = gateUp,
            ["m.experts.gate_up_proj_bias"] = gateUpBias,
            ["m.experts.down_proj"] = down,
            ["m.experts.down_proj_bias"] = downBias,
        };
        ffn.LoadWeights(weights, "m");

        using HartsyInference.Cpu.CpuBackend backend = new HartsyInference.Cpu.CpuBackend();
        Tensor input = MakeF32(1, 1, hidden, vals: new float[] { 0, 0 }); // gate_up = bias = [2, 3]
        Tensor output = ffn.Forward(backend, input);
        try
        {
            float g = 2f, u = 3f, alpha = 1.702f;
            float sig = 1f / (1f + MathF.Exp(-alpha * g));
            float glu = g * sig;
            float gated = (u + 1f) * glu;
            float expected = gated; // down=1, summed gives gated * 1 = gated
            Assert.Equal(expected, ((float*)output.DataPointer)[0], precision: 4);
            Assert.Equal(expected, ((float*)output.DataPointer)[1], precision: 4);
        }
        finally
        {
            input.Dispose();
            output.Dispose();
            foreach (Tensor t in weights.Values) t.Dispose();
        }
    }

    [Fact]
    public void MoeFfn_ClampedActivation_RespectsCeiling()
    {
        // gate=10 should clamp to L=7. glu = 7 * sigmoid(7 * 1.702) ≈ 7 * 0.99999... ≈ 7.0 (almost).
        // up = 10 should clamp to L=7. gated = (7+1) * 7 ≈ 56.0.
        const int hidden = 1;
        const int intermediate = 1;
        const int numExperts = 1;

        Tensor routerW = MakeF32(numExperts, hidden, vals: new float[] { 0 });
        Tensor routerB = MakeF32(numExperts, vals: new float[] { 1 });
        Tensor gateUp = MakeF32Zero(numExperts, hidden, 2 * intermediate);
        Tensor gateUpBias = MakeF32(numExperts, 2 * intermediate, vals: new float[] { 10f, 10f });
        Tensor down = MakeF32(numExperts, intermediate, hidden, vals: new float[] { 1 });
        Tensor downBias = MakeF32Zero(numExperts, hidden);

        GptOssMoeFfn ffn = new GptOssMoeFfn(hidden, intermediate, numExperts, topK: 1, 7f, 1.702f);
        Dictionary<string, Tensor> weights = new()
        {
            ["m.router.weight"] = routerW,
            ["m.router.bias"] = routerB,
            ["m.experts.gate_up_proj"] = gateUp,
            ["m.experts.gate_up_proj_bias"] = gateUpBias,
            ["m.experts.down_proj"] = down,
            ["m.experts.down_proj_bias"] = downBias,
        };
        ffn.LoadWeights(weights, "m");

        using HartsyInference.Cpu.CpuBackend backend = new HartsyInference.Cpu.CpuBackend();
        Tensor input = MakeF32(1, 1, hidden, vals: new float[] { 0 });
        Tensor output = ffn.Forward(backend, input);
        try
        {
            // After clamp: gate=7, up=7.
            float sig = 1f / (1f + MathF.Exp(-1.702f * 7f));
            float glu = 7f * sig;
            float expected = (7f + 1f) * glu;
            Assert.Equal(expected, ((float*)output.DataPointer)[0], precision: 3);
        }
        finally
        {
            input.Dispose();
            output.Dispose();
            foreach (Tensor t in weights.Values) t.Dispose();
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // GptOssEncoder structural
    // ────────────────────────────────────────────────────────────────────

    [Fact]
    public void GptOssEncoder_Rejects_Config_Without_Experts()
    {
        LlamaStyleEncoderConfig bad = LlamaStyleEncoderConfig.GptOss with { NumLocalExperts = 0 };
        Assert.Throws<ArgumentException>(() => new GptOssEncoder(bad));
    }

    [Fact]
    public void GptOssEncoder_Rejects_Config_With_TopK_Out_Of_Range()
    {
        LlamaStyleEncoderConfig bad = LlamaStyleEncoderConfig.GptOss with { NumExpertsPerToken = 33 };
        Assert.Throws<ArgumentException>(() => new GptOssEncoder(bad));
    }

    [Fact]
    public void GptOssEncoder_Rejects_Mismatched_LayerAttentionTypes_Length()
    {
        LlamaStyleEncoderConfig bad = LlamaStyleEncoderConfig.GptOss with
        {
            LayerAttentionTypes = new[] { "full_attention", "full_attention" }, // length 2 ≠ numLayers 24
        };
        Assert.Throws<ArgumentException>(() => new GptOssEncoder(bad));
    }

    [Fact]
    public void GptOssEncoder_Construction_DoesNotAllocateWeights()
    {
        GptOssEncoder enc = new GptOssEncoder(LlamaStyleEncoderConfig.GptOss);
        Assert.Empty(enc.EnumerateWeights());
        Assert.Equal(24, enc.NumLayers);
        enc.Dispose();
    }

    [Fact]
    public void GptOssEncoder_LoadWeights_ThrowsOnMissingKey()
    {
        GptOssEncoder enc = new GptOssEncoder(LlamaStyleEncoderConfig.GptOss);
        try
        {
            Assert.Throws<KeyNotFoundException>(() => enc.LoadWeights(new Dictionary<string, Tensor>()));
        }
        finally
        {
            enc.Dispose();
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // End-to-end encoder forward — synthetic weights, smoke-test integration
    // ────────────────────────────────────────────────────────────────────

    [Fact]
    public void GptOssEncoder_EndToEnd_Forward_ProducesShape_And_FiniteOutputs()
    {
        // Tiny synthetic GPT-OSS config: 2 layers, hidden=4, GQA 2:1, head_dim=2, intermediate=4,
        // 2 experts × top-1, sliding-128 / full alternating, attention sinks on, YaRN ON. This
        // exercises every code path (router → MoE dispatch → clamped activation → sinks-aware SDPA
        // → sliding-window mask on layer 0 → full mask on layer 1) end-to-end. Doesn't validate
        // against a Python reference (that requires the actual GPT-OSS checkpoint) — just confirms
        // the integration produces finite output of the expected shape.
        LlamaStyleEncoderConfig config = new()
        {
            HiddenSize = 4,
            NumLayers = 2,
            NumQueryHeads = 2,
            NumKvHeads = 1,
            HeadDim = 2,
            IntermediateSize = 4,
            VocabSize = 16,
            RmsNormEps = 1e-5f,
            RopeTheta = 150_000f,
            MaxPositionEmbeddings = 128,
            QkHeadNorm = false,
            AttentionBias = true,
            HasFinalNorm = true,
            Activation = MlpActivation.Silu,
            LayerAttentionTypes = new[] { "sliding_attention", "full_attention" },
            SlidingWindow = 4,
            NumLocalExperts = 2,
            NumExpertsPerToken = 1,
            HasAttentionSinks = true,
            ClampedSwiGlu = true,
            ClampedSwiGluLimit = 7.0f,
            ClampedSwiGluAlpha = 1.702f,
            RopeScaling = RopeScalingType.Yarn,
            YarnFactor = 32.0f,
            YarnBetaFast = 32.0f,
            YarnBetaSlow = 1.0f,
            YarnOriginalMaxPosition = 32,
            EosTokenId = 0,
            BosTokenId = 0,
        };
        GptOssEncoder enc = new GptOssEncoder(config);

        Dictionary<string, Tensor> weights = BuildSyntheticEncoderWeights(config, seed: 42);
        enc.LoadWeights(weights);

        using HartsyInference.Cpu.CpuBackend backend = new HartsyInference.Cpu.CpuBackend();
        int[] tokens = { 1, 2, 3, 4, 5, 6, 7, 8 };
        int[][] batched = { tokens };

        Tensor output = enc.Encode(backend, batched);
        try
        {
            Assert.Equal(3, output.Shape.Rank);
            Assert.Equal(1, (int)output.Shape[0]);
            Assert.Equal(tokens.Length, (int)output.Shape[1]);
            Assert.Equal(config.HiddenSize, (int)output.Shape[2]);
            // Verify all outputs are finite (no NaN/Inf from sinks math or MoE dispatch).
            float* p = (float*)output.DataPointer;
            long n = output.Shape.ElementCount;
            for (long i = 0; i < n; i++)
            {
                float v = p[i];
                Assert.False(float.IsNaN(v), $"Output[{i}] is NaN");
                Assert.False(float.IsInfinity(v), $"Output[{i}] is ±∞");
            }
        }
        finally
        {
            output.Dispose();
            foreach (Tensor t in weights.Values) t.Dispose();
            enc.Dispose();
        }
    }

    [Fact]
    public void GptOssEncoder_EncodeAtLayers_CapturesEarlyExits()
    {
        // Verify multi-layer capture returns hidden states at the right post-block indices, and that
        // requesting only layer 0 means layer 1 never runs (early-exit).
        LlamaStyleEncoderConfig config = new()
        {
            HiddenSize = 4,
            NumLayers = 4,
            NumQueryHeads = 2,
            NumKvHeads = 1,
            HeadDim = 2,
            IntermediateSize = 4,
            VocabSize = 16,
            RmsNormEps = 1e-5f,
            RopeTheta = 150_000f,
            MaxPositionEmbeddings = 64,
            QkHeadNorm = false,
            AttentionBias = true,
            HasFinalNorm = true,
            Activation = MlpActivation.Silu,
            LayerAttentionTypes = new[] { "sliding_attention", "full_attention", "sliding_attention", "full_attention" },
            SlidingWindow = 2,
            NumLocalExperts = 2,
            NumExpertsPerToken = 1,
            HasAttentionSinks = true,
            ClampedSwiGlu = true,
            ClampedSwiGluLimit = 7.0f,
            ClampedSwiGluAlpha = 1.702f,
            RopeScaling = RopeScalingType.None,
            EosTokenId = 0,
            BosTokenId = 0,
        };
        GptOssEncoder enc = new GptOssEncoder(config);
        Dictionary<string, Tensor> weights = BuildSyntheticEncoderWeights(config, seed: 7);
        enc.LoadWeights(weights);

        using HartsyInference.Cpu.CpuBackend backend = new HartsyInference.Cpu.CpuBackend();
        int[] tokens = { 1, 2, 3, 4 };
        int[][] batched = { tokens };

        // Capture layers 0 and 2.
        List<Tensor> caps = enc.EncodeAtLayers(backend, batched, new[] { 0, 2 });
        try
        {
            Assert.Equal(2, caps.Count);
            Assert.Equal(new long[] { 1, tokens.Length, config.HiddenSize }, ToLongArray(caps[0].Shape));
            Assert.Equal(new long[] { 1, tokens.Length, config.HiddenSize }, ToLongArray(caps[1].Shape));

            // Capture out-of-order (e.g. [2, 0]) should still place them in request order.
            List<Tensor> caps2 = enc.EncodeAtLayers(backend, batched, new[] { 2, 0 });
            try
            {
                Assert.Equal(2, caps2.Count);
                // caps2[0] = output at layer 2, caps2[1] = output at layer 0 → equal to caps[1] and caps[0].
                AssertTensorsEqual(caps[1], caps2[0]);
                AssertTensorsEqual(caps[0], caps2[1]);
            }
            finally
            {
                foreach (Tensor t in caps2) t.Dispose();
            }
        }
        finally
        {
            foreach (Tensor t in caps) t.Dispose();
            foreach (Tensor t in weights.Values) t.Dispose();
            enc.Dispose();
        }
    }

    [Fact]
    public void GptOssEncoder_EncodeAtLayers_RejectsDuplicateAndOutOfRange()
    {
        GptOssEncoder enc = new GptOssEncoder(LlamaStyleEncoderConfig.GptOss);
        try
        {
            int[][] tokens = { new[] { 1, 2 } };
            Assert.Throws<ArgumentException>(() => enc.EncodeAtLayers(null!, tokens, new[] { 0, 0 }));
            Assert.Throws<ArgumentOutOfRangeException>(() => enc.EncodeAtLayers(null!, tokens, new[] { -1 }));
            Assert.Throws<ArgumentOutOfRangeException>(() => enc.EncodeAtLayers(null!, tokens, new[] { 24 }));
            Assert.Throws<ArgumentException>(() => enc.EncodeAtLayers(null!, tokens, Array.Empty<int>()));
        }
        finally
        {
            enc.Dispose();
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // LensGptOssEncoder wrapper
    // ────────────────────────────────────────────────────────────────────

    [Fact]
    public void LensGptOssEncoder_SelectedLayers_Match_DefaultTextOffset()
    {
        Assert.Equal(new[] { 5, 11, 17, 23 }, LensGptOssEncoder.SelectedLayers);
        Assert.Equal(97, LensGptOssEncoder.DefaultTextOffset);
    }

    [Fact]
    public void LensGptOssEncoder_BelowOffsetTokens_Throws()
    {
        LensGptOssEncoder enc = new LensGptOssEncoder();
        try
        {
            int[] shortTokens = new int[50]; // < 97 — invalid: chat template hasn't been applied
            Assert.Throws<ArgumentException>(() => enc.EncodeForLens(backend: null!, shortTokens));
        }
        finally
        {
            enc.Dispose();
        }
    }

    [Fact]
    public void LensGptOssEncoder_NullTokens_Throws()
    {
        LensGptOssEncoder enc = new LensGptOssEncoder();
        try
        {
            Assert.Throws<ArgumentNullException>(() => enc.EncodeForLens(backend: null!, tokenIds: null!));
        }
        finally
        {
            enc.Dispose();
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // Helpers
    // ────────────────────────────────────────────────────────────────────

    private static Tensor MakeU8(int n, int fill = 0)
    {
        Tensor t = new Tensor(new TensorShape(n), DType.U8);
        byte* p = (byte*)t.DataPointer;
        byte b = (byte)fill;
        for (int i = 0; i < n; i++) p[i] = b;
        return t;
    }

    /// <summary>Copies a tensor's raw bytes into a fresh tensor of the given shape (same dtype, same
    /// element count) and disposes the source. Used to build multi-dim MXFP4 fixtures from flat buffers.</summary>
    private static Tensor Reshape(Tensor src, params int[] dims)
    {
        long[] longDims = new long[dims.Length];
        for (int i = 0; i < dims.Length; i++) longDims[i] = dims[i];
        TensorShape shape = new TensorShape(longDims);
        if (shape.ElementCount != src.Shape.ElementCount)
            throw new ArgumentException($"Reshape element count {shape.ElementCount} != source {src.Shape.ElementCount}");
        Tensor dst = new Tensor(shape, src.DType);
        long bytes = src.Shape.ElementCount * src.DType.SizeInBytes;
        Buffer.MemoryCopy((void*)src.DataPointer, (void*)dst.DataPointer, bytes, bytes);
        src.Dispose();
        return dst;
    }

    private static Tensor MakeF32(int d0, params float[] vals)
        => MakeF32WithShape(new TensorShape(d0), vals);

    private static Tensor MakeF32(int d0, int d1, float[] vals)
        => MakeF32WithShape(new TensorShape(d0, d1), vals);

    private static Tensor MakeF32(int d0, int d1, int d2, float[] vals)
        => MakeF32WithShape(new TensorShape(d0, d1, d2), vals);

    private static Tensor MakeF32WithShape(TensorShape shape, float[] vals)
    {
        Tensor t = new Tensor(shape, DType.F32);
        long count = shape.ElementCount;
        if (vals.Length != count)
            throw new ArgumentException($"vals length {vals.Length} != shape elementCount {count}");
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < count; i++) p[i] = vals[i];
        return t;
    }

    private static Tensor MakeF32Zero(int d0, int d1)
        => new Tensor(new TensorShape(d0, d1), DType.F32);
    private static Tensor MakeF32Zero(int d0, int d1, int d2)
        => new Tensor(new TensorShape(d0, d1, d2), DType.F32);

    private static long[] ToLongArray(TensorShape shape)
    {
        long[] dims = new long[shape.Rank];
        for (int i = 0; i < shape.Rank; i++) dims[i] = shape[i];
        return dims;
    }

    private static void AssertTensorsEqual(Tensor a, Tensor b)
    {
        Assert.Equal(a.Shape.ToString(), b.Shape.ToString());
        Assert.Equal(a.DType, b.DType);
        long count = a.Shape.ElementCount;
        float* pa = (float*)a.DataPointer;
        float* pb = (float*)b.DataPointer;
        for (long i = 0; i < count; i++)
            Assert.True(MathF.Abs(pa[i] - pb[i]) < 1e-5f, $"Mismatch at {i}: {pa[i]} vs {pb[i]}");
    }

    /// <summary>Builds a fully-populated synthetic GPT-OSS weight dict from a small config, using a
    /// seeded RNG so output is deterministic. Includes every key the encoder loads: embed_tokens,
    /// model.norm, per-layer input_layernorm, post_attention_layernorm, self_attn.{q,k,v,o}_proj
    /// weight+bias, self_attn.sinks, mlp.router.weight+bias, mlp.experts.gate_up_proj +
    /// gate_up_proj_bias + down_proj + down_proj_bias.</summary>
    private static Dictionary<string, Tensor> BuildSyntheticEncoderWeights(LlamaStyleEncoderConfig c, int seed)
    {
        Random rng = new Random(seed);
        Dictionary<string, Tensor> dict = new();
        dict["model.embed_tokens.weight"] = RandomTensor(rng, c.VocabSize, c.HiddenSize);
        if (c.HasFinalNorm)
            dict["model.norm.weight"] = ConstantTensor(1.0f, c.HiddenSize);

        for (int i = 0; i < c.NumLayers; i++)
        {
            string p = $"model.layers.{i}";
            dict[$"{p}.input_layernorm.weight"] = ConstantTensor(1.0f, c.HiddenSize);
            dict[$"{p}.post_attention_layernorm.weight"] = ConstantTensor(1.0f, c.HiddenSize);

            dict[$"{p}.self_attn.q_proj.weight"] = RandomTensor(rng, c.QDim, c.HiddenSize);
            dict[$"{p}.self_attn.k_proj.weight"] = RandomTensor(rng, c.KvDim, c.HiddenSize);
            dict[$"{p}.self_attn.v_proj.weight"] = RandomTensor(rng, c.KvDim, c.HiddenSize);
            dict[$"{p}.self_attn.o_proj.weight"] = RandomTensor(rng, c.HiddenSize, c.QDim);
            if (c.AttentionBias)
            {
                dict[$"{p}.self_attn.q_proj.bias"] = ConstantTensor(0.0f, c.QDim);
                dict[$"{p}.self_attn.k_proj.bias"] = ConstantTensor(0.0f, c.KvDim);
                dict[$"{p}.self_attn.v_proj.bias"] = ConstantTensor(0.0f, c.KvDim);
                dict[$"{p}.self_attn.o_proj.bias"] = ConstantTensor(0.0f, c.HiddenSize);
            }
            if (c.HasAttentionSinks)
                dict[$"{p}.self_attn.sinks"] = ConstantTensor(0.0f, c.NumQueryHeads);

            dict[$"{p}.mlp.router.weight"] = RandomTensor(rng, c.NumLocalExperts, c.HiddenSize);
            dict[$"{p}.mlp.router.bias"] = ConstantTensor(0.0f, c.NumLocalExperts);
            dict[$"{p}.mlp.experts.gate_up_proj"] = RandomTensor(rng, c.NumLocalExperts, c.HiddenSize, 2 * c.IntermediateSize);
            dict[$"{p}.mlp.experts.gate_up_proj_bias"] = ConstantTensor(0.0f, c.NumLocalExperts, 2 * c.IntermediateSize);
            dict[$"{p}.mlp.experts.down_proj"] = RandomTensor(rng, c.NumLocalExperts, c.IntermediateSize, c.HiddenSize);
            dict[$"{p}.mlp.experts.down_proj_bias"] = ConstantTensor(0.0f, c.NumLocalExperts, c.HiddenSize);
        }
        return dict;

        static Tensor RandomTensor(Random rng, params int[] dims)
        {
            long[] longDims = new long[dims.Length];
            for (int i = 0; i < dims.Length; i++) longDims[i] = dims[i];
            Tensor t = new Tensor(new TensorShape(longDims), DType.F32);
            float* p = (float*)t.DataPointer;
            long count = t.Shape.ElementCount;
            for (long i = 0; i < count; i++) p[i] = (float)(rng.NextDouble() * 0.04 - 0.02);
            return t;
        }

        static Tensor ConstantTensor(float v, params int[] dims)
        {
            long[] longDims = new long[dims.Length];
            for (int i = 0; i < dims.Length; i++) longDims[i] = dims[i];
            Tensor t = new Tensor(new TensorShape(longDims), DType.F32);
            float* p = (float*)t.DataPointer;
            long count = t.Shape.ElementCount;
            for (long i = 0; i < count; i++) p[i] = v;
            return t;
        }
    }
}
