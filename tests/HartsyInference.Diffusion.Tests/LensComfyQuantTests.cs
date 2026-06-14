using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.BlockScale;
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.ModelHandler.Mxfp8;
using HartsyInference.ModelHandler.Nvfp4;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Unit tests for the ComfyUI-packaged Microsoft Lens checkpoint formats
/// (<c>Comfy-Org/Lens</c>): the MXFP8 DiT codec, the NVFP4 GPT-OSS text-encoder codec, the NVIDIA
/// swizzled block-scale index math, and the ComfyUI split-file converter (key remap + expert-bias
/// rename + comfy_quant/tokenizer_json dropping). The dequant formulas + swizzle inversion were
/// validated byte-/noise-exact against <c>comfy.float</c> in the python reference; these tests pin the
/// C# implementation against hand-computed golden values using powers-of-two scales (exact in E4M3/E8M0).</summary>
public sealed unsafe class LensComfyQuantTests
{
    // ────────────────────────────────────────────────────────────────────
    // Block-scale swizzle (NVIDIA d-block-scaling layout)
    // ────────────────────────────────────────────────────────────────────

    [Fact]
    public void Swizzle_MatchesHandComputed_ForSingleBlock()
    {
        // Hp=128, Wp=4 → ncb=1, g=0; flat(r,c) = (r%32)*16 + (r/32)*4 + c.
        Assert.Equal(0, BlockScaleSwizzle.SwizzledIndex(0, 0, 4));
        Assert.Equal(16, BlockScaleSwizzle.SwizzledIndex(1, 0, 4));
        Assert.Equal(4, BlockScaleSwizzle.SwizzledIndex(32, 0, 4));
        Assert.Equal(1, BlockScaleSwizzle.SwizzledIndex(0, 1, 4));
        Assert.Equal(511, BlockScaleSwizzle.SwizzledIndex(127, 3, 4));
    }

    [Fact]
    public void Swizzle_IsBijection_OverPaddedBuffer()
    {
        const int Hp = 256, Wp = 8;            // 2 row-blocks, 2 col-blocks
        bool[] seen = new bool[Hp * Wp];
        for (int r = 0; r < Hp; r++)
            for (int c = 0; c < Wp; c++)
            {
                long idx = BlockScaleSwizzle.SwizzledIndex(r, c, Wp);
                Assert.InRange(idx, 0, Hp * Wp - 1);
                Assert.False(seen[idx], $"collision at logical ({r},{c}) → {idx}");
                seen[idx] = true;
            }
        for (int i = 0; i < seen.Length; i++) Assert.True(seen[i], $"index {i} never produced");
    }

    // ────────────────────────────────────────────────────────────────────
    // MXFP8 DiT codec
    // ────────────────────────────────────────────────────────────────────

    [Fact]
    public void Mxfp8_DequantLinear_AppliesBlockScale_AndProducesBf16()
    {
        const int outDim = 128, inDim = 64;    // out aligned to 128; in = 2 blocks of 32
        // Weight all 1.0 (exact in F8E4M3).
        Tensor wF32 = Filled(1.0f, outDim, inDim);
        Tensor weight = wF32.CastTo(DType.F8E4M3); wF32.Dispose();

        // Swizzled E8M0 scale [128, 4] (paddedCols=4): block 0 → exp byte 128 (2^1), block 1 → 127 (2^0).
        Tensor scale = new Tensor(new TensorShape(outDim, 4), DType.U8);
        byte* sp = (byte*)scale.DataPointer;
        for (int r = 0; r < outDim; r++)
        {
            sp[BlockScaleSwizzle.SwizzledIndex(r, 0, 4)] = 128; // 2^(128-127)=2
            sp[BlockScaleSwizzle.SwizzledIndex(r, 1, 4)] = 127; // 2^0=1
        }

        Tensor outBf16 = Mxfp8Codec.DequantLinear(weight, scale);
        Tensor outF32 = outBf16.CastTo(DType.F32);
        try
        {
            Assert.Equal(DType.BF16, outBf16.DType);
            float* o = (float*)outF32.DataPointer;
            for (int r = 0; r < outDim; r++)
            {
                Assert.Equal(2.0f, o[r * inDim + 0]);   // block 0 → ×2
                Assert.Equal(2.0f, o[r * inDim + 31]);
                Assert.Equal(1.0f, o[r * inDim + 32]);  // block 1 → ×1
                Assert.Equal(1.0f, o[r * inDim + 63]);
            }
        }
        finally { outBf16.Dispose(); outF32.Dispose(); weight.Dispose(); scale.Dispose(); }
    }

    [Fact]
    public void Mxfp8_DequantInPlace_StripsCompanions_LeavesPlainBf16Alone()
    {
        Tensor wF32 = Filled(1.0f, 128, 32);
        Dictionary<string, Tensor> dict = new()
        {
            ["transformer_blocks.0.attn.img_qkv.weight"] = wF32.CastTo(DType.F8E4M3),
            ["transformer_blocks.0.attn.img_qkv.weight_scale"] = ConstU8(128, 4, 127),
            ["transformer_blocks.0.attn.img_qkv.comfy_quant"] = ConstU8(74, 1, 0),
            ["transformer_blocks.0.attn.img_qkv.bias"] = new Tensor(new TensorShape(4608), DType.BF16),
            ["img_in.weight"] = new Tensor(new TensorShape(1536, 128), DType.BF16),
        };
        wF32.Dispose();
        int n = Mxfp8Codec.DequantInPlace(dict);
        try
        {
            Assert.Equal(1, n);
            Assert.Equal(DType.BF16, dict["transformer_blocks.0.attn.img_qkv.weight"].DType);
            Assert.False(dict.ContainsKey("transformer_blocks.0.attn.img_qkv.weight_scale"));
            Assert.False(dict.ContainsKey("transformer_blocks.0.attn.img_qkv.comfy_quant"));
            Assert.True(dict.ContainsKey("transformer_blocks.0.attn.img_qkv.bias"));   // untouched
            Assert.Equal(DType.BF16, dict["img_in.weight"].DType);                      // plain BF16 untouched
        }
        finally { foreach (Tensor t in dict.Values) t.Dispose(); }
    }

    // ────────────────────────────────────────────────────────────────────
    // NVFP4 GPT-OSS expert codec
    // ────────────────────────────────────────────────────────────────────

    [Fact]
    public void Nvfp4_DequantExpert_DecodesNibbles_Scales_AndTransposes()
    {
        const int E = 1, outDim = 128, inDim = 16; // 1 expert, out aligned to 128, in = 1 block of 16
        int inHalf = inDim / 2;
        // Packed byte 0x21: high nibble = 2 (+1.0, even element), low nibble = 1 (+0.5, odd element).
        Tensor weight = ConstU8(E * outDim * inHalf, 1, 0x21);
        Tensor weight3d = Reshape3d(weight, E, outDim, inHalf);

        // Block scale = 1.0 (exact E4M3), swizzled [128, 4].
        Tensor scaleF32 = Filled(1.0f, outDim, 4);
        Tensor scaleBlock = new Tensor(new TensorShape(E, outDim, 4), DType.F8E4M3);
        // Fill expert 0's swizzled scale with E4M3(1.0) everywhere.
        Tensor oneF8 = Filled(1.0f, outDim, 4).CastTo(DType.F8E4M3);
        Buffer.MemoryCopy((void*)oneF8.DataPointer, (void*)scaleBlock.DataPointer, outDim * 4, outDim * 4);
        scaleF32.Dispose(); oneF8.Dispose();

        Tensor global = Filled(2.0f, E);       // per-expert global = 2.0

        Tensor outF32 = Nvfp4Codec.DequantExpert(weight3d, scaleBlock, global);
        try
        {
            // Output is transposed: [E, in, out].
            Assert.Equal(new long[] { E, inDim, outDim }, new[] { outF32.Shape[0], outF32.Shape[1], outF32.Shape[2] });
            float* o = (float*)outF32.DataPointer;
            for (int r = 0; r < outDim; r++)
            {
                // even in-col (0,2,...) = 1.0*global*1.0 = 2.0 ; odd = 0.5*2*1 = 1.0
                Assert.Equal(2.0f, o[(0 * inDim + 0) * outDim + r]);
                Assert.Equal(1.0f, o[(0 * inDim + 1) * outDim + r]);
                Assert.Equal(2.0f, o[(0 * inDim + 14) * outDim + r]);
                Assert.Equal(1.0f, o[(0 * inDim + 15) * outDim + r]);
            }
        }
        finally { outF32.Dispose(); weight3d.Dispose(); scaleBlock.Dispose(); global.Dispose(); }
    }

    // ────────────────────────────────────────────────────────────────────
    // ComfyUI text-encoder converter (remap + bias rename + drop blobs)
    // ────────────────────────────────────────────────────────────────────

    [Fact]
    public void ConvertComfyTextEncoder_Remaps_RenamesBias_DropsBlobs_DequantsExperts()
    {
        const int E = 1, outDim = 128, inHalf = 8; // gate_up-ish tiny: in=16
        Tensor guWeight = Reshape3d(ConstU8(E * outDim * inHalf, 1, 0x21), E, outDim, inHalf);
        Tensor guScale = new Tensor(new TensorShape(E, outDim, 4), DType.F8E4M3);
        Tensor oneF8 = Filled(1.0f, outDim, 4).CastTo(DType.F8E4M3);
        Buffer.MemoryCopy((void*)oneF8.DataPointer, (void*)guScale.DataPointer, outDim * 4, outDim * 4); oneF8.Dispose();

        Dictionary<string, Tensor> te = new()
        {
            ["embed_tokens.weight"] = new Tensor(new TensorShape(16, 8), DType.BF16),
            ["norm.weight"] = new Tensor(new TensorShape(8), DType.BF16),
            ["tokenizer_json"] = ConstU8(10, 1, 65),
            ["layers.0.self_attn.q_proj.weight"] = new Tensor(new TensorShape(8, 8), DType.BF16),
            ["layers.0.mlp.experts.gate_up_proj.weight"] = guWeight,
            ["layers.0.mlp.experts.gate_up_proj.weight_scale"] = guScale,
            ["layers.0.mlp.experts.gate_up_proj.weight_scale_2"] = Filled(2.0f, E),
            ["layers.0.mlp.experts.gate_up_proj.bias"] = new Tensor(new TensorShape(E, outDim), DType.BF16),
            ["layers.0.mlp.experts.gate_up_proj.comfy_quant"] = ConstU8(74, 1, 0),
        };

        Dictionary<string, Tensor> outd = LensCheckpointConverter.ConvertComfyTextEncoder(te);
        try
        {
            // model. prefix applied; tokenizer_json + comfy_quant dropped.
            Assert.True(outd.ContainsKey("model.embed_tokens.weight"));
            Assert.True(outd.ContainsKey("model.norm.weight"));
            Assert.DoesNotContain(outd.Keys, k => k.Contains("tokenizer_json"));
            Assert.DoesNotContain(outd.Keys, k => k.Contains("comfy_quant"));
            // BF16 weights cast to F32 for the CPU encoder GEMM.
            Assert.Equal(DType.F32, outd["model.layers.0.self_attn.q_proj.weight"].DType);
            Assert.Equal(DType.F32, outd["model.embed_tokens.weight"].DType);
            // Expert weight dequantized to the bare HF key, transposed [E, in, out] = [1, 16, 128], F32.
            Tensor gu = outd["model.layers.0.mlp.experts.gate_up_proj"];
            Assert.Equal(DType.F32, gu.DType);
            Assert.Equal(new long[] { E, 16, outDim }, new[] { gu.Shape[0], gu.Shape[1], gu.Shape[2] });
            // Expert bias renamed `.bias` → `_bias` and cast to F32.
            Assert.True(outd.ContainsKey("model.layers.0.mlp.experts.gate_up_proj_bias"));
            Assert.Equal(DType.F32, outd["model.layers.0.mlp.experts.gate_up_proj_bias"].DType);
        }
        finally { foreach (Tensor t in outd.Values) t.Dispose(); }
    }

    // ────────────────────────────────────────────────────────────────────
    // Helpers
    // ────────────────────────────────────────────────────────────────────

    private static Tensor Filled(float v, params int[] dims)
    {
        long[] d = new long[dims.Length];
        for (int i = 0; i < dims.Length; i++) d[i] = dims[i];
        Tensor t = new Tensor(new TensorShape(d), DType.F32);
        float* p = (float*)t.DataPointer;
        long n = t.Shape.ElementCount;
        for (long i = 0; i < n; i++) p[i] = v;
        return t;
    }

    private static Tensor ConstU8(int count, int _unused, byte val)
    {
        Tensor t = new Tensor(new TensorShape(count), DType.U8);
        byte* p = (byte*)t.DataPointer;
        for (int i = 0; i < count; i++) p[i] = val;
        return t;
    }

    private static Tensor Reshape3d(Tensor src, int d0, int d1, int d2)
    {
        Tensor dst = new Tensor(new TensorShape(d0, d1, d2), src.DType);
        long bytes = src.Shape.ElementCount * src.DType.SizeInBytes;
        Buffer.MemoryCopy((void*)src.DataPointer, (void*)dst.DataPointer, bytes, bytes);
        src.Dispose();
        return dst;
    }
}
