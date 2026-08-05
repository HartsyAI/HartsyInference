using Xunit;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.ModelAssets.BlockScale;
using HartsyInference.ModelAssets.Nvfp4;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Covers the MiniMax-H3 conditioning tower's quantized weight path and a full forward over a tiny
/// synthetic nvfp4 checkpoint. The two Unit tests pin down the facts that were verified against the real
/// <c>qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors</c> and that a plausible-looking wrong implementation still
/// passes a shape/finiteness check: block scales are read through NVIDIA's <b>swizzled</b> blocked layout, and
/// <c>pre_quant_scale</c> multiplies the <b>activation</b> (the quantizer stored <c>W/s</c>).</summary>
public sealed unsafe class MiniMaxH3TextEncoderTests
{
    /// <summary>E4M3 byte for 1.0 (exponent bias 7, zero mantissa).</summary>
    private const byte One = 0x38;

    /// <summary>Packs two E2M1 nibbles, even element in the high nibble.</summary>
    private static byte Pack(byte even, byte odd) => (byte)((even << 4) | odd);

    private static int _seed = 7;

    private static float NextUniform()
    {
        _seed = _seed * 1103515245 + 12345;
        return ((_seed >> 16) & 0x7fff) / 32768f;
    }

    private static Tensor Filled(TensorShape shape, DType dtype, byte value)
    {
        Tensor t = new Tensor(shape, dtype);
        byte* p = (byte*)t.DataPointer;
        long bytes = dtype.ComputeByteCount(shape.ElementCount);
        for (long i = 0; i < bytes; i++) p[i] = value;
        return t;
    }

    private static Tensor Scalar(float value)
    {
        Tensor t = new Tensor(new TensorShape(1), DType.F32);
        ((float*)t.DataPointer)[0] = value;
        return t;
    }

    private static Tensor F32Filled(TensorShape shape, float value)
    {
        Tensor t = new Tensor(shape, DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = value;
        return t;
    }

    /// <summary>Builds an nvfp4 linear whose every stored element is E2M1 <c>1.0</c> (nibble 2), with all block
    /// scales at E4M3 1.0 and the requested global scale — so the dequantized weight is a constant matrix.</summary>
    private static Dictionary<string, Tensor> UnitBank(string prefix, int outFeatures, int inFeatures,
        float globalScale, Dictionary<string, Tensor>? into = null)
    {
        Dictionary<string, Tensor> w = into ?? new Dictionary<string, Tensor>();
        int scaleRows = 128 * ((outFeatures + 127) / 128);
        w[$"{prefix}.weight"] = Filled(new TensorShape(outFeatures, inFeatures / 2), DType.U8, Pack(2, 2));
        w[$"{prefix}.weight_scale"] = Filled(new TensorShape(scaleRows, inFeatures / 16), DType.F8E4M3, One);
        w[$"{prefix}.weight_scale_2"] = Scalar(globalScale);
        return w;
    }

    /// <summary>Builds an nvfp4 linear with varied nibbles and varied (positive, non-NaN) E4M3 block scales, so a
    /// dequant that drops the mantissa, mis-rounds, or mis-indexes a group cannot pass by symmetry.</summary>
    private static Dictionary<string, Tensor> RandomBank(string prefix, int outFeatures, int inFeatures,
        float globalScale, Dictionary<string, Tensor>? into = null)
    {
        Dictionary<string, Tensor> w = into ?? new Dictionary<string, Tensor>();
        int scaleRows = 128 * ((outFeatures + 127) / 128);
        Tensor packed = new Tensor(new TensorShape(outFeatures, inFeatures / 2), DType.U8);
        byte* p = (byte*)packed.DataPointer;
        for (long i = 0; i < packed.ElementCount; i++) p[i] = (byte)(NextUniform() * 256f);
        Tensor scales = new Tensor(new TensorShape(scaleRows, inFeatures / 16), DType.F8E4M3);
        byte* s = (byte*)scales.DataPointer;
        for (long i = 0; i < scales.ElementCount; i++) s[i] = (byte)(0x20 + (int)(NextUniform() * 48f));
        w[$"{prefix}.weight"] = packed;
        w[$"{prefix}.weight_scale"] = scales;
        w[$"{prefix}.weight_scale_2"] = Scalar(globalScale);
        return w;
    }

    /// <summary>An identity activation reads the whole weight out of a forward exactly: <c>output[s, o] = W[o, s]</c>
    /// with a single non-zero product per dot, so no accumulation rounding hides a dequant difference.</summary>
    private static Tensor IdentityInput(int inFeatures)
    {
        Tensor input = new Tensor(new TensorShape(1, inFeatures, inFeatures), DType.F32);
        float* p = (float*)input.DataPointer;
        for (int i = 0; i < inFeatures; i++) p[(long)i * inFeatures + i] = 1.0f;
        return input;
    }

    [Fact]
    public void Bf16DequantMatchesTheF32CodecNarrowedToBf16()
    {
        // The forward dequantizes straight to BF16 instead of going through Nvfp4Codec's F32 slice; the two must
        // agree bit for bit, or the fused narrowing has drifted from the codec (E4M3 table, swizzle, or rounding).
        const int OutFeatures = 128;
        const int InFeatures = 64;
        Dictionary<string, Tensor> weights = RandomBank("q", OutFeatures, InFeatures, globalScale: 0.03f);

        Tensor f32 = new Tensor(new TensorShape(OutFeatures, InFeatures), DType.F32);
        Nvfp4Codec.DequantExpertSlice(
            weights["q.weight"].Reshape(new TensorShape(1, OutFeatures, InFeatures / 2)),
            weights["q.weight_scale"].Reshape(new TensorShape(1, weights["q.weight_scale"].Shape[0], InFeatures / 16)),
            weights["q.weight_scale_2"].Reshape(new TensorShape(1)), 0, f32);
        using Tensor bf16 = f32.CastTo(DType.BF16);
        using Tensor expected = bf16.CastTo(DType.F32);
        f32.Dispose();

        Nvfp4Linear linear = Nvfp4Linear.Load(weights, "q");
        IBackend backend = new CpuBackend();
        using Tensor input = IdentityInput(InFeatures);
        using Tensor output = new Tensor(new TensorShape(1, InFeatures, OutFeatures), DType.F32);
        linear.Forward(backend, output, input);

        float* e = (float*)expected.DataPointer;
        float* o = (float*)output.DataPointer;
        for (int i = 0; i < InFeatures; i++)
            for (int j = 0; j < OutFeatures; j++)
                Assert.Equal(e[(long)j * InFeatures + i], o[(long)i * OutFeatures + j]);
    }

    [Fact]
    public void SharedDequantScratchServesDifferentlyShapedLayers()
    {
        // The scratch is sized by the widest layer and reused by every other one; a narrow layer must see its own
        // freshly dequantized bytes, not the tail of the wide layer that ran before it.
        const int WideOut = 256, NarrowOut = 128, InFeatures = 64;
        Dictionary<string, Tensor> weights = RandomBank("wide", WideOut, InFeatures, globalScale: 0.03f);
        RandomBank("narrow", NarrowOut, InFeatures, globalScale: 0.07f, into: weights);

        Nvfp4Linear wide = Nvfp4Linear.Load(weights, "wide");
        Nvfp4Linear narrow = Nvfp4Linear.Load(weights, "narrow");
        Assert.Equal((long)WideOut * InFeatures, wide.DequantScratchElements);

        IBackend backend = new CpuBackend();
        using Tensor input = IdentityInput(InFeatures);
        using Tensor reference = new Tensor(new TensorShape(1, InFeatures, NarrowOut), DType.F32);
        narrow.Forward(backend, reference, input);

        using Tensor scratch = Nvfp4Linear.CreateDequantScratch(wide.DequantScratchElements);
        using Tensor wideOut = new Tensor(new TensorShape(1, InFeatures, WideOut), DType.F32);
        using Tensor narrowOut = new Tensor(new TensorShape(1, InFeatures, NarrowOut), DType.F32);
        wide.Forward(backend, wideOut, input, scratch);
        narrow.Forward(backend, narrowOut, input, scratch);

        float* r = (float*)reference.DataPointer;
        float* n = (float*)narrowOut.DataPointer;
        for (long i = 0; i < narrowOut.ElementCount; i++) Assert.Equal(r[i], n[i]);

        Assert.Throws<ArgumentException>(() =>
        {
            using Tensor tooSmall = Nvfp4Linear.CreateDequantScratch(narrow.DequantScratchElements);
            wide.Forward(backend, wideOut, input, tooSmall);
        });
    }

    [Fact]
    public void BlockScalesAreReadThroughTheSwizzledBlockedLayout()
    {
        // Row 1's first block scale lives at flat 16 under the blocked layout and at flat 4 under a plain
        // [out, in/16] reading; the two are given different values so only one answer can come out.
        const int OutFeatures = 128;
        const int InFeatures = 64;
        Dictionary<string, Tensor> weights = UnitBank("q", OutFeatures, InFeatures, globalScale: 1.0f);
        byte* scales = (byte*)weights["q.weight_scale"].DataPointer;
        Assert.Equal(16, BlockScaleSwizzle.SwizzledIndex(1, 0, InFeatures / 16));
        scales[16] = 0x40;                       // 2.0 — the swizzled home of (row 1, block 0)
        scales[1 * (InFeatures / 16) + 0] = 0x30; // 0.5 — where a row-major reading would look

        Nvfp4Linear linear = Nvfp4Linear.Load(weights, "q");
        IBackend backend = new CpuBackend();
        Tensor input = new Tensor(new TensorShape(1, 1, InFeatures), DType.F32);
        ((float*)input.DataPointer)[0] = 1.0f;
        Tensor output = new Tensor(new TensorShape(1, 1, OutFeatures), DType.F32);
        linear.Forward(backend, output, input);

        // input is a one-hot on column 0, so output[r] is exactly W[r, 0] = 1.0 * blockScale(r, 0).
        float* o = (float*)output.DataPointer;
        Assert.Equal(2.0f, o[1], 5);
        Assert.Equal(1.0f, o[0], 5);
        Assert.Equal(0.5f, o[32], 5); // flat 4 is the swizzled home of row 32, which must see the 0.5
    }

    [Fact]
    public void PreQuantScaleMultipliesTheActivationNotTheWeight()
    {
        // The AWQ quantizer stored W/s, so the forward must undo it on the activation side: y = (x*s)·(W/s)^T.
        // Applying s to the weight instead (or inverting it) lands nowhere near the expected sum.
        const int OutFeatures = 128;
        const int InFeatures = 64;
        Dictionary<string, Tensor> weights = UnitBank("o", OutFeatures, InFeatures, globalScale: 1.0f);
        Tensor preQuant = new Tensor(new TensorShape(InFeatures), DType.F32);
        float* s = (float*)preQuant.DataPointer;
        for (int i = 0; i < InFeatures; i++) s[i] = i + 1;
        weights["o.pre_quant_scale"] = preQuant;

        Nvfp4Linear linear = Nvfp4Linear.Load(weights, "o");
        IBackend backend = new CpuBackend();
        Tensor input = F32Filled(new TensorShape(1, 1, InFeatures), 1.0f);
        Tensor output = new Tensor(new TensorShape(1, 1, OutFeatures), DType.F32);
        linear.Forward(backend, output, input);

        float expected = InFeatures * (InFeatures + 1) / 2f; // sum of 1..64 with W ≡ 1
        Assert.Equal(expected, ((float*)output.DataPointer)[0], 2);
    }

    [Fact]
    [Trait("Category", "SyntheticSmoke")]
    public void SyntheticNvfp4CheckpointProducesFiniteHiddenStates()
    {
        const int Hidden = 128, HeadDim = 32, QueryHeads = 8, KvHeads = 4, Intermediate = 256, Vocab = 64, Layers = 2;
        Dictionary<string, Tensor> weights = new Dictionary<string, Tensor>();

        Tensor embed = new Tensor(new TensorShape(Vocab, Hidden), DType.I8);
        sbyte* e = (sbyte*)embed.DataPointer;
        for (long i = 0; i < embed.ElementCount; i++) e[i] = (sbyte)(NextUniform() * 40f - 20f);
        weights["model.embed_tokens.weight"] = embed;
        weights["model.embed_tokens.weight_scale"] = F32Filled(new TensorShape(Vocab, 1), 0.01f);

        for (int l = 0; l < Layers; l++)
        {
            string p = $"model.layers.{l}";
            weights[$"{p}.input_layernorm.weight"] = F32Filled(new TensorShape(Hidden), 1.0f);
            weights[$"{p}.post_attention_layernorm.weight"] = F32Filled(new TensorShape(Hidden), 1.0f);
            weights[$"{p}.self_attn.q_norm.weight"] = F32Filled(new TensorShape(HeadDim), 1.0f);
            weights[$"{p}.self_attn.k_norm.weight"] = F32Filled(new TensorShape(HeadDim), 1.0f);
            UnitBank($"{p}.self_attn.q_proj", QueryHeads * HeadDim, Hidden, 0.02f, weights);
            UnitBank($"{p}.self_attn.k_proj", KvHeads * HeadDim, Hidden, 0.02f, weights);
            UnitBank($"{p}.self_attn.v_proj", KvHeads * HeadDim, Hidden, 0.02f, weights);
            UnitBank($"{p}.self_attn.o_proj", Hidden, QueryHeads * HeadDim, 0.02f, weights);
            UnitBank($"{p}.mlp.gate_proj", Intermediate, Hidden, 0.02f, weights);
            UnitBank($"{p}.mlp.up_proj", Intermediate, Hidden, 0.02f, weights);
            UnitBank($"{p}.mlp.down_proj", Hidden, Intermediate, 0.02f, weights);
            weights[$"{p}.self_attn.o_proj.pre_quant_scale"] = F32Filled(new TensorShape(QueryHeads * HeadDim), 1.5f);
            weights[$"{p}.mlp.down_proj.pre_quant_scale"] = F32Filled(new TensorShape(Intermediate), 0.5f);
        }

        using MiniMaxH3TextEncoder encoder = new MiniMaxH3TextEncoder();
        encoder.LoadWeights(weights);
        Assert.Equal(Layers, encoder.NumLayers);
        Assert.Equal(Hidden, encoder.HiddenSize);
        Assert.Null(encoder.VisionConfig);

        MiniMaxH3TextEncoding.Encoded presentation = MiniMaxH3TextEncoding.Build(
            text => { int[] ids = new int[text.Length]; for (int i = 0; i < text.Length; i++) ids[i] = text[i] % Vocab; return ids; },
            "a cat");
        IBackend backend = new CpuBackend();
        MiniMaxH3TextEncoder.Result result = encoder.Encode(backend, presentation);

        Assert.Equal(presentation.Length, (int)result.HiddenStates.Shape[0]);
        Assert.Equal(Hidden, (int)result.HiddenStates.Shape[1]);
        float* h = (float*)result.HiddenStates.DataPointer;
        for (long i = 0; i < result.HiddenStates.ElementCount; i++)
            Assert.True(float.IsFinite(h[i]), $"non-finite hidden state at {i}");
        Assert.Equal([(0, presentation.Length, MiniMaxH3TextEncoding.TextTag)], result.TagRuns);
        result.HiddenStates.Dispose();
    }
}
