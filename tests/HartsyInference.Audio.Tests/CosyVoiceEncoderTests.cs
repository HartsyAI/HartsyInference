using System.Collections.Generic;
using HartsyInference.Audio.Models.CosyVoice;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Synthetic-weights forward tests for the two real CosyVoice 2 encoders. The S3 speech tokenizer
/// must turn a mel into finite FSQ features / valid <c>[0, 6561)</c> tokens; the UpsampleConformerEncoder
/// must produce a finite <c>[1, T·2, out]</c> sequence. Random tiny configs on the CPU backend — no
/// checkpoint needed (the exact-weights path is covered by the env-gated generation test).</summary>
public sealed unsafe class CosyVoiceEncoderTests
{
    private static Tensor Random(TensorShape shape, int seed, float scale = 0.05f)
    {
        Tensor t = new(shape, DType.F32);
        float* p = (float*)t.DataPointer;
        uint s = (uint)(seed * 2654435761u + 1u);
        for (long i = 0; i < t.ElementCount; i++)
        {
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            p[i] = ((s & 0xFFFFFF) / (float)0xFFFFFF - 0.5f) * 2f * scale;
        }
        return t;
    }

    private static void AddNorm(Dictionary<string, Tensor> w, string prefix, int dim, int seed)
    {
        w[$"{prefix}.weight"] = Random(new TensorShape(dim), seed, 0.1f);
        w[$"{prefix}.bias"] = Random(new TensorShape(dim), seed + 1, 0.05f);
    }

    private static void AddLinear(Dictionary<string, Tensor> w, string prefix, int outDim, int inDim, int seed, bool bias = true)
    {
        w[$"{prefix}.weight"] = Random(new TensorShape(outDim, inDim), seed);
        if (bias) w[$"{prefix}.bias"] = Random(new TensorShape(outDim), seed + 1);
    }

    private static bool AllFinite(Tensor t)
    {
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++)
            if (!float.IsFinite(p[i])) return false;
        return true;
    }

    [Fact]
    public void S3Tokenizer_SyntheticWeights_ProducesValidTokens()
    {
        using CpuBackend backend = new();
        const int hidden = 32, numHeads = 8;            // matches the encoder's 8-head default
        const int melBins = 80, melLen = 64;

        Dictionary<string, Tensor> w = new();
        // Conv stem: 80 → hidden (stride 2), hidden → hidden (stride 2). Kernel 3.
        w["subsample.0.weight"] = Random(new TensorShape(hidden, melBins, 3), 1);
        w["subsample.0.bias"] = Random(new TensorShape(hidden), 2);
        w["subsample.1.weight"] = Random(new TensorShape(hidden, hidden, 3), 3);
        w["subsample.1.bias"] = Random(new TensorShape(hidden), 4);
        for (int i = 0; i < 6; i++)
        {
            string e = $"encoders.{i}";
            AddNorm(w, $"{e}.norm1", hidden, 10 + i * 30);
            AddLinear(w, $"{e}.self_attn.linear_q", hidden, hidden, 12 + i * 30);
            AddLinear(w, $"{e}.self_attn.linear_k", hidden, hidden, 14 + i * 30);
            AddLinear(w, $"{e}.self_attn.linear_v", hidden, hidden, 16 + i * 30);
            AddLinear(w, $"{e}.self_attn.linear_out", hidden, hidden, 18 + i * 30);
            AddNorm(w, $"{e}.norm2", hidden, 20 + i * 30);
            AddLinear(w, $"{e}.feed_forward.w_1", hidden * 2, hidden, 22 + i * 30);
            AddLinear(w, $"{e}.feed_forward.w_2", hidden, hidden * 2, 24 + i * 30);
        }
        AddNorm(w, "after_norm", hidden, 500);
        AddLinear(w, "quantizer.project_down", 8, hidden, 600);

        using S3Tokenizer s3 = new();
        s3.LoadWeights(w);
        using Tensor mel = Random(new TensorShape(1, melBins, melLen), 999, 0.5f);
        int[] tokens = s3.Forward(backend, mel);

        // 4× subsample → ~melLen/4 tokens, each a valid base-3^8 code.
        Assert.True(tokens.Length >= melLen / 4 - 2 && tokens.Length <= melLen / 4 + 2,
            $"expected ~{melLen / 4} tokens, got {tokens.Length}");
        foreach (int tok in tokens)
            Assert.InRange(tok, 0, s3.VocabSize - 1);
        _ = numHeads;
        foreach (Tensor t in w.Values) t.Dispose();
    }

    [Fact]
    public void UpsampleConformerEncoder_SyntheticWeights_ProducesFiniteUpsampledSequence()
    {
        using CpuBackend backend = new();
        const int inputSize = 24, outputSize = 32, numHeads = 8;
        const int tTokens = 10;
        const int preBlocks = 2, postBlocks = 1;

        Dictionary<string, Tensor> w = new();
        AddLinear(w, "embed.out.0", outputSize, inputSize, 1);
        BuildConformer(w, "encoders", preBlocks, outputSize, 100);
        // up_layer ConvTranspose1d: [Cin, Cout, K] = [out, out, 4], stride 2.
        w["up_layer.weight"] = Random(new TensorShape(outputSize, outputSize, 4), 700);
        w["up_layer.bias"] = Random(new TensorShape(outputSize), 701);
        BuildConformer(w, "up_encoders", postBlocks, outputSize, 2000);
        AddNorm(w, "after_norm", outputSize, 9000);

        using UpsampleConformerEncoder enc = new(outputSize, numHeads, preBlocks, postBlocks);
        enc.LoadWeights(w);
        using Tensor tokEmb = Random(new TensorShape(1, tTokens, inputSize), 4242, 0.3f);
        using Tensor outT = enc.Forward(backend, tokEmb, inputSize);

        Assert.Equal(1, (int)outT.Shape[0]);
        Assert.Equal(tTokens * UpsampleConformerEncoder.TokenMelRatio, (int)outT.Shape[1]);
        Assert.Equal(outputSize, (int)outT.Shape[2]);
        Assert.True(AllFinite(outT), "UpsampleConformerEncoder output must be finite.");
        foreach (Tensor t in w.Values) t.Dispose();
    }

    private static void BuildConformer(Dictionary<string, Tensor> w, string stack, int blocks, int c, int seedBase)
    {
        const int ffDim = 2;            // tiny FFN inner dim
        for (int i = 0; i < blocks; i++)
        {
            string b = $"{stack}.{i}";
            int s = seedBase + i * 200;
            AddNorm(w, $"{b}.norm_ff_macaron", c, s);
            AddLinear(w, $"{b}.feed_forward_macaron.w_1", c * ffDim, c, s + 2);
            AddLinear(w, $"{b}.feed_forward_macaron.w_2", c, c * ffDim, s + 4);
            AddNorm(w, $"{b}.norm_mha", c, s + 6);
            AddLinear(w, $"{b}.self_attn.linear_q", c, c, s + 8);
            AddLinear(w, $"{b}.self_attn.linear_k", c, c, s + 10);
            AddLinear(w, $"{b}.self_attn.linear_v", c, c, s + 12);
            AddLinear(w, $"{b}.self_attn.linear_out", c, c, s + 14);
            AddNorm(w, $"{b}.norm_conv", c, s + 16);
            // pointwise_conv1: C → 2C (kernel 1).
            w[$"{b}.conv_module.pointwise_conv1.weight"] = Random(new TensorShape(2 * c, c, 1), s + 18);
            w[$"{b}.conv_module.pointwise_conv1.bias"] = Random(new TensorShape(2 * c), s + 19);
            // depthwise_conv: groups = C, weight [C, 1, K].
            w[$"{b}.conv_module.depthwise_conv.weight"] = Random(new TensorShape(c, 1, 5), s + 20);
            w[$"{b}.conv_module.depthwise_conv.bias"] = Random(new TensorShape(c), s + 21);
            AddNorm(w, $"{b}.conv_module.norm", c, s + 22);
            w[$"{b}.conv_module.pointwise_conv2.weight"] = Random(new TensorShape(c, c, 1), s + 24);
            w[$"{b}.conv_module.pointwise_conv2.bias"] = Random(new TensorShape(c), s + 25);
            AddNorm(w, $"{b}.norm_ff", c, s + 26);
            AddLinear(w, $"{b}.feed_forward.w_1", c * ffDim, c, s + 28);
            AddLinear(w, $"{b}.feed_forward.w_2", c, c * ffDim, s + 30);
            AddNorm(w, $"{b}.norm_final", c, s + 32);
        }
    }
}
