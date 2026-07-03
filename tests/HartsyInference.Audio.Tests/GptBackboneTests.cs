using HartsyInference.Audio.Models.Bark;
using HartsyInference.Audio.Models.LanguageModels.Gpt;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Validates the shared GPT-2 backbone (reused by Bark's three stages, and reusable by XTTS /
/// ChatTTS) end-to-end on a tiny synthetic-weight model — shapes flow, output is finite, causal and
/// non-causal both run. The shared transformer is the load-bearing piece; the per-model checkpoints are
/// gated.</summary>
public sealed unsafe class GptBackboneTests
{
    private static uint _rng = 0x12345u;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return ((_rng & 0xFFFF) / 65535f - 0.5f) * 0.2f; }

    private static Tensor Fill(Tensor t) { float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand(); return t; }
    private static Tensor Filled(int a) => Fill(new Tensor(new TensorShape(a), DType.F32));
    private static Tensor Filled(int a, int b) => Fill(new Tensor(new TensorShape(a, b), DType.F32));
    private static Tensor Filled(int a, int b, int c) => Fill(new Tensor(new TensorShape(a, b, c), DType.F32));

    private static Dictionary<string, Tensor> SyntheticGptWeights(GptConfig c)
    {
        int h = c.Hidden;
        Dictionary<string, Tensor> w = new()
        {
            ["pos"] = Filled(c.BlockSize, h),
            ["lnf.weight"] = Ones(h),
            ["lnf.bias"] = Filled(h),
        };
        for (int i = 0; i < c.NumLayers; i++)
        {
            string p = $"blk.{i}";
            w[$"{p}.layernorm_1.weight"] = Ones(h);
            w[$"{p}.layernorm_1.bias"] = Filled(h);
            w[$"{p}.attn.att_proj.weight"] = Filled(3 * h, h);
            w[$"{p}.attn.out_proj.weight"] = Filled(h, h);
            w[$"{p}.layernorm_2.weight"] = Ones(h);
            w[$"{p}.layernorm_2.bias"] = Filled(h);
            w[$"{p}.mlp.in_proj.weight"] = Filled(c.MlpDim, h);
            w[$"{p}.mlp.out_proj.weight"] = Filled(h, c.MlpDim);
        }
        return w;
    }

    private static Tensor Ones(int n)
    {
        Tensor t = new(new TensorShape(n), DType.F32);
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < n; i++) p[i] = 1f;
        return t;
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Backbone_Forward_ProducesFiniteOutput(bool nonCausal)
    {
        GptConfig cfg = new() { Hidden = 16, NumLayers = 2, NumHeads = 4, BlockSize = 32 };
        using CpuBackend backend = new();
        using GptBackbone bb = new(cfg);
        bb.LoadWeights(SyntheticGptWeights(cfg), "pos", "blk", "lnf.weight", "lnf.bias");

        using Tensor embeds = Filled(1, 6, cfg.Hidden);
        using Tensor outT = bb.Forward(backend, embeds, nonCausal);
        Assert.Equal(1, (int)outT.Shape[0]);
        Assert.Equal(6, (int)outT.Shape[1]);
        Assert.Equal(cfg.Hidden, (int)outT.Shape[2]);
        float* op = (float*)outT.DataPointer;
        for (long i = 0; i < outT.ElementCount; i++) Assert.True(float.IsFinite(op[i]));
    }

    [Fact]
    public void Backbone_ForwardStep_MatchesFullSequenceForward()
    {
        GptConfig cfg = new() { Hidden = 16, NumLayers = 2, NumHeads = 4, BlockSize = 32 };
        using CpuBackend backend = new();
        using GptBackbone bb = new(cfg);
        bb.LoadWeights(SyntheticGptWeights(cfg), "pos", "blk", "lnf.weight", "lnf.bias");

        const int prefillLen = 4, totalLen = 9;
        using Tensor allEmbeds = Filled(1, totalLen, cfg.Hidden);
        float* ap = (float*)allEmbeds.DataPointer;

        // Incremental: prefill the first 4 positions, then step tokens 4..8 one at a time.
        GptKvCache cache = bb.CreateCache();
        Tensor prefill = new(new TensorShape(1, prefillLen, cfg.Hidden), DType.F32);
        new ReadOnlySpan<float>(ap, prefillLen * cfg.Hidden).CopyTo(prefill.AsSpan<float>());
        using Tensor prefillOut = bb.Forward(backend, prefill, nonCausal: false, cache);
        prefill.Dispose();
        Assert.Equal(prefillLen, cache.Length);

        float[][] stepOuts = new float[totalLen][];
        for (int pos = prefillLen; pos < totalLen; pos++)
        {
            Tensor one = new(new TensorShape(1, 1, cfg.Hidden), DType.F32);
            new ReadOnlySpan<float>(ap + (long)pos * cfg.Hidden, cfg.Hidden).CopyTo(one.AsSpan<float>());
            using Tensor stepOut = bb.ForwardStep(backend, one, cache);
            one.Dispose();
            stepOuts[pos] = new Span<float>((void*)stepOut.DataPointer, cfg.Hidden).ToArray();
        }
        Assert.Equal(totalLen, cache.Length);

        // Reference: one full causal forward over all 9 positions.
        using Tensor fullOut = bb.Forward(backend, allEmbeds, nonCausal: false);
        float* fp = (float*)fullOut.DataPointer;
        for (int pos = prefillLen; pos < totalLen; pos++)
        {
            for (int c = 0; c < cfg.Hidden; c++)
            {
                Assert.True(MathF.Abs(fp[(long)pos * cfg.Hidden + c] - stepOuts[pos][c]) < 1e-4f,
                    $"pos {pos} ch {c}: full={fp[(long)pos * cfg.Hidden + c]} step={stepOuts[pos][c]}");
            }
        }
    }

    [Fact]
    public void BarkConfig_Presets_HaveExpectedShape()
    {
        Assert.Equal(1024, BarkConfig.Full.Stage.Hidden);
        Assert.Equal(24, BarkConfig.Full.Stage.NumLayers);
        Assert.Equal(768, BarkConfig.Small.Stage.Hidden);
        Assert.Equal(8, BarkConfig.Full.NumCodebooks);
        Assert.Equal(10_048, BarkConfig.Full.TextEncodingOffset);
    }
}
