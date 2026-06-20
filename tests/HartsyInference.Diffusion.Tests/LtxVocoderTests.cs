using Xunit;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Music;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Structural tests for the LTX-2 audio vocoder. Two parts:
/// <list type="bullet">
/// <item><b>Forward</b> — a tiny <see cref="LtxBigVganGenerator"/> in both activation modes
/// (<see cref="LtxVocoderActivation.SnakeBeta"/>, <see cref="LtxVocoderActivation.LeakyRelu"/>) decodes to a finite
/// waveform of the expected length (T × ∏ upsample factors).</item>
/// <item><b>Detection</b> — <see cref="LtxAudioVocoder"/> loads a weight dict carrying the <i>real</i> single-file
/// checkpoint key names for the BWE dual-stage and the single-stage variants, picking the right variant and sample
/// rate. This is the permanent guard against the real-checkpoint <c>KeyNotFoundException</c> (<c>conv_pre</c> /
/// <c>resblocks</c> / <c>act_post</c> / <c>conv_post</c>) that this test file was added to catch.</item>
/// </list>
/// Numerics vs the real checkpoint are validation-pending; these gate wiring, naming, and finiteness.</summary>
public unsafe class LtxVocoderTests
{
    private static int _seed = 7;

    [Theory]
    [InlineData(LtxVocoderActivation.SnakeBeta)]
    [InlineData(LtxVocoderActivation.LeakyRelu)]
    public void Generator_TinyConfig_ProducesFiniteWaveformOfExpectedLength(LtxVocoderActivation act)
    {
        CpuBackend backend = new();
        int inCh = 4, hidden = 8, outCh = 2;
        int[] factors = [2, 2], kernels = [4, 4];
        int[] resKernels = [3, 7, 11];
        int[][] resDil = [[1, 3, 5], [1, 3, 5], [1, 3, 5]];

        LtxBigVganGenerator gen = new(
            inCh, hidden, outCh, factors, kernels, resKernels, resDil,
            applyTanh: act == LtxVocoderActivation.LeakyRelu, activation: act);
        gen.LoadWeights(BuildGenerator("g", inCh, hidden, outCh, factors, kernels, resKernels, resDil, act), prefix: "g");

        int t = 5;
        Tensor x = Rand([1, inCh, t]);
        Tensor wav = gen.Forward(backend, x);

        Assert.Equal(outCh, (int)wav.Shape[1]);
        Assert.Equal(t * 2 * 2, (int)wav.Shape[2]);   // ∏ factors = 4
        float* p = (float*)wav.DataPointer;
        for (long i = 0; i < wav.Shape.ElementCount; i++) Assert.True(float.IsFinite(p[i]), $"non-finite at {i}");
        gen.Dispose();
    }

    [Fact]
    public void Vocoder_BweRealKeyNames_LoadsAndReports48kHz()
    {
        Dictionary<string, Tensor> w = new();
        // Main generator: hidden 1536, 6 upsample stages; BWE generator: hidden 512, 5 stages. SnakeBeta.
        AddGeneratorKeys(w, "vocoder.vocoder", upsampleStages: 6, resPerUp: 3, dilCount: 3, snake: true);
        AddGeneratorKeys(w, "vocoder.bwe_generator", upsampleStages: 5, resPerUp: 3, dilCount: 3, snake: true);
        w["vocoder.mel_stft.stft_fn.forward_basis"] = Stub();
        w["vocoder.mel_stft.mel_basis"] = Stub();

        LtxAudioVocoder voc = new();
        voc.LoadWeights(w);   // throws KeyNotFound if any real name is mis-mapped
        Assert.Equal(48000, voc.SampleRate);
        voc.Dispose();
    }

    [Fact]
    public void Vocoder_SingleStageRealKeyNames_LoadsAndReports24kHz()
    {
        Dictionary<string, Tensor> w = new();
        // Single-stage: hidden 1024, 5 upsample stages, leaky-ReLU (no activation params).
        AddGeneratorKeys(w, "vocoder", upsampleStages: 5, resPerUp: 3, dilCount: 3, snake: false);

        LtxAudioVocoder voc = new();
        voc.LoadWeights(w);
        Assert.Equal(24000, voc.SampleRate);
        voc.Dispose();
    }

    /// <summary>Emits every key name <see cref="LtxBigVganGenerator.LoadWeights"/> requests, with [1]-stub tensors
    /// (load stores references without checking shapes, so this exercises naming only). Mirrors the real single-file
    /// naming: <c>conv_pre</c>, <c>ups.N</c>, <c>resblocks.N.convs{1,2}.M</c> (+ <c>acts{1,2}.M.act.{alpha,beta}</c>
    /// and <c>act_post.act.{alpha,beta}</c> for SnakeBeta), <c>conv_post</c>.</summary>
    private static void AddGeneratorKeys(Dictionary<string, Tensor> w, string p, int upsampleStages, int resPerUp, int dilCount, bool snake)
    {
        w[$"{p}.conv_pre.weight"] = Stub(); w[$"{p}.conv_pre.bias"] = Stub();
        for (int i = 0; i < upsampleStages; i++) { w[$"{p}.ups.{i}.weight"] = Stub(); w[$"{p}.ups.{i}.bias"] = Stub(); }
        for (int n = 0; n < upsampleStages * resPerUp; n++)
            for (int m = 0; m < dilCount; m++)
            {
                w[$"{p}.resblocks.{n}.convs1.{m}.weight"] = Stub(); w[$"{p}.resblocks.{n}.convs1.{m}.bias"] = Stub();
                w[$"{p}.resblocks.{n}.convs2.{m}.weight"] = Stub(); w[$"{p}.resblocks.{n}.convs2.{m}.bias"] = Stub();
                if (snake)
                {
                    w[$"{p}.resblocks.{n}.acts1.{m}.act.alpha"] = Stub(); w[$"{p}.resblocks.{n}.acts1.{m}.act.beta"] = Stub();
                    w[$"{p}.resblocks.{n}.acts2.{m}.act.alpha"] = Stub(); w[$"{p}.resblocks.{n}.acts2.{m}.act.beta"] = Stub();
                }
            }
        if (snake) { w[$"{p}.act_post.act.alpha"] = Stub(); w[$"{p}.act_post.act.beta"] = Stub(); }
        w[$"{p}.conv_post.weight"] = Stub(); w[$"{p}.conv_post.bias"] = Stub();
    }

    /// <summary>Real-shaped tiny generator weights (for the forward test): channels halve each upsample stage.</summary>
    private static Dictionary<string, Tensor> BuildGenerator(
        string p, int inCh, int hidden, int outCh, int[] factors, int[] kernels,
        int[] resKernels, int[][] resDil, LtxVocoderActivation act)
    {
        bool snake = act == LtxVocoderActivation.SnakeBeta;
        Dictionary<string, Tensor> w = new();
        w[$"{p}.conv_pre.weight"] = Rand([hidden, inCh, 7]); w[$"{p}.conv_pre.bias"] = Rand([hidden]);
        int ch = hidden;
        int resPerUp = resKernels.Length;
        for (int i = 0; i < factors.Length; i++)
        {
            int outC = ch / 2;
            w[$"{p}.ups.{i}.weight"] = Rand([ch, outC, kernels[i]]); w[$"{p}.ups.{i}.bias"] = Rand([outC]);
            for (int j = 0; j < resPerUp; j++)
            {
                int n = i * resPerUp + j;
                for (int m = 0; m < resDil[j].Length; m++)
                {
                    w[$"{p}.resblocks.{n}.convs1.{m}.weight"] = Rand([outC, outC, resKernels[j]]);
                    w[$"{p}.resblocks.{n}.convs1.{m}.bias"] = Rand([outC]);
                    w[$"{p}.resblocks.{n}.convs2.{m}.weight"] = Rand([outC, outC, resKernels[j]]);
                    w[$"{p}.resblocks.{n}.convs2.{m}.bias"] = Rand([outC]);
                    if (snake)
                    {
                        w[$"{p}.resblocks.{n}.acts1.{m}.act.alpha"] = Rand([outC]); w[$"{p}.resblocks.{n}.acts1.{m}.act.beta"] = Rand([outC]);
                        w[$"{p}.resblocks.{n}.acts2.{m}.act.alpha"] = Rand([outC]); w[$"{p}.resblocks.{n}.acts2.{m}.act.beta"] = Rand([outC]);
                    }
                }
            }
            ch = outC;
        }
        if (snake) { w[$"{p}.act_post.act.alpha"] = Rand([ch]); w[$"{p}.act_post.act.beta"] = Rand([ch]); }
        w[$"{p}.conv_post.weight"] = Rand([outCh, ch, 7]); w[$"{p}.conv_post.bias"] = Rand([outCh]);
        return w;
    }

    private static Tensor Stub() => new(new TensorShape(1), DType.F32);

    private static Tensor Rand(long[] dims)
    {
        Tensor t = new(new TensorShape(dims), DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new(_seed++);
        for (long i = 0; i < t.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 0.1 - 0.05);
        return t;
    }
}
