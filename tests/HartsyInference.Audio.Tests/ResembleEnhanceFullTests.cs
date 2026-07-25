using HartsyInference.Audio.Models.ResembleEnhance;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Synthetic-weights forwards for the resemble-enhance denoiser and UnivNet vocoder using the REAL
/// checkpoint key layout (verified against upstream <c>denoiser/unet.py</c> / <c>univnet/*.py</c> and the
/// enhancer_stage2 DeepSpeed state dict): <c>encoder_blocks/middle_blocks/decoder_blocks</c> with
/// <c>pre_conv</c> + Sequential-indexed PreactResBlocks for the denoiser; <c>blocks.i</c> with
/// <c>convt_pre.1</c>, AMP blocks, <c>conv_blocks.d.1</c>, and residual-conv kernel predictors (legacy
/// <c>weight_g</c>/<c>weight_v</c>) for the vocoder.</summary>
public sealed unsafe class ResembleEnhanceFullTests
{
    private static uint _rng = 0x51A7u;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return ((_rng & 0xFFFF) / 65535f - 0.5f) * 0.1f; }
    private static Tensor Fill(Tensor t) { float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand(); return t; }
    private static Tensor F1(int a) => Fill(new Tensor(new TensorShape(a), DType.F32));
    private static Tensor F3(int a, int b, int c) => Fill(new Tensor(new TensorShape(a, b, c), DType.F32));
    private static Tensor F4(int a, int b, int c, int d) => Fill(new Tensor(new TensorShape(a, b, c, d), DType.F32));
    private static Tensor Ones(int n) { Tensor t = new(new TensorShape(n), DType.F32); float* p = (float*)t.DataPointer; for (int i = 0; i < n; i++) p[i] = 1f; return t; }
    private static Tensor Ones3(int a) { Tensor t = new(new TensorShape(a, 1, 1), DType.F32); float* p = (float*)t.DataPointer; for (int i = 0; i < a; i++) p[i] = 1f; return t; }
    private static Tensor Zeros(int n) => new(new TensorShape(n), DType.F32);

    [Fact]
    public void Denoiser_SyntheticForward_NoisyPcmToFinitePcm()
    {
        using CpuBackend backend = new();
        ResembleDenoiser denoiser = new();
        denoiser.LoadWeights(new ResembleWeightReader(DenoiserWeights()));

        // ~12 frames worth of audio at hop 420.
        float[] pcm = new float[420 * 12];
        uint r = 0x1234u;
        for (int i = 0; i < pcm.Length; i++) { r ^= r << 13; r ^= r >> 17; r ^= r << 5; pcm[i] = ((r & 0xFFFF) / 65535f - 0.5f) * 0.2f; }

        float[] outPcm = denoiser.Denoise(backend, pcm);
        Assert.Equal(pcm.Length, outPcm.Length);
        foreach (float v in outPcm) Assert.True(float.IsFinite(v));
    }

    [Fact]
    public void UnivNet_SyntheticForward_FeaturesToFinitePcm()
    {
        using CpuBackend backend = new();
        // Tiny config so the kernel predictor's [layers*nc*2nc*k] output stays small.
        int condDim = 12, nc = 4, dNoise = 6, kHidden = 8;
        ResembleUnivNet vocoder = new(condDim, seed: 7, dNoise: dNoise, nc: nc, kHidden: kHidden);
        vocoder.LoadWeights(new ResembleWeightReader(VocoderWeights(condDim, nc, dNoise, kHidden)));

        int t = 4;
        using Tensor cond = F3(1, condDim, t);
        float[] wav = vocoder.Forward(backend, cond);
        Assert.Equal(t * 420, wav.Length);
        foreach (float v in wav) Assert.True(float.IsFinite(v));
    }

    private static Dictionary<string, Tensor> DenoiserWeights()
    {
        int[] ch = [16, 32, 64, 128, 256];
        Dictionary<string, Tensor> w = new()
        {
            ["denoiser.net.input_proj.weight"] = F4(ch[0], 3, 3, 3), ["denoiser.net.input_proj.bias"] = F1(ch[0]),
            ["denoiser.net.head.0.weight"] = F4(ch[0], ch[0], 3, 3), ["denoiser.net.head.0.bias"] = F1(ch[0]),
            ["denoiser.net.head.2.weight"] = F4(3, ch[0], 1, 1), ["denoiser.net.head.2.bias"] = F1(3),
        };
        for (int i = 0; i < 4; i++)
        {
            AddUNetBlock(w, $"denoiser.net.encoder_blocks.{i}", ch[i], ch[i + 1]);
        }
        for (int i = 0; i < 2; i++)
        {
            AddUNetBlock(w, $"denoiser.net.middle_blocks.{i}", ch[4], ch[4]);
        }
        for (int i = 0; i < 4; i++)
        {
            AddUNetBlock(w, $"denoiser.net.decoder_blocks.{i}", ch[4 - i], ch[3 - i]);
        }
        return w;
    }

    /// <summary>Upstream UNetBlock: pre_conv + two PreactResBlocks (Sequential indices 0=GN, 2=Conv, 3=GN,
    /// 5=Conv). Down/upsampling is parameter-free nearest resampling.</summary>
    private static void AddUNetBlock(Dictionary<string, Tensor> w, string p, int inC, int outC)
    {
        w[$"{p}.pre_conv.weight"] = F4(outC, inC, 3, 3); w[$"{p}.pre_conv.bias"] = F1(outC);
        foreach (string res in new[] { "res_block1", "res_block2" })
        {
            w[$"{p}.{res}.0.weight"] = Ones(outC); w[$"{p}.{res}.0.bias"] = F1(outC);
            w[$"{p}.{res}.2.weight"] = F4(outC, outC, 3, 3); w[$"{p}.{res}.2.bias"] = F1(outC);
            w[$"{p}.{res}.3.weight"] = Ones(outC); w[$"{p}.{res}.3.bias"] = F1(outC);
            w[$"{p}.{res}.5.weight"] = F4(outC, outC, 3, 3); w[$"{p}.{res}.5.bias"] = F1(outC);
        }
    }

    private static Dictionary<string, Tensor> VocoderWeights(int condDim, int nc, int dNoise, int kHidden)
    {
        int nLayers = 4, k = 3;
        int[] strides = [7, 5, 4, 3];
        Dictionary<string, Tensor> w = new()
        {
            ["vocoder.conv_pre.weight_g"] = Ones3(nc), ["vocoder.conv_pre.weight_v"] = F3(nc, dNoise, 7),
            ["vocoder.conv_pre.bias"] = F1(nc),
            ["vocoder.conv_post.1.weight_g"] = Ones3(1), ["vocoder.conv_post.1.weight_v"] = F3(1, nc, 7),
            ["vocoder.conv_post.1.bias"] = F1(1),
        };
        for (int i = 0; i < strides.Length; i++)
        {
            string p = $"vocoder.blocks.{i}";
            w[$"{p}.convt_pre.1.weight_g"] = Ones3(nc); w[$"{p}.convt_pre.1.weight_v"] = F3(nc, nc, 2 * strides[i]);
            w[$"{p}.convt_pre.1.bias"] = F1(nc);
            for (int l = 0; l < 3; l++)
            {
                string amp = $"{p}.amp_block.{l}";
                w[$"{amp}.0.weight_g"] = Ones3(nc); w[$"{amp}.0.weight_v"] = F3(nc, nc, 3); w[$"{amp}.0.bias"] = F1(nc);
                w[$"{amp}.1.act.log_alpha"] = Zeros(nc); w[$"{amp}.1.act.log_beta"] = Zeros(nc);
                w[$"{amp}.1.upsample.filter"] = F3(1, 1, 12); w[$"{amp}.1.downsample.lowpass.filter"] = F3(1, 1, 12);
                w[$"{amp}.2.weight_g"] = Ones3(nc); w[$"{amp}.2.weight_v"] = F3(nc, nc, 3); w[$"{amp}.2.bias"] = F1(nc);
            }
            for (int d = 0; d < 4; d++)
            {
                w[$"{p}.conv_blocks.{d}.1.weight_g"] = Ones3(nc); w[$"{p}.conv_blocks.{d}.1.weight_v"] = F3(nc, nc, 3);
                w[$"{p}.conv_blocks.{d}.1.bias"] = F1(nc);
            }
            string kp = $"{p}.kernel_predictor";
            w[$"{kp}.input_conv.0.weight_g"] = Ones3(kHidden); w[$"{kp}.input_conv.0.weight_v"] = F3(kHidden, condDim, 5);
            w[$"{kp}.input_conv.0.bias"] = F1(kHidden);
            for (int rIdx = 0; rIdx < 3; rIdx++)
            {
                foreach (int seq in new[] { 1, 3 })
                {
                    w[$"{kp}.residual_convs.{rIdx}.{seq}.weight_g"] = Ones3(kHidden);
                    w[$"{kp}.residual_convs.{rIdx}.{seq}.weight_v"] = F3(kHidden, kHidden, 3);
                    w[$"{kp}.residual_convs.{rIdx}.{seq}.bias"] = F1(kHidden);
                }
            }
            int kOut = nLayers * nc * (2 * nc) * k;
            w[$"{kp}.kernel_conv.weight_g"] = Ones3(kOut); w[$"{kp}.kernel_conv.weight_v"] = F3(kOut, kHidden, 3);
            w[$"{kp}.kernel_conv.bias"] = F1(kOut);
            int bOut = nLayers * 2 * nc;
            w[$"{kp}.bias_conv.weight_g"] = Ones3(bOut); w[$"{kp}.bias_conv.weight_v"] = F3(bOut, kHidden, 3);
            w[$"{kp}.bias_conv.bias"] = F1(bOut);
        }
        return w;
    }
}
