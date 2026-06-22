using HartsyInference.Audio.Models.ResembleEnhance;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Synthetic-weights forwards for the staged resemble-enhance components: the 2D-STFT denoiser
/// (noisy PCM → finite PCM) and the UnivNet LVCNet vocoder (mel → finite PCM).</summary>
public sealed unsafe class ResembleEnhanceFullTests
{
    private static uint _rng = 0x51A7u;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return ((_rng & 0xFFFF) / 65535f - 0.5f) * 0.1f; }
    private static Tensor Fill(Tensor t) { float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand(); return t; }
    private static Tensor F1(int a) => Fill(new Tensor(new TensorShape(a), DType.F32));
    private static Tensor F3(int a, int b, int c) => Fill(new Tensor(new TensorShape(a, b, c), DType.F32));
    private static Tensor F4(int a, int b, int c, int d) => Fill(new Tensor(new TensorShape(a, b, c, d), DType.F32));
    private static Tensor Ones(int n) { Tensor t = new(new TensorShape(n), DType.F32); float* p = (float*)t.DataPointer; for (int i = 0; i < n; i++) p[i] = 1f; return t; }

    [Fact]
    public void Denoiser_SyntheticForward_NoisyPcmToFinitePcm()
    {
        using CpuBackend backend = new();
        ResembleDenoiser denoiser = new();
        denoiser.LoadWeights(DenoiserWeights());

        // ~10 frames worth of audio at hop 420.
        float[] pcm = new float[420 * 12];
        uint r = 0x1234u;
        for (int i = 0; i < pcm.Length; i++) { r ^= r << 13; r ^= r >> 17; r ^= r << 5; pcm[i] = ((r & 0xFFFF) / 65535f - 0.5f) * 0.2f; }

        float[] outPcm = denoiser.Denoise(backend, pcm);
        Assert.True(outPcm.Length > 0);
        foreach (float v in outPcm) Assert.True(float.IsFinite(v));
    }

    [Fact]
    public void UnivNet_SyntheticForward_MelToFinitePcm()
    {
        using CpuBackend backend = new();
        // Tiny config so the kernel-predictor's [nDil*2*nc*nc*k] output stays small.
        int mels = 8, nc = 4, dNoise = 6, condExtra = 4, kHidden = 8;
        ResembleUnivNet vocoder = new(mels, seed: 7, dNoise: dNoise, nc: nc, condExtra: condExtra, kHidden: kHidden);
        vocoder.LoadWeights(VocoderWeights(mels, nc, dNoise, condExtra, kHidden));

        int t = 4;
        using Tensor mel = F3(1, mels, t);
        float[] wav = vocoder.Forward(backend, mel);
        Assert.True(wav.Length > 0);
        foreach (float v in wav) Assert.True(float.IsFinite(v));
    }

    private static Dictionary<string, Tensor> DenoiserWeights()
    {
        int[] ch = [16, 32, 64, 128, 256];
        Dictionary<string, Tensor> w = new()
        {
            ["denoiser.net.in_conv.weight"] = F4(ch[0], 3, 3, 3), ["denoiser.net.in_conv.bias"] = F1(ch[0]),
            ["denoiser.net.out_conv.weight"] = F4(3, ch[0], 3, 3), ["denoiser.net.out_conv.bias"] = F1(3),
        };
        for (int i = 0; i < 4; i++)
        {
            int inC = ch[i], outC = ch[i + 1];
            string p = $"denoiser.net.down.{i}";
            w[$"{p}.conv1.weight"] = F4(outC, inC, 3, 3); w[$"{p}.conv1.bias"] = F1(outC);
            w[$"{p}.norm1.weight"] = Ones(outC); w[$"{p}.norm1.bias"] = F1(outC);
            w[$"{p}.conv2.weight"] = F4(outC, outC, 3, 3); w[$"{p}.conv2.bias"] = F1(outC);
            w[$"{p}.norm2.weight"] = Ones(outC); w[$"{p}.norm2.bias"] = F1(outC);
            w[$"{p}.downsample.weight"] = F4(outC, outC, 3, 3); w[$"{p}.downsample.bias"] = F1(outC);
        }
        for (int i = 0; i < 2; i++)
        {
            string p = $"denoiser.net.mid.{i}";
            w[$"{p}.conv1.weight"] = F4(ch[4], ch[4], 3, 3); w[$"{p}.conv1.bias"] = F1(ch[4]);
            w[$"{p}.norm1.weight"] = Ones(ch[4]); w[$"{p}.norm1.bias"] = F1(ch[4]);
            w[$"{p}.conv2.weight"] = F4(ch[4], ch[4], 3, 3); w[$"{p}.conv2.bias"] = F1(ch[4]);
            w[$"{p}.norm2.weight"] = Ones(ch[4]); w[$"{p}.norm2.bias"] = F1(ch[4]);
        }
        for (int i = 0; i < 4; i++)
        {
            int inC = ch[4 - i], outC = ch[3 - i];
            string p = $"denoiser.net.up.{i}";
            w[$"{p}.upsample.weight"] = F4(inC, inC, 4, 4); w[$"{p}.upsample.bias"] = F1(inC);
            w[$"{p}.conv1.weight"] = F4(outC, 2 * inC, 3, 3); w[$"{p}.conv1.bias"] = F1(outC);
            w[$"{p}.norm1.weight"] = Ones(outC); w[$"{p}.norm1.bias"] = F1(outC);
            w[$"{p}.conv2.weight"] = F4(outC, outC, 3, 3); w[$"{p}.conv2.bias"] = F1(outC);
            w[$"{p}.norm2.weight"] = Ones(outC); w[$"{p}.norm2.bias"] = F1(outC);
        }
        return w;
    }

    private static Dictionary<string, Tensor> VocoderWeights(int mels, int nc, int dNoise, int condExtra, int kHidden)
    {
        int condDim = mels + condExtra, nDil = 4, k = 3;
        int[] strides = [7, 5, 4, 3];
        Dictionary<string, Tensor> w = new()
        {
            ["vocoder.conv_pre.weight"] = F3(nc, dNoise, 7), ["vocoder.conv_pre.bias"] = F1(nc),
            ["vocoder.conv_post.weight"] = F3(1, nc, 7), ["vocoder.conv_post.bias"] = F1(1),
        };
        for (int i = 0; i < strides.Length; i++)
        {
            string p = $"vocoder.res_stack.{i}";
            int kk = 2 * strides[i];
            w[$"{p}.convt_pre.weight"] = F3(nc, nc, kk); w[$"{p}.convt_pre.bias"] = F1(nc);
            string kp = $"{p}.kernel_predictor";
            w[$"{kp}.input_conv.0.weight"] = F3(kHidden, condDim, 3); w[$"{kp}.input_conv.0.bias"] = F1(kHidden);
            int kOut = nDil * 2 * nc * (nc * k);
            w[$"{kp}.kernel_conv.weight"] = F3(kOut, kHidden, 3); w[$"{kp}.kernel_conv.bias"] = F1(kOut);
            int bOut = nDil * 2 * nc;
            w[$"{kp}.bias_conv.weight"] = F3(bOut, kHidden, 3); w[$"{kp}.bias_conv.bias"] = F1(bOut);
        }
        return w;
    }
}
