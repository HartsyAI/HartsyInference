using HartsyInference.Audio.Models.Demucs;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Full-assembly synthetic-weights forward of <see cref="HtDemucs"/> on a tiny config (small
/// channels/depth, short stereo input, small n_fft) — the freq dimension collapses so the merge / cross-transformer
/// path executes. Asserts the pipeline returns 4 finite stereo stems. Mirrors the synthetic-weight pattern in
/// <c>DemucsTests</c> (CpuBackend, random F-filled tensors).</summary>
public sealed unsafe class DemucsAssemblyTests
{
    private static uint _rng = 0x517Au;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return ((_rng & 0xFFFF) / 65535f - 0.5f) * 0.2f; }
    private static Tensor Fill(Tensor t) { float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand(); return t; }
    private static Tensor F1(int a) => Fill(new Tensor(new TensorShape(a), DType.F32));
    private static Tensor F2(int a, int b) => Fill(new Tensor(new TensorShape(a, b), DType.F32));
    private static Tensor F3(int a, int b, int c) => Fill(new Tensor(new TensorShape(a, b, c), DType.F32));
    private static Tensor F4(int a, int b, int c, int d) => Fill(new Tensor(new TensorShape(a, b, c, d), DType.F32));
    private static Tensor Ones(int n) { Tensor t = new(new TensorShape(n), DType.F32); float* p = (float*)t.DataPointer; for (int i = 0; i < n; i++) p[i] = 1f; return t; }

    [Fact]
    public void HtDemucs_SyntheticForward_FourFiniteStems()
    {
        HtDemucsConfig cfg = new()
        {
            AudioChannels = 2,
            Channels = 4,
            Growth = 2,
            Depth = 2,
            NFft = 16,
            HopLength = 4,
            KernelSize = 8,
            Stride = 4,
            BottomChannels = 8,
            TLayers = 3,
            THeads = 2,
            HiddenScale = 2.0f,
        };
        using CpuBackend backend = new();
        DemucsPipeline pipe = new(cfg);
        pipe.LoadWeights(BuildWeights(cfg));

        int length = 256;
        float[] left = new float[length];
        float[] right = new float[length];
        for (int i = 0; i < length; i++) { left[i] = MathF.Sin(i * 0.07f) * 0.3f; right[i] = MathF.Cos(i * 0.05f) * 0.3f; }

        (float[] Left, float[] Right)[] stems = pipe.Separate(backend, left, right);

        Assert.Equal(cfg.NumSources, stems.Length);
        foreach ((float[] l, float[] r) in stems)
        {
            Assert.Equal(length, l.Length);
            Assert.Equal(length, r.Length);
            for (int i = 0; i < length; i++) { Assert.True(float.IsFinite(l[i])); Assert.True(float.IsFinite(r[i])); }
        }
        pipe.Dispose();
    }

    private static Dictionary<string, Tensor> BuildWeights(HtDemucsConfig cfg)
    {
        Dictionary<string, Tensor> w = new();
        int specIn = cfg.SpecInChannels;      // 4
        int timeIn = cfg.AudioChannels;       // 2
        int k = cfg.KernelSize;
        int srcs = cfg.NumSources;
        int[] encCh = new int[cfg.Depth];
        int ch = cfg.Channels;
        for (int i = 0; i < cfg.Depth; i++) { encCh[i] = ch; ch *= cfg.Growth; }

        // Encoders (freq 2D + time 1D).
        for (int i = 0; i < cfg.Depth; i++)
        {
            int inSpec = i == 0 ? specIn : encCh[i - 1];
            int inTime = i == 0 ? timeIn : encCh[i - 1];
            int outCh = encCh[i];
            w[$"encoder.{i}.conv.weight"] = F4(outCh, inSpec, k, 1); w[$"encoder.{i}.conv.bias"] = F1(outCh);
            w[$"encoder.{i}.rewrite.weight"] = F4(2 * outCh, outCh, 1, 1); w[$"encoder.{i}.rewrite.bias"] = F1(2 * outCh);
            w[$"tencoder.{i}.conv.weight"] = F3(outCh, inTime, k); w[$"tencoder.{i}.conv.bias"] = F1(outCh);
            w[$"tencoder.{i}.rewrite.weight"] = F3(2 * outCh, outCh, 1); w[$"tencoder.{i}.rewrite.bias"] = F1(2 * outCh);
        }

        // Decoders mirror encoders; the final layer (di==0) emits srcs·(audio channels).
        for (int i = 0; i < cfg.Depth; i++)
        {
            int di = cfg.Depth - 1 - i;
            int inCh = encCh[di];
            int outSpec = di == 0 ? srcs * specIn : encCh[di - 1];
            int outTime = di == 0 ? srcs * timeIn : encCh[di - 1];
            // DemucsConvBlock decode: rewrite inCh→2·outCh, GLU→outCh, transposed conv outCh→outCh.
            w[$"decoder.{i}.rewrite.weight"] = F4(2 * outSpec, inCh, 1, 1); w[$"decoder.{i}.rewrite.bias"] = F1(2 * outSpec);
            w[$"decoder.{i}.conv.weight"] = F4(outSpec, outSpec, k, 1); w[$"decoder.{i}.conv.bias"] = F1(outSpec);
            w[$"tdecoder.{i}.rewrite.weight"] = F3(2 * outTime, inCh, 1); w[$"tdecoder.{i}.rewrite.bias"] = F1(2 * outTime);
            w[$"tdecoder.{i}.conv.weight"] = F3(outTime, outTime, k); w[$"tdecoder.{i}.conv.bias"] = F1(outTime);
        }

        // freq_emb table: [Fq0, channels0]. Fq0 = nfft/2.
        w["freq_emb.embedding.weight"] = F2(cfg.NFft / 2, cfg.Channels);

        // Cross-transformer + channel up/down samplers.
        int bottom = cfg.BottomChannels, ffn = cfg.TransformerFfn, encDeep = encCh[cfg.Depth - 1];
        w["crosstransformer.norm_in.weight"] = Ones(bottom); w["crosstransformer.norm_in.bias"] = F1(bottom);
        w["crosstransformer.norm_in_t.weight"] = Ones(bottom); w["crosstransformer.norm_in_t.bias"] = F1(bottom);
        w["crosstransformer.weight_pos_embed"] = F1(1);
        w["crosstransformer.channel_upsampler.weight"] = F3(bottom, encDeep, 1); w["crosstransformer.channel_upsampler.bias"] = F1(bottom);
        w["crosstransformer.channel_downsampler.weight"] = F3(encDeep, bottom, 1); w["crosstransformer.channel_downsampler.bias"] = F1(encDeep);
        w["crosstransformer.channel_upsampler_t.weight"] = F3(bottom, encDeep, 1); w["crosstransformer.channel_upsampler_t.bias"] = F1(bottom);
        w["crosstransformer.channel_downsampler_t.weight"] = F3(encDeep, bottom, 1); w["crosstransformer.channel_downsampler_t.bias"] = F1(encDeep);
        for (int i = 0; i < cfg.TLayers; i++)
        {
            bool cross = i % 2 == 1;
            foreach (string stream in new[] { "layers", "layers_t" })
            {
                string p = $"crosstransformer.{stream}.{i}";
                string a = cross ? "cross_attn" : "self_attn";
                foreach (string proj in new[] { "q_proj", "k_proj", "v_proj", "out_proj" })
                { w[$"{p}.{a}.{proj}.weight"] = F2(bottom, bottom); w[$"{p}.{a}.{proj}.bias"] = F1(bottom); }
                w[$"{p}.norm1.weight"] = Ones(bottom); w[$"{p}.norm1.bias"] = F1(bottom);
                w[$"{p}.norm2.weight"] = Ones(bottom); w[$"{p}.norm2.bias"] = F1(bottom);
                w[$"{p}.norm3.weight"] = Ones(bottom); w[$"{p}.norm3.bias"] = F1(bottom);
                w[$"{p}.linear1.weight"] = F2(ffn, bottom); w[$"{p}.linear1.bias"] = F1(ffn);
                w[$"{p}.linear2.weight"] = F2(bottom, ffn); w[$"{p}.linear2.bias"] = F1(bottom);
                w[$"{p}.gamma_1.scale"] = F1(bottom); w[$"{p}.gamma_2.scale"] = F1(bottom);
            }
        }
        return w;
    }
}
