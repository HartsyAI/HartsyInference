using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Layers;
using HartsyInference.Audio.Models.Kokoro;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.StyleTts2;

/// <summary>StyleTTS2 LibriTTS <c>type: hifigan</c> generator (<c>Modules/hifigan.py</c> <c>Generator</c>): a
/// HiFi-GAN vocoder that turns the decoder features <c>x [1, 512, T]</c> + the decoder style <c>s [1, 128]</c>
/// + <c>f0 [1, T]</c> into a raw 24 kHz waveform — four <c>ConvTranspose1d</c> upsample stages (rates 10·5·3·2 =
/// 300 = hop) each with a per-stage harmonic-source noise-conv + AdaIN/Snake <c>noise_res</c> injection and a
/// 3-kernel MRF (AdaIN/Snake <c>resblocks</c>), Snake <c>alphas</c> between stages, then <c>conv_post</c> + tanh.
/// Unlike Kokoro's iSTFTNet generator this outputs the waveform directly (no iSTFT) and feeds the 1-channel
/// harmonic source (not its STFT) into the noise convs. Reuses <see cref="AdaSnakeResLoader"/> (== the reference
/// <c>AdaINResBlock1</c>) + <see cref="NsfVocoderDsp"/>. The additive noise branch is unused, so it is deterministic.</summary>
public sealed unsafe class StyleHifiGanGenerator
{
    private const int InitialChannel = 512;
    private static readonly int[] UpsampleRates = [10, 5, 3, 2];
    private static readonly int[] UpsampleKernels = [20, 10, 6, 4];
    private static readonly int[] ResblockKernels = [3, 7, 11];
    private readonly int _sampleRate;

    // m_source.l_linear (9 harmonics + bias → 1 merged source).
    private Tensor? _mSourceW, _mSourceB;
    // noise_convs[i]: plain Conv1d(1, c_cur, k, stride, pad); noise_res[i]: AdaIN/Snake resblock.
    private readonly Tensor?[] _noiseConvW = new Tensor[4];
    private readonly Tensor?[] _noiseConvB = new Tensor[4];
    private readonly AdaSnakeResLoader[] _noiseRes = new AdaSnakeResLoader[4];
    // ups[i]: weight-norm ConvTranspose1d; resblocks: 4×3 AdaIN/Snake; alphas: 5 Snake params; conv_post.
    private readonly Tensor?[] _upsW = new Tensor[4];
    private readonly Tensor?[] _upsB = new Tensor[4];
    private readonly AdaSnakeResLoader[] _resblocks = new AdaSnakeResLoader[12];
    private readonly Tensor?[] _alphas = new Tensor[5];
    private Tensor? _convPostW, _convPostB;

    public StyleHifiGanGenerator(int sampleRate = 24_000)
    {
        _sampleRate = sampleRate;
        for (int i = 0; i < 4; i++)
        {
            int cCur = InitialChannel >> (i + 1);                 // 256, 128, 64, 32
            _noiseRes[i] = new AdaSnakeResLoader(cCur, kernel: i < 3 ? 7 : 11, dilations: [1, 3, 5]);
            for (int j = 0; j < 3; j++)
                _resblocks[i * 3 + j] = new AdaSnakeResLoader(cCur, kernel: ResblockKernels[j], dilations: [1, 3, 5]);
        }
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "decoder.generator")
    {
        _mSourceW = WhisperOps.EnsureF32(w[$"{prefix}.m_source.l_linear.weight"]);
        _mSourceB = WhisperOps.EnsureF32(w[$"{prefix}.m_source.l_linear.bias"]);
        for (int i = 0; i < 4; i++)
        {
            _noiseConvW[i] = WhisperOps.EnsureF32(w[$"{prefix}.noise_convs.{i}.weight"]);
            _noiseConvB[i] = WhisperOps.EnsureF32(w[$"{prefix}.noise_convs.{i}.bias"]);
            _noiseRes[i].LoadWeights(w, $"{prefix}.noise_res.{i}");
            _upsW[i] = WeightNorm.Compose(w, $"{prefix}.ups.{i}");
            _upsB[i] = WhisperOps.EnsureF32(w[$"{prefix}.ups.{i}.bias"]);
        }
        for (int i = 0; i < 12; i++) _resblocks[i].LoadWeights(w, $"{prefix}.resblocks.{i}");
        for (int i = 0; i < 5; i++) _alphas[i] = WhisperOps.EnsureF32(w[$"{prefix}.alphas.{i}"]);
        _convPostW = WeightNorm.Compose(w, $"{prefix}.conv_post");
        _convPostB = WhisperOps.EnsureF32(w[$"{prefix}.conv_post.bias"]);
    }

    /// <summary>Generates the raw waveform. <paramref name="x0"/> = <c>[1, 512, T]</c>, <paramref name="s"/> =
    /// <c>[1, 128]</c> decoder style, <paramref name="f0"/> = <c>[1, T]</c> frame-level pitch.</summary>
    /// <param name="harOverride">Test hook: a precomputed harmonic source (bypasses <see cref="StyleSineGen"/>)
    /// to isolate the generator body in the parity harness.</param>
    public float[] Forward(IBackend backend, Tensor x0, Tensor s, Tensor f0, float[]? harOverride = null)
    {
        int upProd = 1; foreach (int u in UpsampleRates) upProd *= u;   // 300
        float[] harArr = harOverride ?? StyleSineGen.Source(f0, upProd, _sampleRate, _mSourceW!, _mSourceB!);
        int tAudio = harArr.Length;
        Tensor har = new(new TensorShape(1, 1, tAudio), DType.F32);
        fixed (float* hp = harArr) Buffer.MemoryCopy(hp, (void*)har.DataPointer, (long)tAudio * 4, (long)tAudio * 4);

        Tensor x = new(x0.Shape, DType.F32);
        Buffer.MemoryCopy((void*)x0.DataPointer, (void*)x.DataPointer, x0.ElementCount * 4, x0.ElementCount * 4);

        for (int i = 0; i < 4; i++)
        {
            backend.Snake(x, x, _alphas[i]!, null);                       // x + (1/α)·sin(αx)²

            Tensor xSrc = NoiseConv(backend, har, i);                     // [1, c_cur, T_stage]
            Tensor xSrcRes = _noiseRes[i].Forward(backend, xSrc, s);
            xSrc.Dispose();

            Tensor xUp = UpsampleConvT(backend, x, i);
            x.Dispose(); x = xUp;
            NsfVocoderDsp.AddInPlaceCropped(x, xSrcRes);
            xSrcRes.Dispose();

            Tensor acc = _resblocks[i * 3].Forward(backend, x, s);        // MRF = mean of 3 kernels
            for (int j = 1; j < 3; j++)
            {
                Tensor rb = _resblocks[i * 3 + j].Forward(backend, x, s);
                NsfVocoderDsp.AddInPlaceCropped(acc, rb);
                rb.Dispose();
            }
            x.Dispose();
            NsfVocoderDsp.ScaleInPlace(acc, 1f / 3f);
            x = acc;
        }
        backend.Snake(x, x, _alphas[4]!, null);

        // conv_post: Conv1d(32 → 1, k7, pad3) → tanh.
        int inLen = (int)x.Shape[2];
        int k = (int)_convPostW!.Shape[2];
        Tensor audioT = new(new TensorShape(1, 1, inLen + 2 * 3 - (k - 1) - 1 + 1), DType.F32);
        backend.Conv1d(audioT, x, _convPostW!, _convPostB, stride: 1, padLeft: 3, padRight: 3, dilation: 1, groups: 1);
        x.Dispose(); har.Dispose();

        int n = (int)audioT.Shape[2];
        float[] audio = new float[n];
        float* ap = (float*)audioT.DataPointer;
        for (int t = 0; t < n; t++) audio[t] = MathF.Tanh(ap[t]);
        audioT.Dispose();
        return audio;
    }

    // noise_convs[i]: Conv1d(1, c_cur, kernel=2·stride, stride, pad=(stride+1)/2) for i<3; Conv1d(1, c_cur, 1) for i=3.
    private Tensor NoiseConv(IBackend backend, Tensor har, int i)
    {
        Tensor cw = _noiseConvW[i]!;
        int outCh = (int)cw.Shape[0];
        int kernel = (int)cw.Shape[2];
        int strideF0 = 1;
        for (int j = i + 1; j < UpsampleRates.Length; j++) strideF0 *= UpsampleRates[j];   // prod(rates[i+1:])
        int stride = i < 3 ? strideF0 : 1;
        int pad = i < 3 ? (strideF0 + 1) / 2 : 0;
        int inLen = (int)har.Shape[2];
        int outLen = (inLen + 2 * pad - (kernel - 1) - 1) / stride + 1;
        Tensor outT = new(new TensorShape(1, outCh, outLen), DType.F32);
        backend.Conv1d(outT, har, cw, _noiseConvB[i], stride: stride, padLeft: pad, padRight: pad, dilation: 1, groups: 1);
        return outT;
    }

    // ups[i]: weight-norm ConvTranspose1d, padLeft = u//2+u%2, padRight = u//2 (== torch padding/output_padding).
    private Tensor UpsampleConvT(IBackend backend, Tensor x, int i)
    {
        Tensor w = _upsW[i]!;
        int outCh = (int)w.Shape[1];
        int kernel = (int)w.Shape[2];
        int stride = UpsampleRates[i];
        int padLeft = stride / 2 + stride % 2;
        int padRight = stride / 2;
        int inLen = (int)x.Shape[2];
        int outLen = (inLen - 1) * stride - (padLeft + padRight) + (kernel - 1) + 1;
        Tensor outT = new(new TensorShape(1, outCh, outLen), DType.F32);
        backend.ConvTranspose1d(outT, x, w, _upsB[i], stride: stride, padLeft: padLeft, padRight: padRight, dilation: 1, groups: 1);
        return outT;
    }
}
