using HartsyInference.Audio.Models.Vocoders;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Codecs.Vocos;

/// <summary>Standalone Vocos decoder (ConvNeXt backbone + iSTFT head) as used by YuE's per-stem 44.1 kHz vocoders
/// (<c>decoders/decoder_131000.pth</c> = vocal, <c>decoder_151000.pth</c> = instrumental). Consumes the x-codec
/// quantizer's 1024-dim latent (the same <c>[B, 1024, T]</c> that would otherwise feed the SeaNet/DAC decoder) and
/// reconstructs a waveform directly — this is YuE's REAL output path; the raw x-codec decode is only a 16 kHz draft.
///
/// <para>Architecture (config <c>decoders/config.yaml</c>): <c>backbone.embed</c> Conv1d(1024→512, k7, pad same),
/// <c>backbone.norm</c> LayerNorm, 8× ConvNeXt blocks (depthwise Conv1d k7 + LayerNorm + Linear 512→1536 + GELU +
/// Linear 1536→512 + per-channel γ), <c>backbone.final_layer_norm</c>, then <c>head.out</c> Linear(512→n_fft+2) →
/// (mag, phase) → iSTFT (n_fft 3528, hop 882 ⇒ 44.1 kHz at 50 fps). Same op sequence as
/// <see cref="WavTokenizer.WavTokenizerHead"/>; the shipped Hann window matches <see cref="HannWindow"/>.</para></summary>
public sealed unsafe class VocosDecoder
{
    private const int Dim = 512;
    private const int InputChannels = 1024;
    private const int IntermediateDim = 1536;
    private const int NumLayers = 8;
    private const int NFft = 3528;
    private const int HopLength = 882;
    /// <summary>44.1 kHz = HopLength · frame_rate (882 · 50).</summary>
    public const int SampleRate = 44_100;

    private readonly int _nBins = NFft / 2 + 1;

    private Tensor? _embedW, _embedB;         // backbone.embed: Conv1d(1024→512, k7)
    private Tensor? _normW, _normB;           // backbone.norm (post-embed LayerNorm)
    private readonly Tensor?[] _dwW = new Tensor?[NumLayers];
    private readonly Tensor?[] _dwB = new Tensor?[NumLayers];
    private readonly Tensor?[] _bnW = new Tensor?[NumLayers];
    private readonly Tensor?[] _bnB = new Tensor?[NumLayers];
    private readonly Tensor?[] _fc1W = new Tensor?[NumLayers];
    private readonly Tensor?[] _fc1B = new Tensor?[NumLayers];
    private readonly Tensor?[] _fc2W = new Tensor?[NumLayers];
    private readonly Tensor?[] _fc2B = new Tensor?[NumLayers];
    private readonly Tensor?[] _gamma = new Tensor?[NumLayers];
    private Tensor? _finalNormW, _finalNormB; // backbone.final_layer_norm
    private Tensor? _outW, _outB;             // head.out: Linear(512 → n_fft+2)

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _embedW = WhisperOps.EnsureF32(w["backbone.embed.weight"]);
        _embedB = WhisperOps.EnsureF32(w["backbone.embed.bias"]);
        _normW = WhisperOps.EnsureF32(w["backbone.norm.weight"]);
        _normB = WhisperOps.EnsureF32(w["backbone.norm.bias"]);
        for (int i = 0; i < NumLayers; i++)
        {
            string p = $"backbone.convnext.{i}";
            _dwW[i] = WhisperOps.EnsureF32(w[$"{p}.dwconv.weight"]);
            _dwB[i] = WhisperOps.EnsureF32(w[$"{p}.dwconv.bias"]);
            _bnW[i] = WhisperOps.EnsureF32(w[$"{p}.norm.weight"]);
            _bnB[i] = WhisperOps.EnsureF32(w[$"{p}.norm.bias"]);
            _fc1W[i] = WhisperOps.EnsureF32(w[$"{p}.pwconv1.weight"]);
            _fc1B[i] = WhisperOps.EnsureF32(w[$"{p}.pwconv1.bias"]);
            _fc2W[i] = WhisperOps.EnsureF32(w[$"{p}.pwconv2.weight"]);
            _fc2B[i] = WhisperOps.EnsureF32(w[$"{p}.pwconv2.bias"]);
            _gamma[i] = WhisperOps.EnsureF32(w[$"{p}.gamma"]);
        }
        _finalNormW = WhisperOps.EnsureF32(w["backbone.final_layer_norm.weight"]);
        _finalNormB = WhisperOps.EnsureF32(w["backbone.final_layer_norm.bias"]);
        _outW = WhisperOps.EnsureF32(w["head.out.weight"]);
        _outB = WhisperOps.EnsureF32(w["head.out.bias"]);
    }

    /// <summary>Decodes the quantizer latent <c>[B, 1024, T]</c> to PCM <c>[B, 1, T·hop]</c> at 44.1 kHz.</summary>
    public Tensor Forward(IBackend backend, Tensor latent, int batch, int tFrames)
    {
        if (_embedW is null) throw new InvalidOperationException("VocosDecoder weights not loaded.");

        // backbone.embed: Conv1d(1024→512, kernel 7, "same" padding = 3 each side).
        Tensor x = new(new TensorShape(batch, Dim, tFrames), DType.F32);
        backend.Conv1d(x, latent, _embedW!, _embedB, stride: 1, padLeft: 3, padRight: 3, dilation: 1, groups: 1);

        // backbone.norm: LayerNorm over channels (channels-last), then back to channels-first.
        {
            Tensor cl = new(new TensorShape(batch, tFrames, Dim), DType.F32);
            backend.Transpose2D(cl, x, Dim, tFrames);
            x.Dispose();
            Tensor n = new(cl.Shape, DType.F32);
            backend.LayerNorm(n, cl, _normW!, _normB!, 1e-6f);
            cl.Dispose();
            x = new(new TensorShape(batch, Dim, tFrames), DType.F32);
            backend.Transpose2D(x, n, tFrames, Dim);
            n.Dispose();
        }

        // 8× ConvNeXt blocks (depthwise Conv1d + LN + Linear + GELU + Linear + γ, residual).
        for (int i = 0; i < NumLayers; i++)
        {
            int pad = 3;
            int tDw = tFrames + 2 * pad - 6;   // k7 valid length after "same"-padded conv
            Tensor dw = new(new TensorShape(batch, Dim, tDw), DType.F32);
            backend.Conv1d(dw, x, _dwW[i]!, _dwB[i], stride: 1, padLeft: pad, padRight: pad, dilation: 1, groups: Dim);

            Tensor cl = new(new TensorShape(batch, tDw, Dim), DType.F32);
            backend.Transpose2D(cl, dw, Dim, tDw);
            dw.Dispose();

            Tensor normed = new(cl.Shape, DType.F32);
            backend.LayerNorm(normed, cl, _bnW[i]!, _bnB[i]!, 1e-6f);
            cl.Dispose();

            Tensor up = WhisperOps.ProjectLinear(backend, normed, _fc1W[i]!, _fc1B[i], batch, tDw, Dim, IntermediateDim);
            normed.Dispose();
            Tensor act = new(up.Shape, DType.F32);
            // Vocos uses nn.GELU() — the exact erf form, not the tanh approximation.
            backend.GeluErf(act, up);
            up.Dispose();
            Tensor down = WhisperOps.ProjectLinear(backend, act, _fc2W[i]!, _fc2B[i], batch, tDw, IntermediateDim, Dim);
            act.Dispose();

            // Per-channel layer-scale γ (channels-last rows).
            float* dp = (float*)down.DataPointer;
            float* gp = (float*)_gamma[i]!.DataPointer;
            for (int b = 0; b < batch; b++)
                for (int ti = 0; ti < tDw; ti++)
                {
                    int rowBase = (b * tDw + ti) * Dim;
                    for (int d = 0; d < Dim; d++) dp[rowBase + d] *= gp[d];
                }

            Tensor cf = new(new TensorShape(batch, Dim, tDw), DType.F32);
            backend.Transpose2D(cf, down, tDw, Dim);
            down.Dispose();

            Tensor result = new(x.Shape, DType.F32);
            if (tDw == tFrames)
            {
                backend.Add(result, x, cf);
            }
            else
            {
                int trim = (tDw - tFrames) / 2;
                float* sp = (float*)cf.DataPointer;
                float* xp = (float*)x.DataPointer;
                float* rp = (float*)result.DataPointer;
                for (int b = 0; b < batch; b++)
                    for (int c = 0; c < Dim; c++)
                    {
                        int srcBase = (b * Dim + c) * tDw + trim;
                        int baseB = (b * Dim + c) * tFrames;
                        for (int j = 0; j < tFrames; j++) rp[baseB + j] = xp[baseB + j] + sp[srcBase + j];
                    }
            }
            cf.Dispose();
            x.Dispose();
            x = result;
        }

        // backbone.final_layer_norm (channels-last).
        Tensor cl2 = new(new TensorShape(batch, tFrames, Dim), DType.F32);
        backend.Transpose2D(cl2, x, Dim, tFrames);
        x.Dispose();
        Tensor fn = new(cl2.Shape, DType.F32);
        backend.LayerNorm(fn, cl2, _finalNormW!, _finalNormB!, 1e-6f);
        cl2.Dispose();

        // head.out: Linear(512 → n_fft+2) → (mag, phase).
        Tensor magPhase = WhisperOps.ProjectLinear(backend, fn, _outW!, _outB, batch, tFrames, Dim, _nBins * 2);
        fn.Dispose();

        Tensor pcm = ApplyIStft(magPhase, batch, tFrames);
        magPhase.Dispose();
        return pcm;
    }

    /// <summary>Vocos ISTFTHead: <c>mag = exp(out[:n_bins])</c> (clipped), <c>phase = out[n_bins:]</c>, complex
    /// spectrogram → iSTFT overlap-add with the head's <c>padding="same"</c> (per the shipped config.yaml), which
    /// trims <c>(n_fft - hop)/2</c> per end and yields <c>frames · hop</c> samples.</summary>
    private Tensor ApplyIStft(Tensor magPhase, int batch, int tFrames)
    {
        // "same", NOT center: trimming n_fft/2 instead would start the waveform 441 samples late and drop 882 —
        // and since the x-codec band the post-process crosses this against is correctly timed, that offset combs.
        int edgePad = (NFft - HopLength) / 2;
        int outLen = tFrames * HopLength;
        Tensor pcm = new(new TensorShape(batch, 1, outLen), DType.F32);
        float* dp = (float*)pcm.DataPointer;
        float* mp = (float*)magPhase.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            float[] re = new float[tFrames * _nBins];
            float[] im = new float[tFrames * _nBins];
            for (int t = 0; t < tFrames; t++)
            {
                int srcBase = (b * tFrames + t) * (_nBins * 2);
                for (int k = 0; k < _nBins; k++)
                {
                    // Vocos clips the MAGNITUDE at 1e2, not the log-magnitude: torch.clip(torch.exp(x), max=1e2).
                    // Clipping before the exp lets mag reach e^100 and disables the safeguard entirely.
                    float mag = MathF.Min(MathF.Exp(mp[srcBase + k]), 100f);
                    float phase = mp[srcBase + _nBins + k];
                    re[t * _nBins + k] = mag * MathF.Cos(phase);
                    im[t * _nBins + k] = mag * MathF.Sin(phase);
                }
            }
            float[] wav = IStft.Apply(re, im, tFrames, NFft, HopLength, edgePad);
            int copy = Math.Min(wav.Length, outLen);
            for (int j = 0; j < copy; j++) dp[b * outLen + j] = wav[j];
            for (int j = copy; j < outLen; j++) dp[b * outLen + j] = 0f;
        }
        return pcm;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] singles = [_embedW, _embedB, _normW, _normB, _finalNormW, _finalNormB, _outW, _outB];
        foreach (Tensor? t in singles) if (t is not null) yield return t;
        for (int i = 0; i < NumLayers; i++)
        {
            Tensor?[] blk = [_dwW[i], _dwB[i], _bnW[i], _bnB[i], _fc1W[i], _fc1B[i], _fc2W[i], _fc2B[i], _gamma[i]];
            foreach (Tensor? t in blk) if (t is not null) yield return t;
        }
    }
}
