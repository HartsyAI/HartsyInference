using HartsyInference.Audio.Models.Kokoro;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Models.StyleTts2;
using HartsyInference.Audio.Preprocessing;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.PyTorch;

namespace HartsyInference.Audio.Pipelines;

/// <summary>StyleTTS 2 text-to-speech pipeline. Reuses Kokoro's (bit-identical) PLBERT + TextEncoder +
/// ProsodyPredictor + iSTFTNet decoder via <see cref="KokoroPipeline.SynthesizeFromStyle"/>, and adds
/// the StyleTTS 2-specific style path: the 256-d style vector comes from a reference-audio
/// <see cref="StyleEncoder"/> pair (zero-shot cloning) and/or the <see cref="StyleDiffusionSampler"/>
/// (random / perturbed style), instead of a fixed Kokoro voice pack. Supports the three StyleTTS 2
/// inference modes:
/// <list type="bullet">
///   <item><b>Clone</b> — <see cref="SynthesizeClone"/>: reference mel → acoustic ⊕ prosodic style.</item>
///   <item><b>Random</b> — <see cref="SynthesizeRandom"/>: diffusion sampler with no reference.</item>
///   <item><b>Perturbed clone</b> — <see cref="SynthesizeClonePerturbed"/>: diffusion sampler seeded by
///         the reference style for a more text-adapted result.</item>
/// </list>
/// Phoneme-in (IPA), like Kokoro — text→IPA G2P is the caller's responsibility.</summary>
public sealed unsafe class StyleTts2Pipeline : IDisposable
{
    private readonly StyleTts2Config _cfg;
    private readonly KokoroPipeline _kokoro;
    private readonly KokoroPlBert _plBert;
    private readonly KokoroPhonemeTokenizer _tokenizer;
    private readonly StyleEncoder? _styleEncoder;     // acoustic half (multispeaker only)
    private readonly StyleEncoder? _predictorEncoder; // prosodic half (multispeaker only)
    private readonly StyleDenoiser _denoiser;
    private readonly StyleDiffusionSampler _sampler;
    private IDisposable[] _retain = [];   // weight loader held for the pipeline's lifetime (tensors borrow it)
    private int _disposed;

    public StyleTts2Pipeline(StyleTts2Config cfg, KokoroPipeline kokoro, KokoroPlBert plBert,
        KokoroPhonemeTokenizer tokenizer, StyleDenoiser denoiser,
        StyleEncoder? styleEncoder = null, StyleEncoder? predictorEncoder = null)
    {
        _cfg = cfg;
        _kokoro = kokoro;
        _plBert = plBert;
        _tokenizer = tokenizer;
        _denoiser = denoiser;
        _styleEncoder = styleEncoder;
        _predictorEncoder = predictorEncoder;
        _sampler = new StyleDiffusionSampler(cfg);
    }

    /// <summary>Builds the full StyleTTS2-LibriTTS pipeline from the published checkpoint
    /// (<c>yl4579/StyleTTS2-LibriTTS/epochs_2nd_00020.pth</c>) + a StyleTTS2 vocab <c>tokenizer_config.json</c>
    /// (<c>{"vocab": {symbol: id}}</c> for the 178-symbol set). The <c>bert/text_encoder/predictor/decoder</c>
    /// reuse Kokoro's submodule loaders (LibriTTS dims are Kokoro-compatible); the two StarGAN-v2
    /// <see cref="StyleEncoder"/>s and the diffusion <see cref="StyleDenoiser"/> load their StyleTTS2-specific
    /// keys via <see cref="StyleTts2Weights.Adapt"/>.</summary>
    public static StyleTts2Pipeline LoadFromCheckpoint(string pthPath, string? tokenizerConfigPath = null)
    {
        StyleTts2Config cfg = new();
        KokoroConfig bb = cfg.Backbone;

        PytorchPickleLoader loader = new();
        loader.Load(pthPath, recursiveFlatten: true);
        Dictionary<string, Tensor> w = StyleTts2Weights.Adapt(loader.GetAllTensors());

        // Tokenizer: the embedded 178-symbol StyleTTS2 set (self-contained) unless a config path is supplied.
        KokoroPhonemeTokenizer tok = tokenizerConfigPath is null
            ? KokoroPhonemeTokenizer.FromVocab(StyleTts2Symbols.BuildVocab())
            : KokoroPhonemeTokenizer.LoadFromConfig(tokenizerConfigPath);

        KokoroPlBert plBert = new(bb); plBert.LoadWeights(w);
        KokoroTextEncoder textEnc = new(bb); textEnc.LoadWeights(w);
        KokoroProsodyPredictor pred = new(bb); pred.LoadWeights(w);
        KokoroIStftNetDecoder dec = new(bb, useHifiGan: true); dec.LoadWeights(w);
        StyleEncoder styleEnc = new(bb.StyleDim); styleEnc.LoadWeights(w, "style_encoder");
        StyleEncoder predEnc = new(bb.StyleDim); predEnc.LoadWeights(w, "predictor_encoder");
        StyleDenoiser denoiser = new(cfg); denoiser.LoadWeights(w, "diffusion");

        KokoroPipeline kokoro = new(bb, tok, plBert, textEnc, pred, dec);
        StyleTts2Pipeline p = new(cfg, kokoro, plBert, tok, denoiser, styleEnc, predEnc)
        {
            _retain = [loader]
        };
        return p;
    }

    /// <summary>StyleTTS2 reference-mel front-end, matching <c>meldataset.preprocess</c> /
    /// <c>compute_style</c>: reference audio resampled to 24 kHz, then
    /// <c>torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)</c> →
    /// <c>log(mel + 1e-5)</c> → <c>(·+4)/4</c>. <b>Critical quirk:</b> that <c>to_mel</c> passes NO
    /// <c>sample_rate</c>/<c>f_max</c>, so torchaudio builds the mel filterbank with its DEFAULTS
    /// (<c>sample_rate=16000</c>, <c>f_max=8000</c>) even though the audio is 24 kHz — the StyleEncoder was
    /// trained on exactly that filterbank. So the STFT runs on 24 kHz samples (n_fft/win/hop are in samples,
    /// rate-independent) but the filterbank is built for 16 kHz / 8 kHz. Using a 24 kHz / 12 kHz filterbank
    /// instead yields only corr ≈ 0.93 vs the real mel — enough to stay intelligible but it warps the timbre
    /// (wrong voice identity, "tinny" quality), so match the 16 kHz default here.</summary>
    public static Tensor ComputeReferenceMel(ReadOnlySpan<float> pcm, int sampleRate)
    {
        float[] mono24 = sampleRate == 24_000 ? pcm.ToArray() : Resampler.Create(sampleRate, 24_000).Resample(pcm.ToArray());
        // Natural-log power mel with a 1e-5 floor (≡ StyleTTS2's log(mel+1e-5) to <1 ULP for any real mel), then
        // apply the (·+4)/4 shift. SampleRate/Fmax below drive ONLY the mel filterbank (see MelFilterbank.Get);
        // 16000/8000 replicates torchaudio's default filterbank that meldataset.to_mel relies on.
        MelSpectrogramExtractor.Config cfg = new(
            SampleRate: 16_000, NFft: 2048, WinLength: 1200, HopLength: 300, NMels: 80,
            Fmin: 0.0, Fmax: 8_000.0, Norm: MelSpectrogramExtractor.Normalization.None, DropLastStftFrame: false,
            LogBase: MelSpectrogramExtractor.LogBase.Natural, LogFloor: 1e-5f, DynamicRangeDb: 0f,
            NormOffset: 0f, NormScale: 1f, PowerSpectrum: true,
            Scale: MelScale.Htk, SlaneyNorm: false, Center: true, CenterWindowInFft: true);
        MelSpectrogramExtractor ext = new(cfg);
        float[,] logMel = ext.Compute(mono24);                    // [80, frames] = log(max(mel, 1e-5))
        int frames = logMel.GetLength(1);
        Tensor mel = new(new TensorShape(1, 80, frames), DType.F32);
        float* mp = (float*)mel.DataPointer;
        for (int m = 0; m < 80; m++)
            for (int t = 0; t < frames; t++) mp[m * frames + t] = (logMel[m, t] + 4f) / 4f;
        return mel;
    }

    /// <summary>Clone from a raw reference waveform (computes the StyleTTS2 mel, then <see cref="SynthesizeClone"/>).</summary>
    public float[] SynthesizeCloneFromAudio(IBackend backend, string phonemes, ReadOnlySpan<float> referencePcm, int sampleRate, float speed = 1f)
    {
        using Tensor mel = ComputeReferenceMel(referencePcm, sampleRate);
        return SynthesizeClone(backend, phonemes, mel, speed);
    }

    /// <summary>Zero-shot voice cloning: extract the 256-d style directly from a reference mel
    /// <c>[1, 80, T]</c> (acoustic ⊕ prosodic) and synthesize <paramref name="phonemes"/> in that voice.</summary>
    public float[] SynthesizeClone(IBackend backend, string phonemes, Tensor referenceMel, float speed = 1f)
    {
        ThrowIfDisposed();
        using Tensor refStyle = CloneStyle(backend, referenceMel);
        return _kokoro.SynthesizeFromStyle(backend, phonemes, refStyle, speed);
    }

    /// <summary>Random / unconditional style sampling: the diffusion sampler invents a 256-d style
    /// conditioned only on the text (the only mode applicable to the single-speaker LJSpeech model).</summary>
    public float[] SynthesizeRandom(IBackend backend, string phonemes, int seed = 0, float speed = 1f)
    {
        ThrowIfDisposed();
        using Tensor refStyle = SampleStyle(backend, phonemes, referenceStyle: null, seed);
        return _kokoro.SynthesizeFromStyle(backend, phonemes, refStyle, speed);
    }

    /// <summary>Perturbed clone: the diffusion sampler refines the reference-extracted style toward the
    /// requested text — often more natural than the raw clone.</summary>
    public float[] SynthesizeClonePerturbed(IBackend backend, string phonemes, Tensor referenceMel, int seed = 0, float speed = 1f)
    {
        ThrowIfDisposed();
        using Tensor refClone = CloneStyle(backend, referenceMel);
        using Tensor refStyle = SampleStyle(backend, phonemes, referenceStyle: refClone, seed);
        return _kokoro.SynthesizeFromStyle(backend, phonemes, refStyle, speed);
    }

    /// <summary>Extracts the 256-d reference style = <c>style_encoder(mel) ⊕ predictor_encoder(mel)</c>.</summary>
    private Tensor CloneStyle(IBackend backend, Tensor referenceMel)
    {
        if (_styleEncoder is null || _predictorEncoder is null)
            throw new InvalidOperationException("Cloning requires the multi-speaker style + predictor encoders (LibriTTS).");
        using Tensor sAcoustic = _styleEncoder.Forward(backend, referenceMel);   // [1, 128]
        using Tensor sProsodic = _predictorEncoder.Forward(backend, referenceMel);
        return Concat(sAcoustic, sProsodic);                                     // [1, 256]
    }

    /// <summary>Runs the diffusion style sampler conditioned on PLBERT text features (and optional
    /// reference style), returning a squeezed <c>[1, 256]</c> style vector.</summary>
    private Tensor SampleStyle(IBackend backend, string phonemes, Tensor? referenceStyle, int seed)
    {
        int[] tokenIds = _tokenizer.Encode(phonemes);
        using Tensor textFeatures = _plBert.Forward(backend, tokenIds);
        _denoiser.Prepare(textFeatures, referenceStyle);
        using Tensor latent = _sampler.Sample(backend, _denoiser, seed);          // [1, 1, 256]
        return latent.Reshape(new TensorShape(1, _cfg.StyleDim));
    }

    private static Tensor Concat(Tensor a, Tensor b)
    {
        int da = (int)a.Shape[1], db = (int)b.Shape[1];
        Tensor outT = new(new TensorShape(1, da + db), DType.F32);
        Buffer.MemoryCopy((void*)a.DataPointer, (void*)outT.DataPointer, da * 4, da * 4);
        Buffer.MemoryCopy((void*)b.DataPointer, (void*)((float*)outT.DataPointer + da), db * 4, db * 4);
        return outT;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _kokoro.Dispose();
        _styleEncoder?.Dispose();
        _predictorEncoder?.Dispose();
        foreach (IDisposable d in _retain) d.Dispose();
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(StyleTts2Pipeline));
    }
}
