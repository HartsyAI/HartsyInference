using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Zonos;

/// <summary>Builds the full Zonos prefix-conditioner sequence: the 7 conditioners are concatenated along the
/// sequence (token) axis, then projected by <c>Linear(2048→2048)</c> + <c>LayerNorm</c>. The conditioners are:
/// <list type="number">
///   <item>espeak phonemes → an <c>Embedding(vocab → 2048)</c>, one token per phoneme</item>
///   <item>speaker → a passthrough <c>Linear(128 → 2048)</c>, one token</item>
///   <item>emotion (8-dim) → <see cref="ZonosFourierConditioner"/>, one token</item>
///   <item>fmax → Fourier (min 0 / max 24000), one token</item>
///   <item>pitch_std → Fourier (min 0 / max 400), one token</item>
///   <item>speaking_rate → Fourier (min 0 / max 40), one token</item>
///   <item>language_id → an integer <c>Embedding(105 → 2048)</c>, one token</item>
/// </list>
///
/// <para>Output is channels-first <c>[1, 2048, P]</c> where <c>P = phonemeCount + 6</c> — the layout the Zonos
/// backbone consumes after the conditioner. The IntegerConditioner is built inline (a plain embedding lookup
/// with an optional learned uncond vector). The Fourier conditioners reuse
/// <see cref="ZonosFourierConditioner"/>.</para></summary>
public sealed unsafe class ZonosConditioning : IDisposable
{
    public const int DModel = 2048;
    public const int SpeakerDim = 128;
    public const int EmotionDim = 8;

    /// <summary>IntegerConditioner language range is [-1, 126]; the embedding index is <c>languageId - (-1)</c>.</summary>
    public const int LanguageMinVal = -1;

    private Tensor? _phonemeEmbed;     // [phonemeVocab, 2048]
    private Tensor? _speakerW, _speakerB; // Linear(128 → 2048)
    private Tensor? _speakerUncond;    // conditioners.1.uncond_vector [2048]
    private Tensor? _languageEmbed;    // [numLanguages, 2048]
    private Tensor? _languageUncond;   // conditioners.6.uncond_vector [2048]
    private Tensor? _projW, _projB;    // Linear(2048 → 2048)
    private Tensor? _normW, _normB;    // LayerNorm(2048)

    private readonly ZonosFourierConditioner _emotion;
    private readonly ZonosFourierConditioner _fmax;
    private readonly ZonosFourierConditioner _pitchStd;
    private readonly ZonosFourierConditioner _speakingRate;
    private int _disposed;

    public ZonosConditioning()
    {
        _emotion = new ZonosFourierConditioner(EmotionDim, DModel, 0f, 1f);
        _fmax = new ZonosFourierConditioner(1, DModel, 0f, 24_000f);
        _pitchStd = new ZonosFourierConditioner(1, DModel, 0f, 400f);
        _speakingRate = new ZonosFourierConditioner(1, DModel, 0f, 40f);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "prefix_conditioner")
    {
        _phonemeEmbed = WhisperOps.EnsureF32(w[$"{prefix}.conditioners.0.phoneme_embedder.weight"]);
        _speakerW = WhisperOps.EnsureF32(w[$"{prefix}.conditioners.1.project.weight"]);
        _speakerB = TryGet(w, $"{prefix}.conditioners.1.project.bias");
        _speakerUncond = TryGet(w, $"{prefix}.conditioners.1.uncond_vector");
        _emotion.LoadWeights(w, $"{prefix}.conditioners.2");
        _fmax.LoadWeights(w, $"{prefix}.conditioners.3");
        _pitchStd.LoadWeights(w, $"{prefix}.conditioners.4");
        _speakingRate.LoadWeights(w, $"{prefix}.conditioners.5");
        _languageEmbed = WhisperOps.EnsureF32(w[$"{prefix}.conditioners.6.int_embedder.weight"]);
        _languageUncond = TryGet(w, $"{prefix}.conditioners.6.uncond_vector");
        _projW = WhisperOps.EnsureF32(w[$"{prefix}.project.weight"]);
        _projB = TryGet(w, $"{prefix}.project.bias");
        _normW = WhisperOps.EnsureF32(w[$"{prefix}.norm.weight"]);
        _normB = WhisperOps.EnsureF32(w[$"{prefix}.norm.bias"]);
    }

    /// <summary>Assembles the conditioning prefix and returns it channels-first <c>[1, 2048, P]</c>. When
    /// <paramref name="conditional"/> is <c>false</c> this builds the CFG unconditional branch: the espeak
    /// phonemes are kept (espeak is the only required conditioner) while speaker / emotion / fmax / pitch_std /
    /// speaking_rate / language_id are each replaced by their learned <c>uncond_vector</c>.</summary>
    public Tensor BuildPrefix(IBackend backend, ReadOnlySpan<int> phonemes, Tensor speaker128, float[] emotion8,
        float fmax, float pitchStd, float speakingRate, int languageId, bool conditional = true)
    {
        if (_phonemeEmbed is null) throw new InvalidOperationException("ZonosConditioning weights not loaded.");
        if (emotion8.Length != EmotionDim) throw new ArgumentException($"emotion must be {EmotionDim}-dim.", nameof(emotion8));

        int p = phonemes.Length + 6;
        // Build the sequence channels-last [1, P, 2048] first, then project + norm, then transpose to CF.
        Tensor seq = new(new TensorShape(1, p, DModel), DType.F32);
        float* sp = (float*)seq.DataPointer;

        // 1. Phonemes (kept in both branches — espeak is the only required conditioner).
        float* pe = (float*)_phonemeEmbed!.DataPointer;
        for (int i = 0; i < phonemes.Length; i++)
        {
            long src = (long)phonemes[i] * DModel;
            for (int d = 0; d < DModel; d++) sp[(long)i * DModel + d] = pe[src + d];
        }

        if (conditional)
        {
            // 2. Speaker passthrough Linear(128 → 2048).
            Tensor spk = speaker128.Reshape(new TensorShape(1, 1, SpeakerDim));
            Tensor spkProj = WhisperOps.ProjectLinear(backend, spk, _speakerW!, _speakerB, 1, 1, SpeakerDim, DModel);
            CopyTokenInto(sp, (float*)spkProj.DataPointer, phonemes.Length);
            spkProj.Dispose();

            // 3-6. Fourier conditioners.
            CopyArrayInto(sp, _emotion.Forward(emotion8), phonemes.Length + 1);
            CopyArrayInto(sp, _fmax.Forward([fmax]), phonemes.Length + 2);
            CopyArrayInto(sp, _pitchStd.Forward([pitchStd]), phonemes.Length + 3);
            CopyArrayInto(sp, _speakingRate.Forward([speakingRate]), phonemes.Length + 4);

            // 7. Language integer embedding (IntegerConditioner index = languageId - min_val, min_val = -1).
            float* le = (float*)_languageEmbed!.DataPointer;
            long lsrc = (long)(languageId - LanguageMinVal) * DModel;
            for (int d = 0; d < DModel; d++) sp[(long)(phonemes.Length + 5) * DModel + d] = le[lsrc + d];
        }
        else
        {
            // Unconditional branch: each non-espeak conditioner emits its learned uncond_vector.
            CopyArrayInto(sp, Uncond(_speakerUncond), phonemes.Length);
            CopyArrayInto(sp, _emotion.Uncond(), phonemes.Length + 1);
            CopyArrayInto(sp, _fmax.Uncond(), phonemes.Length + 2);
            CopyArrayInto(sp, _pitchStd.Uncond(), phonemes.Length + 3);
            CopyArrayInto(sp, _speakingRate.Uncond(), phonemes.Length + 4);
            CopyArrayInto(sp, Uncond(_languageUncond), phonemes.Length + 5);
        }

        // Project + LayerNorm.
        Tensor proj = WhisperOps.ProjectLinear(backend, seq, _projW!, _projB, 1, p, DModel, DModel);
        seq.Dispose();
        Tensor normed = new(proj.Shape, DType.F32);
        backend.LayerNorm(normed, proj, _normW!, _normB!, 1e-5f);
        proj.Dispose();

        // Channels-last [1, P, 2048] → channels-first [1, 2048, P].
        Tensor cf = new(new TensorShape(1, DModel, p), DType.F32);
        backend.Transpose2D(cf, normed, p, DModel);
        normed.Dispose();
        return cf;
    }

    /// <summary>Reads a <c>[2048]</c> uncond_vector into a fresh array (zeros if absent).</summary>
    private static float[] Uncond(Tensor? v)
    {
        float[] outArr = new float[DModel];
        if (v is not null) new Span<float>((void*)v.DataPointer, DModel).CopyTo(outArr);
        return outArr;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] top = [_phonemeEmbed, _speakerW, _speakerB, _speakerUncond, _languageEmbed, _languageUncond,
            _projW, _projB, _normW, _normB];
        foreach (Tensor? x in top) if (x is not null) yield return x;
        foreach (Tensor x in _emotion.EnumerateWeights()) yield return x;
        foreach (Tensor x in _fmax.EnumerateWeights()) yield return x;
        foreach (Tensor x in _pitchStd.EnumerateWeights()) yield return x;
        foreach (Tensor x in _speakingRate.EnumerateWeights()) yield return x;
    }

    private static void CopyTokenInto(float* dst, float* tokenSrc, int tokenIndex)
    {
        for (int d = 0; d < DModel; d++) dst[(long)tokenIndex * DModel + d] = tokenSrc[d];
    }

    private static void CopyArrayInto(float* dst, float[] tokenSrc, int tokenIndex)
    {
        for (int d = 0; d < DModel; d++) dst[(long)tokenIndex * DModel + d] = tokenSrc[d];
    }

    private static Tensor? TryGet(IReadOnlyDictionary<string, Tensor> w, string key)
        => w.TryGetValue(key, out Tensor? t) ? WhisperOps.EnsureF32(t) : null;

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }
}
