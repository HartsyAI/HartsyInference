using HartsyInference.Audio.Models.GptSoVits;
using HartsyInference.Audio.Models.Hubert;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Pipelines;

/// <summary>GPT-SoVITS v2 zero-shot TTS pipeline: assembles the s1 (text→semantic AR) and s2 (semantic→audio)
/// stages around a reference clip. The reference waveform → <see cref="Hubert"/> SSL features →
/// <see cref="SoVitsSynthesizer.EncodeSsl"/> prompt semantic codes; the text phonemes + BERT features drive
/// <see cref="Text2Semantic"/> to autoregressively predict the target semantic tokens; finally
/// <see cref="SoVitsSynthesizer"/> decodes those (conditioned on the reference speaker embedding from
/// <see cref="SoVitsRefEnc"/>) to 32 kHz audio.
///
/// <para>Phonemes and BERT features are caller-supplied (produced by the language front-end / G2P): the s1
/// text span is <c>refPhonemes + targetPhonemes</c> with the matching concatenated BERT, and the s2 MRTE text
/// stream is the <c>targetPhonemes</c>. The reference spectrogram is the 1025-bin linear spec (the ref encoder
/// consumes its first 704 bins). Components are shared resources passed in by the caller (not owned here).</para></summary>
public sealed unsafe class GptSoVitsPipeline : IDisposable
{
    private const int RefEncBins = 704;     // v2 ref_enc consumes refer[:, :704]

    private readonly Hubert _hubert;
    private readonly Text2Semantic _s1;
    private readonly SoVitsSynthesizer _s2;
    private readonly SoVitsRefEnc _refEnc;
    private int _disposed;

    public GptSoVitsPipeline(Hubert hubert, Text2Semantic s1, SoVitsSynthesizer s2, SoVitsRefEnc refEnc)
    {
        _hubert = hubert ?? throw new ArgumentNullException(nameof(hubert));
        _s1 = s1 ?? throw new ArgumentNullException(nameof(s1));
        _s2 = s2 ?? throw new ArgumentNullException(nameof(s2));
        _refEnc = refEnc ?? throw new ArgumentNullException(nameof(refEnc));
    }

    public int SampleRate => _s2.SampleRate;

    /// <summary>Synthesizes 32 kHz audio for <paramref name="targetPhonemes"/> in the reference voice.
    /// <paramref name="refPcm16k"/> is the 16 kHz reference waveform <c>[1, samples]</c> (for HuBERT);
    /// <paramref name="refSpec"/> is its 1025-bin linear spectrogram <c>[1, 1025, tSpec]</c> (for the speaker
    /// embedding); <paramref name="allPhonemes"/> = ref+target phoneme ids and <paramref name="allBert"/> =
    /// <c>[BertDim, allPhonemes.Length]</c> drive s1.</summary>
    public float[] Generate(IBackend backend, Tensor refPcm16k, int samples, Tensor refSpec, int tSpec,
        ReadOnlySpan<int> allPhonemes, Tensor allBert, ReadOnlySpan<int> targetPhonemes,
        int maxNewTokens = 1500, int seed = 0, float noiseScale = 0.5f)
    {
        ThrowIfDisposed();

        // Reference SSL → prompt semantic codes.
        Tensor ssl = _hubert.Forward(backend, refPcm16k, samples);
        int[] promptSemantic = _s2.EncodeSsl(backend, ssl);
        ssl.Dispose();

        // s1: text + BERT + prompt semantic → predicted target semantic tokens.
        List<int> predicted = _s1.Generate(backend, allPhonemes, allBert, promptSemantic, maxNewTokens, seed);

        // Reference speaker embedding from the first 704 spec bins.
        Tensor refSpec704 = SliceBins(refSpec, RefEncBins, tSpec);
        Tensor ge = _refEnc.Forward(backend, refSpec704, tSpec);
        refSpec704.Dispose();

        // s2: semantic tokens + target phonemes (MRTE text stream) + ge → audio.
        float[] audio = _s2.Forward(backend, predicted.ToArray(), targetPhonemes, ge, seed, noiseScale);
        ge.Dispose();
        return audio;
    }

    /// <summary>Copies the first <paramref name="bins"/> channels of a <c>[1, C, t]</c> spectrogram.</summary>
    private static Tensor SliceBins(Tensor spec, int bins, int t)
    {
        Tensor outT = new(new TensorShape(1, bins, t), DType.F32);
        float* src = (float*)spec.DataPointer, dst = (float*)outT.DataPointer;
        long rowBytes = (long)t * sizeof(float);
        for (int c = 0; c < bins; c++)
            Buffer.MemoryCopy(src + (long)c * t, dst + (long)c * t, rowBytes, rowBytes);
        return outT;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(GptSoVitsPipeline));
    }
}
