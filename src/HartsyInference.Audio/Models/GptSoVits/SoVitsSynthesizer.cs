using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Models.Vits;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.GptSoVits;

/// <summary>GPT-SoVITS stage 2 (SoVITS) — turns semantic tokens into 32 kHz audio. The semantic codebook
/// dequantizes each token to a 768-d latent (upsampled ×2 → 50 Hz), the real <see cref="SoVitsEncP"/> prior
/// encoder (<c>ssl_proj</c> → <c>encoder_ssl</c> → MRTE cross-attention → <c>encoder2</c> → <c>proj</c>)
/// yields <c>(m_p, logs_p)</c>, and the **reused g-conditioned** <see cref="VitsFlow"/> + <see cref="VitsHiFiGan"/>
/// decode under the reference speaker embedding <c>ge</c>.
///
/// <para><c>ge</c> is either caller-supplied (precomputed) or produced by <see cref="SoVitsRefEnc"/> from a
/// reference spec.</para></summary>
public sealed unsafe class SoVitsSynthesizer : IDisposable
{
    private readonly VitsConfig _core;
    private readonly int _sslDim, _inter;
    private readonly SoVitsEncP _encP;
    private readonly VitsFlow _flow;
    private readonly VitsHiFiGan _dec;
    private Tensor? _codebook;        // [1024, 768] semantic VQ
    private int _disposed;

    public SoVitsSynthesizer(VitsConfig core, int sslDim = 768, int sslLayers = 3, int textLayers = 6,
        int enc2Layers = 3, int mrteHidden = 512, int mrteHeads = 4)
    {
        _core = core; _sslDim = sslDim; _inter = core.InterChannels;
        _encP = new SoVitsEncP(core, sslDim, sslLayers, textLayers, enc2Layers, mrteHidden, mrteHeads);
        _flow = new VitsFlow(core);
        _dec = new VitsHiFiGan(core);
    }

    public int SampleRate => _core.SampleRate;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _codebook = WhisperOps.EnsureF32(w["quantizer.vq.layers.0._codebook.embed"]);
        _encP.LoadWeights(w);
        _flow.LoadWeights(w);
        _dec.LoadWeights(w);
    }

    /// <summary>Synthesizes audio from semantic token ids + the reference speaker embedding <c>ge [1,gin,1]</c>.</summary>
    public float[] Forward(IBackend backend, ReadOnlySpan<int> semanticTokens, Tensor ge, int seed = 0)
    {
        ThrowIfDisposed();
        int n = semanticTokens.Length, t = n * 2;   // ×2 nearest upsample (25 → 50 Hz)

        // Dequantize codes → [1, sslDim, T] (nearest ×2 along time).
        Tensor latent = new(new TensorShape(1, _sslDim, t), DType.F32);
        float* lp = (float*)latent.DataPointer;
        float* cb = (float*)_codebook!.DataPointer;
        for (int i = 0; i < n; i++)
            for (int c = 0; c < _sslDim; c++)
            {
                float v = cb[(long)semanticTokens[i] * _sslDim + c];
                lp[(long)c * t + 2 * i] = v; lp[(long)c * t + 2 * i + 1] = v;
            }

        // The ×2-upsampled semantic ids feed the MRTE text stream via enc_p.text_embedding.
        int[] textTokens = new int[t];
        for (int i = 0; i < n; i++) { textTokens[2 * i] = semanticTokens[i]; textTokens[2 * i + 1] = semanticTokens[i]; }

        (Tensor mP, Tensor logsP) = _encP.Forward(backend, latent, t, textTokens, ge); latent.Dispose();

        uint rng = DeterministicRng.Seed(seed);
        Tensor zP = new(new TensorShape(1, _inter, t), DType.F32);
        float* mp = (float*)mP.DataPointer;
        float* lsp = (float*)logsP.DataPointer;
        float* zp = (float*)zP.DataPointer;
        for (int c = 0; c < _inter; c++)
            for (int j = 0; j < t; j++)
                zp[(long)c * t + j] = mp[(long)c * t + j] + DeterministicRng.NextGaussian(ref rng) * MathF.Exp(lsp[(long)c * t + j]);
        mP.Dispose(); logsP.Dispose();

        Tensor z = _flow.Reverse(backend, zP, t, ge); zP.Dispose();
        float[] audio = _dec.Forward(backend, z, t, ge); z.Dispose();
        return audio;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_codebook is not null) yield return _codebook;
        foreach (Tensor t in _encP.EnumerateWeights()) yield return t;
        foreach (Tensor t in _flow.EnumerateWeights()) yield return t;
        foreach (Tensor t in _dec.EnumerateWeights()) yield return t;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(SoVitsSynthesizer));
    }
}
