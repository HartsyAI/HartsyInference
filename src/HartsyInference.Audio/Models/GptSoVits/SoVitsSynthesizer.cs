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
    private Tensor? _sslProjW, _sslProjB;     // top-level ssl_proj (768→768, k2 s2) for SSL→codes encode
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
        if (w.TryGetValue("ssl_proj.weight", out Tensor? sw))
        {
            _sslProjW = WhisperOps.EnsureF32(sw);
            _sslProjB = w.TryGetValue("ssl_proj.bias", out Tensor? sb) ? WhisperOps.EnsureF32(sb) : null;
        }
        _encP.LoadWeights(w);
        _flow.LoadWeights(w);
        _dec.LoadWeights(w);
    }

    /// <summary>Encodes HuBERT SSL features <c>[1, 768, T]</c> into prompt semantic codes (one per 2 frames):
    /// the top-level <c>ssl_proj</c> (Conv1d 768→768, kernel 2, stride 2) downsamples to 25 Hz, then each frame
    /// is assigned to its nearest codebook entry (the VQ <c>extract_latent</c> path).</summary>
    public int[] EncodeSsl(IBackend backend, Tensor ssl)
    {
        ThrowIfDisposed();
        if (_sslProjW is null) throw new InvalidOperationException("ssl_proj weight not loaded (SSL encode unavailable).");
        int hubT = (int)ssl.Shape[2];
        int t = (hubT - 2) / 2 + 1;     // Conv1d k2 s2, pad 0
        Tensor proj = new(new TensorShape(1, _sslDim, t), DType.F32);
        backend.Conv1d(proj, ssl, _sslProjW!, _sslProjB, 2, 0, 0, 1, 1);

        float* pp = (float*)proj.DataPointer;
        float* cb = (float*)_codebook!.DataPointer;
        int codes = (int)_codebook.Shape[0];
        int[] result = new int[t];
        for (int f = 0; f < t; f++)
        {
            int best = 0;
            float bestDist = float.PositiveInfinity;
            for (int e = 0; e < codes; e++)
            {
                float dist = 0f;
                long eoff = (long)e * _sslDim;
                for (int c = 0; c < _sslDim; c++)
                {
                    float d = pp[(long)c * t + f] - cb[eoff + c];
                    dist += d * d;
                }
                if (dist < bestDist) { bestDist = dist; best = e; }
            }
            result[f] = best;
        }
        proj.Dispose();
        return result;
    }

    /// <summary>Synthesizes audio from semantic token ids, the <b>target phonemes</b> (the MRTE text stream,
    /// vocab matching <c>enc_p.text_embedding</c>), and the reference speaker embedding <c>ge [1,gin,1]</c>.
    /// <paramref name="noiseScale"/> scales the prior sampling noise (GPT-SoVITS default 0.5; pass 0 for a
    /// deterministic <c>z_p = m_p</c>).</summary>
    public float[] Forward(IBackend backend, ReadOnlySpan<int> semanticTokens, ReadOnlySpan<int> targetPhonemes,
        Tensor ge, int seed = 0, float noiseScale = 0.5f)
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

        // MRTE text stream = the target phonemes (enc_p.text_embedding → encoder_text → cross-attention).
        (Tensor mP, Tensor logsP) = _encP.Forward(backend, latent, t, targetPhonemes, ge); latent.Dispose();

        uint rng = DeterministicRng.Seed(seed);
        Tensor zP = new(new TensorShape(1, _inter, t), DType.F32);
        float* mp = (float*)mP.DataPointer;
        float* lsp = (float*)logsP.DataPointer;
        float* zp = (float*)zP.DataPointer;
        for (int c = 0; c < _inter; c++)
            for (int j = 0; j < t; j++)
            {
                float noise = noiseScale != 0f ? DeterministicRng.NextGaussian(ref rng) * MathF.Exp(lsp[(long)c * t + j]) * noiseScale : 0f;
                zp[(long)c * t + j] = mp[(long)c * t + j] + noise;
            }
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
