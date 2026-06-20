using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Models.Vits;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.GptSoVits;

/// <summary>GPT-SoVITS stage 2 (SoVITS) — turns semantic tokens into 32 kHz audio. The semantic codebook
/// dequantizes each token to a 768-d latent (upsampled ×2 → 50 Hz), <c>ssl_proj</c> maps it to the prior
/// channels, a prior encoder yields <c>(m_p, logs_p)</c>, and the **reused g-conditioned**
/// <see cref="VitsFlow"/> + <see cref="VitsHiFiGan"/> decode under the reference speaker embedding <c>ge</c>.
///
/// <para>Structural: the bespoke <c>enc_p</c> refinement (dual attention encoders + MRTE cross-attention) and
/// the <c>ref_enc</c> MelStyleEncoder (which produces <c>ge</c>) are staged — <c>ge</c> is caller-supplied.</para></summary>
public sealed unsafe class SoVitsSynthesizer : IDisposable
{
    private readonly VitsConfig _core;
    private readonly int _sslDim, _inter;
    private readonly VitsFlow _flow;
    private readonly VitsHiFiGan _dec;
    private Tensor? _codebook;        // [1024, 768] semantic VQ
    private Tensor? _sslProjW, _sslProjB, _encW, _encB, _projW, _projB;
    private int _disposed;

    public SoVitsSynthesizer(VitsConfig core, int sslDim = 768)
    {
        _core = core; _sslDim = sslDim; _inter = core.InterChannels;
        _flow = new VitsFlow(core);
        _dec = new VitsHiFiGan(core);
    }

    public int SampleRate => _core.SampleRate;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _codebook = WhisperOps.EnsureF32(w["quantizer.vq.layers.0._codebook.embed"]);
        _sslProjW = VitsWeights.Conv(w, "enc_p.ssl_proj"); _sslProjB = VitsWeights.Bias(w, "enc_p.ssl_proj");
        _encW = VitsWeights.Conv(w, "enc_p.encoder_ssl_proj"); _encB = VitsWeights.Bias(w, "enc_p.encoder_ssl_proj");
        _projW = VitsWeights.Conv(w, "enc_p.proj"); _projB = VitsWeights.Bias(w, "enc_p.proj");
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

        Tensor proj = new(new TensorShape(1, _inter, t), DType.F32);
        backend.Conv1d(proj, latent, _sslProjW!, _sslProjB, 1, 0, 0, 1, 1); latent.Dispose();
        Tensor enc = new(new TensorShape(1, _inter, t), DType.F32);
        backend.Conv1d(enc, proj, _encW!, _encB, 1, 0, 0, 1, 1); proj.Dispose();
        Tensor stats = new(new TensorShape(1, 2 * _inter, t), DType.F32);
        backend.Conv1d(stats, enc, _projW!, _projB, 1, 0, 0, 1, 1); enc.Dispose();

        uint rng = DeterministicRng.Seed(seed);
        Tensor zP = new(new TensorShape(1, _inter, t), DType.F32);
        float* sp = (float*)stats.DataPointer;
        float* zp = (float*)zP.DataPointer;
        for (int c = 0; c < _inter; c++)
            for (int j = 0; j < t; j++)
            {
                float m = sp[(long)c * t + j], logs = sp[(long)(_inter + c) * t + j];
                zp[(long)c * t + j] = m + DeterministicRng.NextGaussian(ref rng) * MathF.Exp(logs);
            }
        stats.Dispose();

        Tensor z = _flow.Reverse(backend, zP, t, ge); zP.Dispose();
        float[] audio = _dec.Forward(backend, z, t, ge); z.Dispose();
        return audio;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] own = [_codebook, _sslProjW, _sslProjB, _encW, _encB, _projW, _projB];
        foreach (Tensor? t in own) if (t is not null) yield return t;
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
