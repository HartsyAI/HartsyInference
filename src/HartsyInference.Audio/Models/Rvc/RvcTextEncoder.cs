using HartsyInference.Audio.Models.Vits;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Rvc;

/// <summary>RVC `enc_p` (`TextEncoder768`): content features (768-d) through <c>emb_phone</c> Linear + coarse
/// pitch through <c>emb_pitch</c> Embedding are summed, scaled by <c>√hidden</c>, LeakyReLU'd, then run through
/// the shared VITS encoder layers + projection to the prior <c>(m_p, logs_p)</c>. Reuses
/// <see cref="VitsTextEncoder"/> via its embedding-in entry point.</summary>
public sealed unsafe class RvcTextEncoder
{
    private readonly RvcConfig _cfg;
    private readonly int _h, _content;
    private readonly VitsTextEncoder _core;
    private Tensor? _embPhoneW, _embPhoneB, _embPitch;

    public RvcTextEncoder(RvcConfig cfg)
    {
        _cfg = cfg; _h = cfg.Core.HiddenChannels; _content = cfg.ContentDim;
        _core = new VitsTextEncoder(cfg.Core);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "enc_p")
    {
        _embPhoneW = WhisperOps.EnsureF32(w[$"{prefix}.emb_phone.weight"]);
        _embPhoneB = w.TryGetValue($"{prefix}.emb_phone.bias", out Tensor? b) ? WhisperOps.EnsureF32(b) : null;
        _embPitch = WhisperOps.EnsureF32(w[$"{prefix}.emb_pitch.weight"]);
        _core.LoadWeightsLayersOnly(w, prefix);
    }

    /// <summary>Encodes content features <c>[1, 768, T]</c> + coarse pitch bins <c>[T]</c> → (m_p, logs_p).</summary>
    public (Tensor Hidden, Tensor MP, Tensor LogsP) Forward(IBackend backend, Tensor content, ReadOnlySpan<int> pitch, int t)
    {
        // emb_phone: Linear(768→hidden) over [1, T, 768].
        Tensor contentCl = new(new TensorShape(1, t, _content), DType.F32);
        backend.Transpose2D(contentCl, content, _content, t);
        Tensor phone = WhisperOps.ProjectLinear(backend, contentCl, _embPhoneW!, _embPhoneB, 1, t, _content, _h);
        contentCl.Dispose();

        // + emb_pitch, × √hidden, LeakyReLU(0.1), → channels-first [1, hidden, T].
        float scale = MathF.Sqrt(_h);
        float* pp = (float*)phone.DataPointer;
        float* ep = (float*)_embPitch!.DataPointer;
        Tensor embed = new(new TensorShape(1, _h, t), DType.F32);
        float* xp = (float*)embed.DataPointer;
        for (int j = 0; j < t; j++)
            for (int c = 0; c < _h; c++)
            {
                float v = (pp[(long)j * _h + c] + ep[(long)pitch[j] * _h + c]) * scale;
                xp[(long)c * t + j] = v < 0 ? v * 0.1f : v;     // LeakyReLU(0.1)
            }
        phone.Dispose();
        return _core.ForwardFromEmbedding(backend, embed, t);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_embPhoneW is not null) yield return _embPhoneW;
        if (_embPhoneB is not null) yield return _embPhoneB;
        if (_embPitch is not null) yield return _embPitch;
        foreach (Tensor t in _core.EnumerateWeights()) yield return t;
    }
}
