using SharpInference.Audio.Models.Whisper;
using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Models.Music;

/// <summary>MusicGen causal decoder: K codebook embeddings (summed) + sinusoidal positions → a stack of
/// pre-norm blocks (causal self-attention + cross-attention to the T5 text states + GELU MLP) → K parallel
/// output heads. Reuses the WhisperOps attention helpers (`ProjectLinear`, multi-head reshape, SDPA);
/// runs full-sequence (no KV cache) — AR callers re-feed the prefix. Cross-attn states are the
/// caller-projected T5 features <c>[1, T_text, hidden]</c>.</summary>
public sealed unsafe class MusicGenDecoder : IDisposable
{
    private readonly MusicGenConfig _cfg;
    private readonly MusicGenBlock[] _blocks;
    private int _disposed;

    private Tensor?[] _codebookEmbed;   // K × [codebookSize+1, hidden]
    private Tensor? _encToDecW;         // textDim → hidden (project T5 states once)
    private Tensor? _lnOutG, _lnOutB;
    private Tensor?[] _heads;           // K × [codebookSize, hidden]

    public MusicGenConfig Config => _cfg;

    public MusicGenDecoder(MusicGenConfig cfg)
    {
        _cfg = cfg;
        _blocks = new MusicGenBlock[cfg.NumLayers];
        for (int i = 0; i < cfg.NumLayers; i++) _blocks[i] = new MusicGenBlock(cfg);
        _codebookEmbed = new Tensor?[cfg.NumCodebooks];
        _heads = new Tensor?[cfg.NumCodebooks];
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "model.decoder")
    {
        string p = prefix.Length == 0 ? "" : prefix + ".";
        for (int i = 0; i < _cfg.NumCodebooks; i++)
            _codebookEmbed[i] = WhisperOps.EnsureF32(w[$"{p}embed_tokens.{i}.weight"]);
        _encToDecW = WhisperOps.EnsureF32(w["enc_to_dec_proj.weight"]);
        for (int i = 0; i < _blocks.Length; i++) _blocks[i].LoadWeights(w, $"{p}layers.{i}");
        _lnOutG = WhisperOps.EnsureF32(w[$"{p}layer_norm.weight"]);
        _lnOutB = WhisperOps.EnsureF32(w[$"{p}layer_norm.bias"]);
        for (int i = 0; i < _cfg.NumCodebooks; i++)
            _heads[i] = WhisperOps.EnsureF32(w[$"lm_heads.{i}.weight"]);
    }

    /// <summary>Projects raw T5 states <c>[1, T_text, textDim]</c> to the decoder's cross-attn space
    /// <c>[1, T_text, hidden]</c> (the once-per-generation <c>enc_to_dec_proj</c>).</summary>
    public Tensor ProjectText(IBackend backend, Tensor t5States)
    {
        int tt = (int)t5States.Shape[1];
        return WhisperOps.ProjectLinear(backend, t5States, _encToDecW!, bias: null, 1, tt, _cfg.TextDim, _cfg.Hidden);
    }

    /// <summary>Embeds a sequence of K-codebook frames (sum of the per-codebook embeddings) + sinusoidal
    /// positions → <c>[1, T, hidden]</c>. <paramref name="frames"/> is <c>[T, K]</c>.</summary>
    public Tensor EmbedFrames(int[,] frames)
    {
        int t = frames.GetLength(0);
        int h = _cfg.Hidden;
        Tensor outT = new(new TensorShape(1, t, h), DType.F32);
        float* op = (float*)outT.DataPointer;
        for (int s = 0; s < t; s++)
        {
            long row = (long)s * h;
            for (int cb = 0; cb < _cfg.NumCodebooks; cb++)
            {
                int id = Math.Clamp(frames[s, cb], 0, _cfg.CodebookSize);
                float* tab = (float*)_codebookEmbed[cb]!.DataPointer + (long)id * h;
                for (int c = 0; c < h; c++) op[row + c] += tab[c];
            }
            AddSinusoid(op + row, s, h);
        }
        return outT;
    }

    /// <summary>Runs the decoder stack and returns the K next-step logits <c>[K][codebookSize]</c> from
    /// the last position. <paramref name="cross"/> is the projected T5 states.</summary>
    public float[][] Forward(IBackend backend, Tensor inputEmbeds, Tensor cross)
    {
        int t = (int)inputEmbeds.Shape[1];
        int h = _cfg.Hidden;
        Tensor? mask = t > 1 ? BuildCausalMask(t) : null;
        Tensor hidden = inputEmbeds;
        bool owns = false;
        for (int i = 0; i < _blocks.Length; i++)
        {
            Tensor next = _blocks[i].Forward(backend, hidden, cross, mask);
            if (owns) hidden.Dispose();
            hidden = next; owns = true;
        }
        mask?.Dispose();

        Tensor normed = new(hidden.Shape, DType.F32);
        backend.LayerNorm(normed, hidden, _lnOutG!, _lnOutB!, 1e-5f);
        if (owns) hidden.Dispose();

        Tensor last = SliceLast(normed, h);
        normed.Dispose();
        float[][] logits = new float[_cfg.NumCodebooks][];
        for (int cb = 0; cb < _cfg.NumCodebooks; cb++)
        {
            Tensor l = WhisperOps.ProjectLinear(backend, last, _heads[cb]!, bias: null, 1, 1, h, _cfg.CodebookSize);
            logits[cb] = new float[_cfg.CodebookSize];
            new Span<float>((void*)l.DataPointer, _cfg.CodebookSize).CopyTo(logits[cb]);
            l.Dispose();
        }
        last.Dispose();
        return logits;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? e in _codebookEmbed) if (e is not null) yield return e;
        if (_encToDecW is not null) yield return _encToDecW;
        foreach (MusicGenBlock b in _blocks) foreach (Tensor t in b.EnumerateWeights()) yield return t;
        if (_lnOutG is not null) yield return _lnOutG;
        if (_lnOutB is not null) yield return _lnOutB;
        foreach (Tensor? hd in _heads) if (hd is not null) yield return hd;
    }

    private static void AddSinusoid(float* row, int pos, int h)
    {
        int half = h / 2;
        for (int i = 0; i < half; i++)
        {
            double freq = Math.Exp(-Math.Log(10000.0) * i / Math.Max(1, half - 1));
            row[i] += (float)Math.Sin(pos * freq);
            row[half + i] += (float)Math.Cos(pos * freq);
        }
    }

    private static Tensor BuildCausalMask(int t)
    {
        Tensor mask = new(new TensorShape(1, 1, t, t), DType.F32);
        float* mp = (float*)mask.DataPointer;
        for (int q = 0; q < t; q++)
            for (int k = 0; k < t; k++)
                mp[(long)q * t + k] = k <= q ? 0f : float.NegativeInfinity;
        return mask;
    }

    private static Tensor SliceLast(Tensor hidden, int h)
    {
        int t = (int)hidden.Shape[1];
        Tensor last = new(new TensorShape(1, 1, h), DType.F32);
        Buffer.MemoryCopy((float*)hidden.DataPointer + (long)(t - 1) * h, (void*)last.DataPointer, h * 4, h * 4);
        return last;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        foreach (MusicGenBlock b in _blocks) b.Dispose();
        GC.SuppressFinalize(this);
    }
}
