using HartsyInference.Audio.Models.F5Tts;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Auk;

/// <summary>AuK Flux2Edit DiT velocity predictor: sequence [text | ref | noisy] through double-stream then single-stream blocks, returning the noisy slice.</summary>
// Batch 1, no attention masks: upstream masks are all-true except excess ref frames, which the pipeline avoids by
// trimming the reference to a whole number of latent hops. CFG uncond = ProjectText(drop: true) plus dropAudioCond.
public sealed unsafe class AukDit : IDisposable
{
    private const int RopeGrowStep = 256;

    private readonly AukConfig _config;
    private readonly F5TimestepEmbedding _timeEmbed;
    private readonly AukAudioEmbed _audioEmbed;
    private readonly AukDoubleBlock[] _double;
    private readonly AukSingleBlock[] _single;
    private Tensor? _txtProjW, _txtProjB, _txtNormW;
    private Tensor? _normOutW, _normOutB, _projOutW, _projOutB;
    private float[]? _invFreq;
    private Tensor? _ropeCos, _ropeSin;
    private int _ropePositions;
    private bool _loaded;

    public AukDit(AukConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        if (config.Heads < 1 || config.Dim % config.Heads != 0 || config.HeadDim % 2 != 0)
            throw new ArgumentException("Dim must split into an even head dimension.", nameof(config));
        _config = config;
        _timeEmbed = new F5TimestepEmbedding(new F5TtsConfig { Dim = config.Dim, TimeFreqEmbedDim = config.TimeFreqEmbedDim });
        _audioEmbed = new AukAudioEmbed(config);
        _double = new AukDoubleBlock[config.DoubleBlocks];
        for (int i = 0; i < _double.Length; i++) _double[i] = new AukDoubleBlock(config);
        _single = new AukSingleBlock[config.SingleBlocks];
        for (int i = 0; i < _single.Length; i++) _single[i] = new AukSingleBlock(config);
    }

    public AukConfig Config => _config;

    /// <summary>The rotary inverse frequencies loaded from the checkpoint buffer.</summary>
    public ReadOnlySpan<float> InvFreq => _invFreq;

    /// <summary>Loads and shape-validates every tensor under <paramref name="prefix"/> (<c>transformer</c> in AuK checkpoints); throws <see cref="InvalidDataException"/> on a missing or misshapen key.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix = "transformer")
    {
        ArgumentNullException.ThrowIfNull(weights);
        int dim = _config.Dim;
        string time = $"{prefix}.time_embed.time_mlp";
        AukOps.Require(weights, $"{time}.0.weight", dim, _config.TimeFreqEmbedDim);
        AukOps.Require(weights, $"{time}.0.bias", dim);
        AukOps.Require(weights, $"{time}.2.weight", dim, dim);
        AukOps.Require(weights, $"{time}.2.bias", dim);
        _timeEmbed.LoadWeights(weights, $"{prefix}.time_embed");
        _audioEmbed.LoadWeights(weights, $"{prefix}.audio_embed");
        _txtProjW = AukOps.Take(weights, $"{prefix}.txt_proj.weight", dim, _config.TextDim);
        _txtProjB = AukOps.Take(weights, $"{prefix}.txt_proj.bias", dim);
        _txtNormW = AukOps.Take(weights, $"{prefix}.txt_norm.weight", dim);
        for (int i = 0; i < _double.Length; i++) _double[i].LoadWeights(weights, $"{prefix}.transformer_blocks.{i}");
        for (int i = 0; i < _single.Length; i++) _single[i].LoadWeights(weights, $"{prefix}.single_transformer_blocks.{i}");
        _normOutW = AukOps.Take(weights, $"{prefix}.norm_out.linear.weight", 2 * dim, dim);
        _normOutB = AukOps.Take(weights, $"{prefix}.norm_out.linear.bias", 2 * dim);
        _projOutW = AukOps.Take(weights, $"{prefix}.proj_out.weight", _config.LatentDim, dim);
        _projOutB = AukOps.Take(weights, $"{prefix}.proj_out.bias", _config.LatentDim);

        // The checkpoint's bf16-rounded inv_freq differs from theta^(-2i/d), so it is the source of truth.
        Tensor inv = AukOps.Take(weights, $"{prefix}.rotary_embed.inv_freq", _config.HeadDim / 2);
        _invFreq = new ReadOnlySpan<float>((float*)inv.DataPointer, (int)inv.ElementCount).ToArray();
        _ropeCos?.Dispose();
        _ropeSin?.Dispose();
        _ropeCos = _ropeSin = null;
        _ropePositions = 0;
        _loaded = true;
    }

    /// <summary>Text conditioning <c>RMSNorm(Linear(text))</c> for raw thinker states <c>[1, nt, textDim]</c>; with <paramref name="drop"/> (CFG uncond) it is zeros AFTER the norm, not a projection of zeros. Step-invariant, so compute once per branch; the caller owns the result.</summary>
    public Tensor ProjectText(IBackend backend, Tensor text, bool drop = false)
    {
        EnsureLoaded();
        if (text.Shape.Rank != 3 || text.Shape[0] != 1 || text.Shape[2] != _config.TextDim)
            throw new ArgumentException($"text must be [1, nt, {_config.TextDim}].", nameof(text));
        int nt = (int)text.Shape[1];
        if (nt < 1) throw new ArgumentException("text must have at least one token.", nameof(text));
        Tensor result = new(new TensorShape(1, nt, _config.Dim), DType.F32);
        if (drop)
        {
            backend.Fill(result, 0f);
            return result;
        }
        Tensor proj = WhisperOps.ProjectLinear(backend, text, _txtProjW!, _txtProjB, 1, nt, _config.TextDim, _config.Dim);
        backend.RmsNorm(result, proj, _txtNormW!, AukOps.RmsEps);
        proj.Dispose();
        return result;
    }

    /// <summary>Returns cos/sin tables with at least <paramref name="positions"/> rows built from the loaded inv_freq; owned by this instance, valid until the next growth or dispose.</summary>
    public (Tensor Cos, Tensor Sin) GetRopeTables(int positions)
    {
        EnsureLoaded();
        if (_ropeCos is null || _ropeSin is null || positions > _ropePositions)
        {
            _ropeCos?.Dispose();
            _ropeSin?.Dispose();
            int capacity = (positions + RopeGrowStep - 1) / RopeGrowStep * RopeGrowStep;
            (_ropeCos, _ropeSin) = AukRope.BuildTables(_invFreq, capacity);
            _ropePositions = capacity;
        }
        return (_ropeCos, _ropeSin);
    }

    /// <summary>One velocity prediction <c>[1, N, latent]</c>; <paramref name="refLatent"/> is <c>[1, R, latent]</c> or null for none, and <paramref name="dropAudioCond"/> zeroes it before the embedding (CFG uncond). <paramref name="textCond"/> comes from <see cref="ProjectText"/>.</summary>
    public Tensor Forward(IBackend backend, Tensor noisy, Tensor? refLatent, Tensor textCond, float timestep, bool dropAudioCond = false)
    {
        EnsureLoaded();
        int dim = _config.Dim;
        int latent = _config.LatentDim;
        if (noisy.Shape.Rank != 3 || noisy.Shape[0] != 1 || noisy.Shape[2] != latent)
            throw new ArgumentException($"noisy must be [1, N, {latent}].", nameof(noisy));
        if (textCond.Shape.Rank != 3 || textCond.Shape[0] != 1 || textCond.Shape[2] != dim)
            throw new ArgumentException($"textCond must be [1, nt, {dim}].", nameof(textCond));
        int n = (int)noisy.Shape[1];
        int nt = (int)textCond.Shape[1];
        int r = 0;
        if (refLatent is not null)
        {
            if (refLatent.Shape.Rank != 3 || refLatent.Shape[0] != 1 || refLatent.Shape[2] != latent)
                throw new ArgumentException($"refLatent must be [1, R, {latent}].", nameof(refLatent));
            r = (int)refLatent.Shape[1];
        }
        if (n < 1 || nt < 1) throw new ArgumentException("noisy and text must be non-empty.");
        int seq = r + n;

        Tensor timeEmb = _timeEmbed.Forward(backend, timestep);
        Tensor siluTime = new(timeEmb.Shape, DType.F32);
        backend.Silu(siluTime, timeEmb);
        timeEmb.Dispose();

        Tensor noisyEmb = _audioEmbed.Forward(backend, noisy, n);
        Tensor x = noisyEmb;
        if (r > 0)
        {
            Tensor refEmb = EmbedRef(backend, refLatent!, r, dropAudioCond);
            x = new Tensor(new TensorShape(1, seq, dim), DType.F32);
            backend.Concat(x, [refEmb, noisyEmb], 1);
            refEmb.Dispose();
            noisyEmb.Dispose();
        }

        (Tensor cos, Tensor sin) = GetRopeTables(nt + seq);
        Tensor c = textCond;
        for (int i = 0; i < _double.Length; i++)
        {
            (Tensor cNext, Tensor xNext) = _double[i].Forward(backend, x, c, siluTime, seq, nt, cos, sin);
            x.Dispose();
            if (!ReferenceEquals(c, textCond)) c.Dispose();
            c = cNext;
            x = xNext;
        }

        Tensor joint = new(new TensorShape(1, nt + seq, dim), DType.F32);
        backend.Concat(joint, [c, x], 1);
        if (!ReferenceEquals(c, textCond)) c.Dispose();
        x.Dispose();
        for (int i = 0; i < _single.Length; i++)
        {
            Tensor next = _single[i].Forward(backend, joint, siluTime, nt + seq, cos, sin);
            joint.Dispose();
            joint = next;
        }

        Tensor prefix = new(new TensorShape(1, nt + r, dim), DType.F32);
        Tensor tail = new(new TensorShape(1, n, dim), DType.F32);
        backend.Split([prefix, tail], joint, 1);
        joint.Dispose();
        prefix.Dispose();
        Tensor head = AukOps.FinalAdaLn(backend, tail, _normOutW!, _normOutB!, siluTime, dim, _config.NormEps);
        tail.Dispose();
        siluTime.Dispose();
        Tensor velocity = WhisperOps.ProjectLinear(backend, head, _projOutW!, _projOutB, 1, n, dim, latent);
        head.Dispose();
        return velocity;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _timeEmbed.EnumerateWeights()) yield return t;
        foreach (Tensor t in _audioEmbed.EnumerateWeights()) yield return t;
        Tensor?[] top = [_txtProjW, _txtProjB, _txtNormW, _normOutW, _normOutB, _projOutW, _projOutB];
        foreach (Tensor? t in top) if (t is not null) yield return t;
        foreach (AukDoubleBlock b in _double)
            foreach (Tensor t in b.EnumerateWeights()) yield return t;
        foreach (AukSingleBlock b in _single)
            foreach (Tensor t in b.EnumerateWeights()) yield return t;
    }

    public void Dispose()
    {
        _ropeCos?.Dispose();
        _ropeSin?.Dispose();
        _ropeCos = _ropeSin = null;
    }

    private Tensor EmbedRef(IBackend backend, Tensor refLatent, int r, bool drop)
    {
        if (!drop) return _audioEmbed.Forward(backend, refLatent, r);
        Tensor zeros = new(refLatent.Shape, DType.F32);
        backend.Fill(zeros, 0f);
        Tensor embedded = _audioEmbed.Forward(backend, zeros, r);
        zeros.Dispose();
        return embedded;
    }

    private void EnsureLoaded()
    {
        if (!_loaded) throw new InvalidOperationException("Call LoadWeights first.");
    }
}
