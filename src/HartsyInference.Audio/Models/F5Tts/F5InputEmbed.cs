using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.F5Tts;

/// <summary>F5-TTS input embedding: stacks the noisy target mel, reference/cond mel, and text features along channels ([100+100+512=712 ch]), projects 712→1024, then adds a residual refinement via <see cref="F5ConvPosEmbed"/>.</summary>
internal sealed unsafe class F5InputEmbed
{
    private readonly F5TtsConfig _cfg;

    private Tensor? _projW, _projB;
    private readonly F5ConvPosEmbed _convPos;

    public F5InputEmbed(F5TtsConfig cfg)
    {
        _cfg = cfg;
        _convPos = new F5ConvPosEmbed(cfg);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _projW = WhisperOps.EnsureF32(w[$"{prefix}.proj.weight"]);
        _projB = WhisperOps.EnsureF32(w[$"{prefix}.proj.bias"]);
        _convPos.LoadWeights(w, $"{prefix}.conv_pos_embed");
    }

    /// <summary><paramref name="noisyMel"/> and <paramref name="condMel"/> are both <c>[1, mel_dim, T]</c> channels-first, <paramref name="text"/> is <c>[1, T, text_dim]</c> channels-last; when <paramref name="dropAudioCond"/> is true (CFG uncond pass) the cond_mel is zero'd out.</summary>
    public Tensor Forward(IBackend backend, Tensor noisyMel, Tensor condMel, Tensor text, int t, bool dropAudioCond)
    {
        int melDim = _cfg.MelDim;
        int textDim = _cfg.TextDim;
        int catDim = 2 * melDim + textDim;  // 712
        int outDim = _cfg.Dim;

        // Build the concatenated [1, T, 712] tensor (channels-last). For each frame t,
        // first 100 floats = noisy mel, next 100 = cond mel (or zero), next 512 = text.
        Tensor concat = new(new TensorShape(1, t, catDim), DType.F32);
        float* cp = (float*)concat.DataPointer;
        float* np_ = (float*)noisyMel.DataPointer;  // shape [1, mel_dim, T]
        float* cmp = (float*)condMel.DataPointer;
        float* tp = (float*)text.DataPointer;

        for (int tt = 0; tt < t; tt++)
        {
            int dstRow = tt * catDim;
            // noisy mel [1, melDim, T] at (0, d, tt) = d*T + tt
            for (int d = 0; d < melDim; d++) cp[dstRow + d] = np_[d * t + tt];
            // cond mel
            if (dropAudioCond)
            {
                for (int d = 0; d < melDim; d++) cp[dstRow + melDim + d] = 0f;
            }
            else
            {
                for (int d = 0; d < melDim; d++) cp[dstRow + melDim + d] = cmp[d * t + tt];
            }
            // text features [1, T, textDim] at (0, tt, d) = tt*textDim + d
            int textRow = tt * textDim;
            for (int d = 0; d < textDim; d++) cp[dstRow + 2 * melDim + d] = tp[textRow + d];
        }

        // Linear proj 712 → 1024.
        Tensor projected = WhisperOps.ProjectLinear(backend, concat, _projW!, _projB, 1, t, catDim, outDim);
        concat.Dispose();

        // Residual: x = ConvPosEmbed(x) + x.
        Tensor posOut = _convPos.Forward(backend, projected, t, outDim);
        Tensor result = new(projected.Shape, DType.F32);
        backend.Add(result, projected, posOut);
        projected.Dispose();
        posOut.Dispose();
        return result;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_projW is not null) yield return _projW;
        if (_projB is not null) yield return _projB;
        foreach (Tensor t in _convPos.EnumerateWeights()) yield return t;
    }
}

/// <summary>F5-TTS conv positional embedding: two stacked grouped Conv1D layers (groups=16, kernel=31, channels-first) with Mish activations between, operating on channels-last <c>[1, T, dim]</c> input via transposition.</summary>
internal sealed unsafe class F5ConvPosEmbed
{
    private readonly F5TtsConfig _cfg;

    private Tensor? _conv0W, _conv0B;
    private Tensor? _conv2W, _conv2B;

    public F5ConvPosEmbed(F5TtsConfig cfg) { _cfg = cfg; }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        // The Sequential indexes 0 and 2 are the two Conv1D layers; 1 and 3 are Mish (no params).
        _conv0W = WhisperOps.EnsureF32(w[$"{prefix}.conv1d.0.weight"]);
        _conv0B = WhisperOps.EnsureF32(w[$"{prefix}.conv1d.0.bias"]);
        _conv2W = WhisperOps.EnsureF32(w[$"{prefix}.conv1d.2.weight"]);
        _conv2B = WhisperOps.EnsureF32(w[$"{prefix}.conv1d.2.bias"]);
    }

    /// <summary>Input/output <c>[1, T, dim]</c>; internally transposes to channels-first, runs two grouped Conv1D layers with Mish between, then transposes back.</summary>
    public Tensor Forward(IBackend backend, Tensor x, int t, int dim)
    {
        int kernel = _cfg.ConvPosKernel;
        int pad = kernel / 2;
        int groups = _cfg.ConvPosGroups;

        Tensor xCF = new(new TensorShape(1, dim, t), DType.F32);
        backend.Transpose2D(xCF, x, t, dim);

        // Conv1 — grouped Conv1D on-device (weight [C_out, C_in/groups, K] matches backend.Conv1d). The old
        // F5Ops.GroupedConv1D was a host quintuple-loop (~2.3B MACs/conv × 2 × NFE forwards) that pinned one
        // CPU core and was ~99% of F5's wall time; backend.Conv1d routes to cuDNN / a grouped CUDA kernel.
        Tensor c1 = new(xCF.Shape, DType.F32);
        backend.Conv1d(c1, xCF, _conv0W!, _conv0B!, stride: 1, padLeft: pad, padRight: pad, dilation: 1, groups: groups);
        xCF.Dispose();
        F5Ops.MishInPlace(c1);

        // Conv2
        Tensor c2 = new(c1.Shape, DType.F32);
        backend.Conv1d(c2, c1, _conv2W!, _conv2B!, stride: 1, padLeft: pad, padRight: pad, dilation: 1, groups: groups);
        c1.Dispose();
        F5Ops.MishInPlace(c2);

        // Transpose back to channels-last.
        Tensor outCL = new(new TensorShape(1, t, dim), DType.F32);
        backend.Transpose2D(outCL, c2, dim, t);
        c2.Dispose();
        return outCL;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_conv0W, _conv0B, _conv2W, _conv2B];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }
}
