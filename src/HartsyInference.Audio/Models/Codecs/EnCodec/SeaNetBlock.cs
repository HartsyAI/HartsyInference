using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Codecs.EnCodec;

/// <summary>SEANet residual block. Two-conv bottleneck with ELU activations between,
/// gated residual add. Mirrors <c>SEANetResnetBlock</c> from <c>facebookresearch/encodec</c>:
/// <code>
///   r = x
///   y = Conv1d(dim, dim/compress, k=kernel_size, dilation=d)(ELU(x))
///   y = Conv1d(dim/compress, dim, k=1, dilation=1)(ELU(y))
///   return r + y
/// </code>
/// where <c>compress=2</c> and <c>dilation</c> grows as <c>base^j</c> per block index
/// (just <c>j=0</c>, i.e. dilation=1, in the published 24 kHz checkpoint since
/// <c>n_residual_layers=1</c>).
///
/// <para>All convs use causal padding when <see cref="EnCodecConfig.Causal"/> is true
/// (left-only zero pad sized exactly to preserve the receptive field). Weight-norm is
/// fused at load time via <see cref="WeightNormFusion"/>.</para>
///
/// <para>The shortcut path is always identity here — the published EnCodec uses
/// <c>true_skip=True</c>. If a checkpoint flips that flag we'd need an additional 1×1
/// projection on the shortcut, but no shipping config exercises that path.</para></summary>
internal sealed unsafe class SeaNetBlock
{
    private readonly EnCodecConfig _cfg;
    private readonly string _prefix;
    private readonly int _dim;
    private readonly int _hidden;
    private readonly int _kernelSize;
    private readonly int _dilation;

    // Two convs in the block: index 1 (k=kernel_size, dilation=d) and index 3 (k=1, dilation=1).
    // The block is a Sequential with [ELU, Conv1d, ELU, Conv1d] in PyTorch, so indices
    // 1 and 3 carry weights; 0 and 2 are activations with no params.
    private Tensor? _conv1W;
    private Tensor? _conv1B;
    private Tensor? _conv2W;
    private Tensor? _conv2B;

    public SeaNetBlock(EnCodecConfig cfg, string prefix, int dim, int dilation)
    {
        _cfg = cfg;
        _prefix = prefix;
        _dim = dim;
        _hidden = dim / cfg.Compress;
        if (_hidden * cfg.Compress != dim)
            throw new ArgumentException($"dim {dim} must be divisible by compress {cfg.Compress}.");
        _kernelSize = cfg.ResidualKernelSize;
        _dilation = dilation;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        // Fail fast with a self-explaining message when the residual block this object was
        // constructed for is absent from the checkpoint. This is the common per-checkpoint
        // mismatch: NResidualLayers = ResidualDilations.Count over-counts the blocks (e.g. the
        // Kyutai DSM Mimi ships 1 block/stage but the default config implies 2), so the loader
        // would otherwise throw a bare KeyNotFoundException deep inside LoadFusedConvWeight.
        string conv1 = $"{_prefix}.block.1.conv.conv";
        if (!w.ContainsKey($"{conv1}.weight") && !w.ContainsKey($"{conv1}.weight_g"))
            throw new InvalidOperationException(
                $"SeaNet residual block '{_prefix}': expected conv key '{conv1}.weight[_g]' not found in the " +
                $"checkpoint. The checkpoint likely has fewer residual blocks per stage than NResidualLayers=" +
                $"{_cfg.NResidualLayers} (= ResidualDilations.Count). Reduce ResidualDilations on the config preset.");

        // PyTorch state-dict keys for the SEANet block convs follow the Sequential
        // index ordering, with the SConv1d wrapper adding ".conv.conv" to reach the
        // bare nn.Conv1d. Weight-norm: keys end with weight_g + weight_v (no fused
        // ".weight"). We fuse on load.
        _conv1W = LoadFusedConvWeight(w, $"{_prefix}.block.1.conv.conv");
        _conv1B = WhisperOps.EnsureF32(w[$"{_prefix}.block.1.conv.conv.bias"]);
        _conv2W = LoadFusedConvWeight(w, $"{_prefix}.block.3.conv.conv");
        _conv2B = WhisperOps.EnsureF32(w[$"{_prefix}.block.3.conv.conv.bias"]);
    }

    /// <summary>Forward — <paramref name="x"/> channels-first <c>[B, dim, T]</c>.
    /// Returns a fresh <c>[B, dim, T]</c> tensor (caller-owned). Input is NOT disposed.</summary>
    public Tensor Forward(IBackend backend, Tensor x, int batch, int t)
    {
        if (_conv1W is null) throw new InvalidOperationException($"SeaNetBlock '{_prefix}' weights not loaded.");

        // Path 1: ELU(x) → Conv1d(dim → hidden, k=residual_kernel_size, dilation=d).
        Tensor a1 = new(x.Shape, DType.F32);
        backend.Elu(a1, x, _cfg.EluAlpha);

        // stride=1 → the conv preserves T. Padding is causal (left-only) for the 24 kHz model, symmetric for the
        // non-causal 32/16 kHz codecs, and zero- or reflect-filled per cfg.PadMode (matches EncodecConv1d.forward).
        int t1 = t;
        Tensor mid = EnCodecConvPad.PaddedConv(backend, a1, _conv1W!, _conv1B,
            batch, _dim, t, _hidden, _kernelSize, _dilation, _cfg.Causal, _cfg.PadMode);
        a1.Dispose();

        // Path 2: ELU(mid) → Conv1d(hidden → dim, k=1). With k=1 the conv preserves T.
        Tensor a2 = new(mid.Shape, DType.F32);
        backend.Elu(a2, mid, _cfg.EluAlpha);
        mid.Dispose();

        Tensor proj = new(new TensorShape(batch, _dim, t1), DType.F32);
        backend.Conv1d(proj, a2, _conv2W!, _conv2B,
            stride: 1, padLeft: 0, padRight: 0,
            dilation: 1, groups: 1);
        a2.Dispose();

        // Residual add. proj should have T_out == T_in (extra right pad of conv1 + 1×1
        // means proj T equals x T plus the extra-right alignment). Trim if needed.
        int tProj = (int)proj.Shape[2];
        Tensor result;
        if (tProj == t)
        {
            result = new(x.Shape, DType.F32);
            backend.Add(result, x, proj);
        }
        else
        {
            // Trim trailing samples from proj so it matches x's time dim.
            Tensor projTrim = new(x.Shape, DType.F32);
            float* sp = (float*)proj.DataPointer;
            float* dp = (float*)projTrim.DataPointer;
            for (int b = 0; b < batch; b++)
                for (int c = 0; c < _dim; c++)
                {
                    int srcBase = (b * _dim + c) * tProj;
                    int dstBase = (b * _dim + c) * t;
                    int take = Math.Min(t, tProj);
                    for (int j = 0; j < take; j++) dp[dstBase + j] = sp[srcBase + j];
                    for (int j = take; j < t; j++) dp[dstBase + j] = 0f;
                }
            result = new(x.Shape, DType.F32);
            backend.Add(result, x, projTrim);
            projTrim.Dispose();
        }
        proj.Dispose();
        return result;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_conv1W, _conv1B, _conv2W, _conv2B];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }

    /// <summary>Loads a weight-normed conv weight by reading <c>weight_g</c> + <c>weight_v</c>
    /// and fusing them. Used wherever EnCodec / SEANet checkpoints store reparametrized
    /// conv weights.</summary>
    private static Tensor LoadFusedConvWeight(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        // Accept weight-normed (weight_g/weight_v) and pre-composed (weight) conv checkpoints.
        if (w.TryGetValue($"{prefix}.weight_g", out Tensor? g) && w.TryGetValue($"{prefix}.weight_v", out Tensor? v))
            return WeightNormFusion.Fuse(WhisperOps.EnsureF32(g), WhisperOps.EnsureF32(v));
        return WhisperOps.EnsureF32(w[$"{prefix}.weight"]);
    }

    /// <summary>Replicates the upstream <c>get_extra_padding_for_conv1d</c> stride-alignment
    /// rule so encoder-side convs produce sequence lengths that round up cleanly.</summary>
    private static int GetExtraRightPadding(int tIn, int kernel, int stride, int padTotal)
    {
        float nFrames = ((float)tIn - kernel + padTotal) / stride + 1f;
        int idealLength = ((int)MathF.Ceiling(nFrames) - 1) * stride + (kernel - padTotal);
        return Math.Max(0, idealLength - tIn);
    }
}
