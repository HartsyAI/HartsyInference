using SharpInference.Audio.Layers;
using SharpInference.Audio.Models.Whisper;
using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Models.Codecs.EnCodec;

/// <summary>SEANet encoder for EnCodec. Mirrors <c>SEANetEncoder</c> from
/// <c>facebookresearch/encodec</c>:
/// <code>
///   stem: Conv1d(channels=1 → n_filters=32, k=7)
///   for ratio in reversed(ratios) (= [2, 4, 5, 8] for the 24 kHz model):
///     for j in range(n_residual_layers):
///       SeaNetBlock(dim, dilation=base^j)
///     ELU
///     Conv1d(dim → 2*dim, k=ratio*2, stride=ratio)   # downsample
///     dim *= 2
///   2-layer unidir LSTM at hidden = final dim (512 for the 24 kHz model)
///   ELU
///   final: Conv1d(dim → latent_dim=128, k=7)         # projection
/// </code>
///
/// <para>Input is <c>[B, 1, T_pcm]</c>; output is <c>[B, latent_dim, T_pcm / hop]</c>
/// where <c>hop = product(ratios) = 320</c> for the 24 kHz model. The intermediate LSTM
/// runs on the channels-last view <c>[B, T_frames, dim]</c> and is the part that
/// distinguishes EnCodec's "convolutional-and-recurrent" encoder from the pure-conv
/// VAEs we built for VibeVoice.</para></summary>
internal sealed class SeaNetEncoder
{
    private readonly EnCodecConfig _cfg;
    private readonly string _prefix;
    private readonly int[] _ratiosRev;       // reversed: [2, 4, 5, 8]
    private readonly int _stages;             // = ratios.Count

    // Stem conv (weight-norm fused).
    private Tensor? _stemW;
    private Tensor? _stemB;

    // Per-stage residual blocks (one per j in [0..n_residual_layers)) and downsample conv.
    private readonly SeaNetBlock[][] _stageBlocks;
    private readonly Tensor?[] _downsampleW;
    private readonly Tensor?[] _downsampleB;

    // LSTM bottleneck — null when cfg.LstmLayers == 0 (Mimi substitutes a transformer-of-codecs).
    private readonly UnidirectionalLstm? _lstm;

    // Final projection.
    private Tensor? _finalW;
    private Tensor? _finalB;

    public SeaNetEncoder(EnCodecConfig cfg, string prefix)
    {
        _cfg = cfg;
        _prefix = prefix;
        _stages = cfg.Ratios.Count;
        _ratiosRev = new int[_stages];
        for (int i = 0; i < _stages; i++) _ratiosRev[i] = cfg.Ratios[_stages - 1 - i];

        _stageBlocks = new SeaNetBlock[_stages][];
        _downsampleW = new Tensor?[_stages];
        _downsampleB = new Tensor?[_stages];

        // Per the upstream construction the encoder builds a flat nn.Sequential where
        // index 0 is the stem, then each stage adds N residual blocks + ELU + downsample
        // conv, then LSTM is appended, then ELU + final conv. We mirror this by
        // tracking the running Sequential index that PyTorch state-dict keys use.
        int seqIdx = 1;     // after stem at index 0
        for (int i = 0; i < _stages; i++)
        {
            int dimIn = cfg.NFilters * (1 << i);
            _stageBlocks[i] = new SeaNetBlock[cfg.NResidualLayers];
            for (int j = 0; j < cfg.NResidualLayers; j++)
            {
                int dilation = (int)Math.Pow(cfg.DilationBase, j);
                _stageBlocks[i][j] = new SeaNetBlock(cfg, $"{prefix}.model.{seqIdx}", dimIn, dilation);
                seqIdx++;
            }
            seqIdx++;       // ELU activation slot
            seqIdx++;       // downsample SConv1d slot (recorded below)
        }

        // LSTM at the end of the conv stack — input dim = output channels after all stages.
        // cfg.LstmLayers == 0 disables the LSTM (used by Mimi, which substitutes a
        // transformer-of-codecs after the encoder).
        int finalDim = cfg.NFilters * (1 << _stages);
        _lstm = cfg.LstmLayers > 0
            ? new UnidirectionalLstm(finalDim, finalDim, cfg.LstmLayers)
            : null;
        // ELU + final projection follow.
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        // Stem (Sequential index 0).
        _stemW = LoadFusedConvWeight(w, $"{_prefix}.model.0.conv.conv");
        _stemB = WhisperOps.EnsureF32(w[$"{_prefix}.model.0.conv.conv.bias"]);

        // Stages: each stage is residual blocks + ELU + downsample. Track running index.
        int seqIdx = 1;
        for (int i = 0; i < _stages; i++)
        {
            for (int j = 0; j < _cfg.NResidualLayers; j++)
            {
                _stageBlocks[i][j].LoadWeights(w);
                seqIdx++;
            }
            // ELU at seqIdx (no weights), downsample conv at seqIdx + 1.
            seqIdx++;       // skip ELU
            _downsampleW[i] = LoadFusedConvWeight(w, $"{_prefix}.model.{seqIdx}.conv.conv");
            _downsampleB[i] = WhisperOps.EnsureF32(w[$"{_prefix}.model.{seqIdx}.conv.conv.bias"]);
            seqIdx++;
        }

        // LSTM. PyTorch's state-dict key prefix is the running Sequential index of the
        // SLSTM wrapper. The actual nn.LSTM is named <prefix>.lstm.weight_ih_l0 etc.
        // Skip when LstmLayers == 0 (Mimi — transformer replaces the LSTM bottleneck).
        if (_lstm is not null)
        {
            _lstm.LoadWeights(w, $"{_prefix}.model.{seqIdx}.lstm");
            seqIdx++;
        }

        // Final ELU (no weights) + final conv at seqIdx + 1.
        seqIdx++;       // skip ELU
        _finalW = LoadFusedConvWeight(w, $"{_prefix}.model.{seqIdx}.conv.conv");
        _finalB = WhisperOps.EnsureF32(w[$"{_prefix}.model.{seqIdx}.conv.conv.bias"]);
    }

    /// <summary>Forward — <paramref name="pcm"/> channels-first <c>[batch, 1, T_pcm]</c>.
    /// Returns <c>[batch, latent_dim, T_latent]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor pcm, int batch, int tPcm)
    {
        if (_stemW is null) throw new InvalidOperationException("SeaNetEncoder weights not loaded.");

        // Stem: causal Conv1d(1 → n_filters, k=kernel_size, stride=1).
        int padTotal = _cfg.KernelSize - 1;     // stride=1
        int extraRight = GetExtraRightPadding(tPcm, _cfg.KernelSize, 1, padTotal);
        int tStem = tPcm + padTotal + extraRight - (_cfg.KernelSize - 1);
        Tensor x = new(new TensorShape(batch, _cfg.NFilters, tStem), DType.F32);
        backend.Conv1d(x, pcm, _stemW!, _stemB,
            stride: 1, padLeft: padTotal, padRight: extraRight, dilation: 1, groups: 1);

        int t = tStem;
        int dim = _cfg.NFilters;

        // Stages.
        for (int i = 0; i < _stages; i++)
        {
            // Residual blocks.
            foreach (SeaNetBlock block in _stageBlocks[i])
            {
                Tensor next = block.Forward(backend, x, batch, t);
                x.Dispose();
                x = next;
            }

            // ELU.
            Tensor activated = new(x.Shape, DType.F32);
            backend.Elu(activated, x, _cfg.EluAlpha);
            x.Dispose();
            x = activated;

            // Downsample conv: k=ratio*2, stride=ratio, causal.
            int ratio = _ratiosRev[i];
            int kDown = ratio * 2;
            int padTotalDown = (kDown - 1) - (ratio - 1);    // (k-1)*1 - (stride-1) for dilation=1
            int extraRightDown = GetExtraRightPadding(t, kDown, ratio, padTotalDown);
            int tDown = (t + padTotalDown + extraRightDown - (kDown - 1) - 1) / ratio + 1;
            int dimOut = dim * 2;
            Tensor down = new(new TensorShape(batch, dimOut, tDown), DType.F32);
            backend.Conv1d(down, x, _downsampleW[i]!, _downsampleB[i],
                stride: ratio, padLeft: padTotalDown, padRight: extraRightDown,
                dilation: 1, groups: 1);
            x.Dispose();
            x = down;
            t = tDown;
            dim = dimOut;
        }

        // LSTM bottleneck (skipped when _lstm is null — Mimi case). Convert
        // [B, dim, T] → [B, T, dim] for the LSTM, residual-add, back to channels-first.
        Tensor cf;
        if (_lstm is not null)
        {
            Tensor cl = new(new TensorShape(batch, t, dim), DType.F32);
            backend.Transpose2D(cl, x, dim, t);
            x.Dispose();
            Tensor lstmOut = _lstm.Forward(backend, cl, batch, t);
            // EnCodec adds the LSTM output to the LSTM input (residual). PyTorch's
            // SLSTM.forward does: y = self.lstm(x.permute(2,0,1))[0].permute(1,2,0) + x.
            // Permute-wise our lstmOut already matches cl's [B, T, dim] layout — so just add.
            Tensor cl2 = new(cl.Shape, DType.F32);
            backend.Add(cl2, cl, lstmOut);
            cl.Dispose();
            lstmOut.Dispose();
            cf = new(new TensorShape(batch, dim, t), DType.F32);
            backend.Transpose2D(cf, cl2, t, dim);
            cl2.Dispose();
        }
        else
        {
            // No LSTM bottleneck — Mimi handles temporal context via its
            // transformer-of-codecs layer after the encoder.
            cf = x;
        }

        // ELU.
        Tensor activatedFinal = new(cf.Shape, DType.F32);
        backend.Elu(activatedFinal, cf, _cfg.EluAlpha);
        cf.Dispose();

        // Final projection conv: causal Conv1d(dim → latent_dim, k=last_kernel_size, stride=1).
        int padTotalFinal = _cfg.LastKernelSize - 1;
        int extraRightFinal = GetExtraRightPadding(t, _cfg.LastKernelSize, 1, padTotalFinal);
        int tFinal = t + padTotalFinal + extraRightFinal - (_cfg.LastKernelSize - 1);
        Tensor latent = new(new TensorShape(batch, _cfg.LatentDim, tFinal), DType.F32);
        backend.Conv1d(latent, activatedFinal, _finalW!, _finalB,
            stride: 1, padLeft: padTotalFinal, padRight: extraRightFinal,
            dilation: 1, groups: 1);
        activatedFinal.Dispose();

        return latent;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_stemW is not null) yield return _stemW;
        if (_stemB is not null) yield return _stemB;
        for (int i = 0; i < _stages; i++)
        {
            foreach (SeaNetBlock block in _stageBlocks[i])
                foreach (Tensor t in block.EnumerateWeights()) yield return t;
            if (_downsampleW[i] is not null) yield return _downsampleW[i]!;
            if (_downsampleB[i] is not null) yield return _downsampleB[i]!;
        }
        if (_lstm is not null)
            foreach (Tensor t in _lstm.EnumerateWeights()) yield return t;
        if (_finalW is not null) yield return _finalW;
        if (_finalB is not null) yield return _finalB;
    }

    private static Tensor LoadFusedConvWeight(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        Tensor g = WhisperOps.EnsureF32(w[$"{prefix}.weight_g"]);
        Tensor v = WhisperOps.EnsureF32(w[$"{prefix}.weight_v"]);
        return WeightNormFusion.Fuse(g, v);
    }

    private static int GetExtraRightPadding(int tIn, int kernel, int stride, int padTotal)
    {
        float nFrames = ((float)tIn - kernel + padTotal) / stride + 1f;
        int idealLength = ((int)MathF.Ceiling(nFrames) - 1) * stride + (kernel - padTotal);
        return Math.Max(0, idealLength - tIn);
    }
}
