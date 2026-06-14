using HartsyInference.Audio.Layers;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Codecs.EnCodec;

/// <summary>SEANet decoder for EnCodec. Mirror of <see cref="SeaNetEncoder"/>:
/// <code>
///   initial: Conv1d(latent_dim=128 → max_channels=512, k=7)
///   2-layer unidir LSTM at hidden = 512 (residual add)
///   for ratio in ratios (= [8, 5, 4, 2] for the 24 kHz model):
///     ELU
///     ConvTranspose1d(dim → dim/2, k=ratio*2, stride=ratio)   # upsample, causal trim
///     for j in range(n_residual_layers):
///       SeaNetBlock(dim/2, dilation=base^j)
///     dim /= 2
///   ELU
///   final: Conv1d(n_filters=32 → channels=1, k=7)
/// </code>
///
/// <para>Input is <c>[B, latent_dim, T_frames]</c>; output is
/// <c>[B, channels=1, T_frames * 320]</c> for the 24 kHz model. The LSTM is positioned at
/// the "narrowest" point (bottleneck) — directly after the initial projection conv and
/// before any upsample stages, matching the encoder's symmetric placement.</para></summary>
internal sealed class SeaNetDecoder
{
    private readonly EnCodecConfig _cfg;
    private readonly string _prefix;
    private readonly int _stages;
    private readonly int[] _ratios;             // forward order: [8, 5, 4, 2]

    // Initial projection conv (Sequential index 0).
    private Tensor? _initialW;
    private Tensor? _initialB;

    // LSTM bottleneck (Sequential index 1).
    private readonly UnidirectionalLstm? _lstm;     // null when cfg.LstmLayers == 0 (Mimi case)

    // Per-stage upsample ConvTranspose1d + N residual blocks.
    private readonly Tensor?[] _upsampleW;
    private readonly Tensor?[] _upsampleB;
    private readonly SeaNetBlock[][] _stageBlocks;

    // Final ELU + Conv1d.
    private Tensor? _finalW;
    private Tensor? _finalB;

    public SeaNetDecoder(EnCodecConfig cfg, string prefix)
    {
        _cfg = cfg;
        _prefix = prefix;
        _stages = cfg.Ratios.Count;
        _ratios = [.. cfg.Ratios];

        int maxChannels = cfg.NFilters * (1 << _stages);
        _lstm = cfg.LstmLayers > 0 ? new UnidirectionalLstm(maxChannels, maxChannels, cfg.LstmLayers) : null;

        _upsampleW = new Tensor?[_stages];
        _upsampleB = new Tensor?[_stages];
        _stageBlocks = new SeaNetBlock[_stages][];

        int seqIdx = 2;     // after initial conv (0) + LSTM (1)
        for (int i = 0; i < _stages; i++)
        {
            int dimIn = cfg.NFilters * (1 << (_stages - i));
            int dimOut = dimIn / 2;
            seqIdx++;       // skip ELU
            seqIdx++;       // upsample SConvTranspose1d slot (recorded at LoadWeights)
            _stageBlocks[i] = new SeaNetBlock[cfg.NResidualLayers];
            for (int j = 0; j < cfg.NResidualLayers; j++)
            {
                int dilation = (int)Math.Pow(cfg.DilationBase, j);
                _stageBlocks[i][j] = new SeaNetBlock(cfg, $"{prefix}.model.{seqIdx}", dimOut, dilation);
                seqIdx++;
            }
        }
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        // Initial conv (Sequential index 0).
        _initialW = LoadFusedConvWeight(w, $"{_prefix}.model.0.conv.conv");
        _initialB = WhisperOps.EnsureF32(w[$"{_prefix}.model.0.conv.conv.bias"]);

        // LSTM (Sequential index 1) — absent when cfg.LstmLayers == 0 (Mimi case).
        _lstm?.LoadWeights(w, $"{_prefix}.model.1.lstm");

        // Per stage: ELU at idx, upsample at idx+1, residual blocks at idx+2..idx+1+N.
        int seqIdx = 2;
        for (int i = 0; i < _stages; i++)
        {
            seqIdx++;   // skip ELU
            // Upsample SConvTranspose1d wraps NormConvTranspose1d which wraps nn.ConvTranspose1d.
            // Weight key: model.{seqIdx}.convtr.convtr.weight_g / weight_v.
            _upsampleW[i] = LoadFusedConvTransposeWeight(w, $"{_prefix}.model.{seqIdx}.convtr.convtr");
            _upsampleB[i] = WhisperOps.EnsureF32(w[$"{_prefix}.model.{seqIdx}.convtr.convtr.bias"]);
            seqIdx++;
            for (int j = 0; j < _cfg.NResidualLayers; j++)
            {
                _stageBlocks[i][j].LoadWeights(w);
                seqIdx++;
            }
        }

        // Final ELU + Conv1d.
        seqIdx++;   // skip ELU
        _finalW = LoadFusedConvWeight(w, $"{_prefix}.model.{seqIdx}.conv.conv");
        _finalB = WhisperOps.EnsureF32(w[$"{_prefix}.model.{seqIdx}.conv.conv.bias"]);
    }

    /// <summary>Forward — <paramref name="latent"/> channels-first
    /// <c>[batch, latent_dim, T_frames]</c>. Returns <c>[batch, 1, T_frames * 320]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor latent, int batch, int tFrames)
    {
        if (_initialW is null) throw new InvalidOperationException("SeaNetDecoder weights not loaded.");

        // Initial projection: causal Conv1d(latent_dim → maxChannels, k=kernel_size).
        int maxChannels = _cfg.NFilters * (1 << _stages);
        int padTotalInit = _cfg.KernelSize - 1;
        int extraRightInit = GetExtraRightPadding(tFrames, _cfg.KernelSize, 1, padTotalInit);
        int tInit = tFrames + padTotalInit + extraRightInit - (_cfg.KernelSize - 1);
        Tensor x = new(new TensorShape(batch, maxChannels, tInit), DType.F32);
        backend.Conv1d(x, latent, _initialW!, _initialB,
            stride: 1, padLeft: padTotalInit, padRight: extraRightInit, dilation: 1, groups: 1);

        int t = tInit;
        int dim = maxChannels;

        // LSTM bottleneck on channels-last [B, T, dim], residual add, back to channels-first.
        // Skipped entirely when _lstm is null (Mimi — transformer-of-codecs replaces it).
        if (_lstm is not null)
        {
            Tensor cl = new(new TensorShape(batch, t, dim), DType.F32);
            backend.Transpose2D(cl, x, dim, t);
            x.Dispose();
            Tensor lstmOut = _lstm.Forward(backend, cl, batch, t);
            Tensor cl2 = new(cl.Shape, DType.F32);
            backend.Add(cl2, cl, lstmOut);
            cl.Dispose();
            lstmOut.Dispose();
            Tensor cf = new(new TensorShape(batch, dim, t), DType.F32);
            backend.Transpose2D(cf, cl2, t, dim);
            cl2.Dispose();
            x = cf;
        }

        // Upsample stages.
        for (int i = 0; i < _stages; i++)
        {
            // ELU.
            Tensor activated = new(x.Shape, DType.F32);
            backend.Elu(activated, x, _cfg.EluAlpha);
            x.Dispose();
            x = activated;

            // ConvTranspose1d(dim → dim/2, k=ratio*2, stride=ratio). Causal trim:
            // padLeft = 0, padRight = K - stride.
            int ratio = _ratios[i];
            int kUp = ratio * 2;
            int padLeftUp = 0;
            int padRightUp = kUp - ratio;
            int dimOut = dim / 2;
            int tUp = t * ratio;     // (T_in - 1) * stride + K - padRight = T_in * stride
            Tensor up = new(new TensorShape(batch, dimOut, tUp), DType.F32);
            backend.ConvTranspose1d(up, x, _upsampleW[i]!, _upsampleB[i],
                stride: ratio, padLeft: padLeftUp, padRight: padRightUp, dilation: 1);
            x.Dispose();
            x = up;
            t = tUp;
            dim = dimOut;

            // Residual blocks.
            foreach (SeaNetBlock block in _stageBlocks[i])
            {
                Tensor next = block.Forward(backend, x, batch, t);
                x.Dispose();
                x = next;
            }
        }

        // Final ELU.
        Tensor activatedFinal = new(x.Shape, DType.F32);
        backend.Elu(activatedFinal, x, _cfg.EluAlpha);
        x.Dispose();

        // Final Conv1d(n_filters → channels=1, k=last_kernel_size).
        int padTotalFinal = _cfg.LastKernelSize - 1;
        int extraRightFinal = GetExtraRightPadding(t, _cfg.LastKernelSize, 1, padTotalFinal);
        int tOut = t + padTotalFinal + extraRightFinal - (_cfg.LastKernelSize - 1);
        Tensor pcm = new(new TensorShape(batch, _cfg.Channels, tOut), DType.F32);
        backend.Conv1d(pcm, activatedFinal, _finalW!, _finalB,
            stride: 1, padLeft: padTotalFinal, padRight: extraRightFinal, dilation: 1, groups: 1);
        activatedFinal.Dispose();

        return pcm;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_initialW is not null) yield return _initialW;
        if (_initialB is not null) yield return _initialB;
        if (_lstm is not null)
            foreach (Tensor t in _lstm.EnumerateWeights()) yield return t;
        for (int i = 0; i < _stages; i++)
        {
            if (_upsampleW[i] is not null) yield return _upsampleW[i]!;
            if (_upsampleB[i] is not null) yield return _upsampleB[i]!;
            foreach (SeaNetBlock block in _stageBlocks[i])
                foreach (Tensor t in block.EnumerateWeights()) yield return t;
        }
        if (_finalW is not null) yield return _finalW;
        if (_finalB is not null) yield return _finalB;
    }

    private static Tensor LoadFusedConvWeight(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        Tensor g = WhisperOps.EnsureF32(w[$"{prefix}.weight_g"]);
        Tensor v = WhisperOps.EnsureF32(w[$"{prefix}.weight_v"]);
        return WeightNormFusion.Fuse(g, v);
    }

    /// <summary>ConvTranspose1d weight has shape <c>[C_in, C_out, K]</c>. WeightNormFusion
    /// computes the per-out-channel norm along axis 0 of the input, but for transpose
    /// conv the "out" dim is axis 1, not axis 0. We need to fuse along axis 1 → easiest
    /// is to call the generic Fuse with the trailing-axis assumption swapped: weight_g
    /// for ConvTranspose1d has shape <c>[1, C_out, 1]</c> (per-output-channel scale on
    /// the middle axis), and weight_v has the same shape as the final weight. The
    /// L2-norm direction is along axis 0 and 2 (treating axis 1 = C_out as the "kept"
    /// dimension).</summary>
    private static Tensor LoadFusedConvTransposeWeight(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        Tensor g = WhisperOps.EnsureF32(w[$"{prefix}.weight_g"]);
        Tensor v = WhisperOps.EnsureF32(w[$"{prefix}.weight_v"]);
        return WeightNormFusionT.Fuse(g, v);
    }

    private static int GetExtraRightPadding(int tIn, int kernel, int stride, int padTotal)
    {
        float nFrames = ((float)tIn - kernel + padTotal) / stride + 1f;
        int idealLength = ((int)MathF.Ceiling(nFrames) - 1) * stride + (kernel - padTotal);
        return Math.Max(0, idealLength - tIn);
    }
}

/// <summary>WeightNormFusion variant for PyTorch's <c>ConvTranspose1d</c>. The
/// transpose-conv weight has layout <c>[C_in, C_out, K]</c> — so the "kept" axis (the
/// one we DON'T sum over for the L2 norm) is axis 1, not axis 0. The companion
/// <c>weight_g</c> has shape <c>[1, C_out, 1]</c> with one scalar per output channel.
///
/// <para>Fused weight is computed as
/// <c>w[ic, oc, k] = g[oc] * v[ic, oc, k] / ||v[:, oc, :]||_2</c>.</para></summary>
internal static unsafe class WeightNormFusionT
{
    public static Tensor Fuse(Tensor weightG, Tensor weightV)
    {
        if (weightV.Shape.Rank != 3) throw new ArgumentException($"WeightNormFusionT expects rank-3 weightV [C_in, C_out, K], got {weightV.Shape}.");
        int cIn = (int)weightV.Shape[0];
        int cOut = (int)weightV.Shape[1];
        int kernel = (int)weightV.Shape[2];
        if (weightG.ElementCount != cOut)
            throw new ArgumentException($"WeightNormFusionT expects weightG with {cOut} elements (C_out), got {weightG.ElementCount}.");

        Tensor fused = new(weightV.Shape, DType.F32);
        float* vp = (float*)weightV.DataPointer;
        float* gp = (float*)weightG.DataPointer;
        float* fp = (float*)fused.DataPointer;

        // Compute per-out-channel norm over (cIn, kernel) and apply.
        for (int oc = 0; oc < cOut; oc++)
        {
            double sumSq = 0d;
            for (int ic = 0; ic < cIn; ic++)
            {
                for (int k = 0; k < kernel; k++)
                {
                    float v = vp[(ic * cOut + oc) * kernel + k];
                    sumSq += (double)v * v;
                }
            }
            float norm = MathF.Sqrt((float)sumSq);
            float scale = gp[oc] * (norm > 0f ? 1f / norm : 0f);
            for (int ic = 0; ic < cIn; ic++)
            {
                for (int k = 0; k < kernel; k++)
                {
                    int idx = (ic * cOut + oc) * kernel + k;
                    fp[idx] = vp[idx] * scale;
                }
            }
        }
        return fused;
    }
}
