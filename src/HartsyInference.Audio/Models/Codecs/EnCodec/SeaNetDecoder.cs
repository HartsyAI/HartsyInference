using HartsyInference.Audio.Layers;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Codecs.EnCodec;

/// <summary>SEANet decoder for EnCodec — mirror of <see cref="SeaNetEncoder"/>.</summary>
/// <remarks>
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
/// before any upsample stages, matching the encoder's symmetric placement.</para>
/// </remarks>
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

        // Sequential index of the first stage: initial conv is 0; the LSTM occupies slot 1 ONLY when present
        // (EnCodec). Mimi has no LSTM (transformer-of-codecs instead), so its stages start at 1 — the upsample
        // convtr/blocks/final conv shift down by one accordingly.
        int seqIdx = _lstm is not null ? 2 : 1;
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

        // Per stage: ELU at idx, upsample at idx+1, residual blocks at idx+2..idx+1+N. Start at 1 when there is
        // no LSTM (Mimi) so the indices match the checkpoint (see constructor note).
        int seqIdx = _lstm is not null ? 2 : 1;
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

    /// <summary>Optional per-Sequential-index activation hook for parity debugging (key = HF layer index). Not used in production paths.</summary>
    internal Action<int, Tensor>? DebugStageHook { get; set; }

    /// <summary>Forward — <paramref name="latent"/> channels-first <c>[batch, latent_dim, T_frames]</c>. Returns <c>[batch, 1, T_frames * 320]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor latent, int batch, int tFrames)
    {
        if (_initialW is null) throw new InvalidOperationException("SeaNetDecoder weights not loaded.");

        // Initial projection Conv1d(latent_dim → maxChannels, k=kernel_size). stride=1 so extra padding is 0;
        // padding is causal (left-only) for the 24 kHz model and symmetric for the non-causal 32/16 kHz codecs.
        int maxChannels = _cfg.NFilters * (1 << _stages);
        int tInit = tFrames;     // stride=1, length preserved
        Tensor x = EnCodecConvPad.PaddedConv(backend, latent, _initialW!, _initialB,
            batch, _cfg.LatentDim, tFrames, maxChannels, _cfg.KernelSize, dilation: 1, _cfg.Causal, _cfg.PadMode);

        int t = tInit;
        int dim = maxChannels;
        DebugStageHook?.Invoke(0, x);

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
        DebugStageHook?.Invoke(1, x);

        int dbgIdx = _lstm is not null ? 2 : 1;
        for (int i = 0; i < _stages; i++)
        {
            dbgIdx++;   // ELU index
            int convtrDbg = dbgIdx;
            Tensor activated = new(x.Shape, DType.F32);
            backend.Elu(activated, x, _cfg.EluAlpha);
            x.Dispose();
            x = activated;

            // ConvTranspose1d(dim → dim/2, k=ratio*2, stride=ratio). padding_total = K - stride.
            // Causal: trim all from the right (padRight = padTotal, padLeft = 0). Non-causal: symmetric trim
            // (padRight = padTotal/2, padLeft = padTotal - padRight). Matches EncodecConvTranspose1d.forward.
            int ratio = _ratios[i];
            int kUp = ratio * 2;
            int padTotalUp = kUp - ratio;
            int padRightUp = _cfg.Causal ? padTotalUp : padTotalUp / 2;
            int padLeftUp = padTotalUp - padRightUp;
            int dimOut = dim / 2;
            int tUp = (t - 1) * ratio + kUp - padLeftUp - padRightUp;  // == t*ratio
            Tensor up = new(new TensorShape(batch, dimOut, tUp), DType.F32);
            backend.ConvTranspose1d(up, x, _upsampleW[i]!, _upsampleB[i],
                stride: ratio, padLeft: padLeftUp, padRight: padRightUp, dilation: 1, groups: 1);
            x.Dispose();
            x = up;
            t = tUp;
            dim = dimOut;
            DebugStageHook?.Invoke(convtrDbg, x);
            dbgIdx++;   // convtr index consumed

            foreach (SeaNetBlock block in _stageBlocks[i])
            {
                Tensor next = block.Forward(backend, x, batch, t);
                x.Dispose();
                x = next;
            }
            DebugStageHook?.Invoke(dbgIdx, x);
            dbgIdx++;   // resblock index consumed
        }

        Tensor activatedFinal = new(x.Shape, DType.F32);
        backend.Elu(activatedFinal, x, _cfg.EluAlpha);
        x.Dispose();

        // Final Conv1d(n_filters → channels=1, k=last_kernel_size). stride=1 → length preserved.
        Tensor pcm = EnCodecConvPad.PaddedConv(backend, activatedFinal, _finalW!, _finalB,
            batch, _cfg.NFilters, t, _cfg.Channels, _cfg.LastKernelSize, dilation: 1, _cfg.Causal, _cfg.PadMode);
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
        // Accept both weight-normed (weight_g/weight_v) and pre-composed (weight) checkpoints; the Kyutai DSM
        // Mimi (tokenizer-e351c8d8) ships composed conv weights.
        if (w.TryGetValue($"{prefix}.weight_g", out Tensor? g) && w.TryGetValue($"{prefix}.weight_v", out Tensor? v))
            return WeightNormFusion.Fuse(WhisperOps.EnsureF32(g), WhisperOps.EnsureF32(v));
        return WhisperOps.EnsureF32(w[$"{prefix}.weight"]);
    }

    /// <summary>Fuses a weight-normed <c>ConvTranspose1d</c> weight, accounting for its transposed axis convention.</summary>
    /// <remarks>ConvTranspose1d weight has shape <c>[C_in, C_out, K]</c>. WeightNormFusion
    /// computes the per-out-channel norm along axis 0 of the input, but for transpose
    /// conv the "out" dim is axis 1, not axis 0. We need to fuse along axis 1 → easiest
    /// is to call the generic Fuse with the trailing-axis assumption swapped: weight_g
    /// for ConvTranspose1d has shape <c>[1, C_out, 1]</c> (per-output-channel scale on
    /// the middle axis), and weight_v has the same shape as the final weight. The
    /// L2-norm direction is along axis 0 and 2 (treating axis 1 = C_out as the "kept"
    /// dimension).</remarks>
    private static Tensor LoadFusedConvTransposeWeight(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        if (w.TryGetValue($"{prefix}.weight_g", out Tensor? g) && w.TryGetValue($"{prefix}.weight_v", out Tensor? v))
            return WeightNormFusionT.Fuse(WhisperOps.EnsureF32(g), WhisperOps.EnsureF32(v));
        return WhisperOps.EnsureF32(w[$"{prefix}.weight"]);
    }

    private static int GetExtraRightPadding(int tIn, int kernel, int stride, int padTotal)
    {
        float nFrames = ((float)tIn - kernel + padTotal) / stride + 1f;
        int idealLength = ((int)MathF.Ceiling(nFrames) - 1) * stride + (kernel - padTotal);
        return Math.Max(0, idealLength - tIn);
    }
}

/// <summary>Shared stride-1 conv helper for the EnCodec SEANet decoder + residual block, reproducing <c>EncodecConv1d.forward</c>: split the total padding into (left, right) per the causal flag, pad the channels-first input with zeros (<c>"constant"</c>) or edge reflection (<c>"reflect"</c>, the published-checkpoint default, using PyTorch semantics that exclude the edge sample), then run the backend conv with no further padding.</summary>
internal static unsafe class EnCodecConvPad
{
    public static (int Left, int Right) Split(int padTotal, bool causal)
    {
        if (causal) return (padTotal, 0);
        int right = padTotal / 2;
        return (padTotal - right, right);
    }

    /// <summary>Stride-1, dilation-aware conv with EnCodec padding. <paramref name="x"/> is <c>[B, C_in, T]</c>; output is <c>[B, C_out, T]</c> (length preserved). Caller owns the result.</summary>
    public static Tensor PaddedConv(IBackend backend, Tensor x, Tensor weight, Tensor? bias,
        int batch, int cIn, int t, int cOut, int kernel, int dilation, bool causal, string padMode, float eluUnused = 0)
    {
        int padTotal = (kernel - 1) * dilation;
        (int padLeft, int padRight) = Split(padTotal, causal);
        bool reflect = padMode == "reflect" && padTotal > 0 && t > 1;
        if (!reflect)
        {
            Tensor outZ = new(new TensorShape(batch, cOut, t), DType.F32);
            backend.Conv1d(outZ, x, weight, bias, stride: 1, padLeft: padLeft, padRight: padRight, dilation: dilation, groups: 1);
            return outZ;
        }
        // Manual reflect pad → conv with zero pad. (PyTorch reflect clamps pad < T; EnCodec never exceeds it here.)
        int tp = t + padLeft + padRight;
        Tensor padded = new(new TensorShape(batch, cIn, tp), DType.F32);
        float* sp = (float*)x.DataPointer;
        float* dp = (float*)padded.DataPointer;
        for (int b = 0; b < batch; b++)
            for (int c = 0; c < cIn; c++)
            {
                float* src = sp + ((long)b * cIn + c) * t;
                float* dst = dp + ((long)b * cIn + c) * tp;
                for (int i = 0; i < padLeft; i++) dst[i] = src[ReflectIndex(padLeft - i, t)];
                for (int i = 0; i < t; i++) dst[padLeft + i] = src[i];
                for (int i = 0; i < padRight; i++) dst[padLeft + t + i] = src[ReflectIndex(t - 2 - i, t)];
            }
        Tensor outR = new(new TensorShape(batch, cOut, t), DType.F32);
        backend.Conv1d(outR, padded, weight, bias, stride: 1, padLeft: 0, padRight: 0, dilation: dilation, groups: 1);
        padded.Dispose();
        return outR;
    }

    private static int ReflectIndex(int idx, int t)
    {
        // Mirror into [0, t-1] without repeating the edge (PyTorch 'reflect').
        if (t == 1) return 0;
        int period = 2 * (t - 1);
        int m = ((idx % period) + period) % period;
        return m < t ? m : period - m;
    }
}

/// <summary>WeightNormFusion variant for PyTorch's <c>ConvTranspose1d</c>, whose <c>weight_norm</c> is normalized per input channel rather than per output channel.</summary>
/// <remarks>
/// The transpose-conv weight has layout <c>[C_in, C_out, K]</c>. Both Meta AudioCraft and HF
/// <c>transformers</c> apply <c>nn.utils.weight_norm</c> with the DEFAULT <c>dim=0</c>, so the
/// norm is taken per <b>input</b> channel (axis 0) over the (C_out, K) slice — NOT per output
/// channel. The companion <c>weight_g</c> therefore has shape <c>[C_in, 1, 1]</c> (one scalar
/// per input channel).
///
/// <para>Fused weight is computed as
/// <c>w[ic, oc, k] = g[ic] * v[ic, oc, k] / ||v[ic, :, :]||_2</c>.</para>
/// </remarks>
public static unsafe class WeightNormFusionT
{
    public static Tensor Fuse(Tensor weightG, Tensor weightV)
    {
        if (weightV.Shape.Rank != 3) throw new ArgumentException($"WeightNormFusionT expects rank-3 weightV [C_in, C_out, K], got {weightV.Shape}.");
        int cIn = (int)weightV.Shape[0];
        int cOut = (int)weightV.Shape[1];
        int kernel = (int)weightV.Shape[2];
        if (weightG.ElementCount != cIn)
            throw new ArgumentException($"WeightNormFusionT expects weightG with {cIn} elements (C_in, weight_norm dim=0), got {weightG.ElementCount}.");

        Tensor fused = new(weightV.Shape, DType.F32);
        float* vp = (float*)weightV.DataPointer;
        float* gp = (float*)weightG.DataPointer;
        float* fp = (float*)fused.DataPointer;

        // Compute per-in-channel norm over (cOut, kernel) and apply (weight_norm dim=0).
        for (int ic = 0; ic < cIn; ic++)
        {
            double sumSq = 0d;
            int baseIdx = ic * cOut * kernel;
            for (int j = 0; j < cOut * kernel; j++)
            {
                float v = vp[baseIdx + j];
                sumSq += (double)v * v;
            }
            float norm = MathF.Sqrt((float)sumSq);
            float scale = gp[ic] * (norm > 0f ? 1f / norm : 0f);
            for (int j = 0; j < cOut * kernel; j++)
                fp[baseIdx + j] = vp[baseIdx + j] * scale;
        }
        return fused;
    }
}
