using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Music;

/// <summary>MiniMax-H3 audio VAE encoder: a DAC-lineage convolutional stack (<c>conv(1→64,k7)</c> → 5 stages of
/// three dilated Snake residual units plus a strided downsample, rates 2/4/4/5/5 → Snake → <c>conv(→2048,k3)</c>)
/// feeding a causal-attention posterior head that pools 2048 channels down to the DiT's 32, then <c>mean_proj</c> and
/// the stored latent statistics. Supplies the reference-audio latents ref2va conditions on; the decoder half is
/// <see cref="MiniMaxH3AudioVaeDecoder"/>.</summary>
///
/// <remarks>Stereo is encoded as two independent mono streams — the latent's channel axis is the only place the two
/// meet — matching the reference's <c>reshape(b·s, 1, L)</c>. <c>logs_proj</c> ships in the checkpoint but inference
/// takes the posterior mean and never samples, so it is deliberately not loaded.</remarks>
public sealed unsafe class MiniMaxH3AudioVaeEncoder : IDisposable
{
    private readonly MiniMaxH3AudioVaeConfig _config;
    private readonly List<Tensor> _owned = [];

    private Tensor? _stemW, _stemB;
    private Stage[] _stages = [];
    private Tensor? _finalAlpha, _finalW, _finalB;
    private Tensor? _norm1W, _norm1B, _norm2W, _norm2B, _norm3W, _norm3B;
    private Tensor? _qkvW, _qkvBias, _attnProjW, _attnProjB, _projW, _projB;
    private Tensor? _mlpNormW, _mlpNormB, _w0W, _w0B, _w1W, _w1B, _w2W, _w2B;
    private Tensor? _meanProjW, _meanProjB;

    private sealed class ResidualUnit
    {
        public required Tensor Alpha1 { get; init; }
        public required Tensor Conv1W { get; init; }
        public required Tensor Conv1B { get; init; }
        public required Tensor Alpha2 { get; init; }
        public required Tensor Conv2W { get; init; }
        public required Tensor Conv2B { get; init; }
        public required int Dilation { get; init; }
    }

    private sealed class Stage
    {
        public required ResidualUnit[] Units { get; init; }
        public required Tensor Alpha { get; init; }
        public required Tensor DownW { get; init; }
        public required Tensor DownB { get; init; }
        public required int Stride { get; init; }
    }

    public MiniMaxH3AudioVaeEncoder(MiniMaxH3AudioVaeConfig? config = null)
    {
        _config = config ?? new MiniMaxH3AudioVaeConfig();
    }

    public MiniMaxH3AudioVaeConfig Config => _config;

    /// <summary>Waveform samples one latent frame spans.</summary>
    public int SamplesPerLatentFrame => _config.EncoderHopLength;

    /// <summary>Loads <c>encoder.*</c>, <c>pre_block.*</c> and <c>mean_proj.*</c>, fusing weight-norm at load.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        ArgumentNullException.ThrowIfNull(w);
        _stemW = Fuse(w, "encoder.block.0");
        _stemB = w["encoder.block.0.bias"];

        _stages = new Stage[_config.EncoderRates.Length];
        for (int i = 0; i < _stages.Length; i++)
        {
            string p = $"encoder.block.{i + 1}";
            int[] dilations = [1, 3, 9];
            ResidualUnit[] units = new ResidualUnit[dilations.Length];
            for (int u = 0; u < units.Length; u++)
            {
                string up = $"{p}.block.{u}.block";
                units[u] = new ResidualUnit
                {
                    Alpha1 = w[$"{up}.0.alpha"],
                    Conv1W = Fuse(w, $"{up}.1"),
                    Conv1B = w[$"{up}.1.bias"],
                    Alpha2 = w[$"{up}.2.alpha"],
                    Conv2W = Fuse(w, $"{up}.3"),
                    Conv2B = w[$"{up}.3.bias"],
                    Dilation = dilations[u],
                };
            }
            _stages[i] = new Stage
            {
                Units = units,
                Alpha = w[$"{p}.block.3.alpha"],
                DownW = Fuse(w, $"{p}.block.4"),
                DownB = w[$"{p}.block.4.bias"],
                Stride = _config.EncoderRates[i],
            };
        }

        int tail = _stages.Length + 1;
        _finalAlpha = w[$"encoder.block.{tail}.alpha"];
        _finalW = Fuse(w, $"encoder.block.{tail + 1}");
        _finalB = w[$"encoder.block.{tail + 1}.bias"];

        _norm1W = w["pre_block.norm1.weight"]; _norm1B = w["pre_block.norm1.bias"];
        _norm2W = w["pre_block.norm2.weight"]; _norm2B = w["pre_block.norm2.bias"];
        _norm3W = w["pre_block.norm3.weight"]; _norm3B = w["pre_block.norm3.bias"];
        _qkvW = w["pre_block.attn.qkv.weight"];
        // The stored bias is split three ways with a zero K half — concatenating once at load keeps the projection
        // a single GEMM instead of three.
        _qkvBias = ConcatBias(w["pre_block.attn.q_bias"], w["pre_block.attn.zero_k_bias"], w["pre_block.attn.v_bias"]);
        _owned.Add(_qkvBias);
        _attnProjW = w["pre_block.attn.proj.weight"]; _attnProjB = w["pre_block.attn.proj.bias"];
        _projW = w["pre_block.proj.weight"]; _projB = w["pre_block.proj.bias"];
        _mlpNormW = w["pre_block.mlp.norm.weight"]; _mlpNormB = w["pre_block.mlp.norm.bias"];
        _w0W = w["pre_block.mlp.w0.weight"]; _w0B = w["pre_block.mlp.w0.bias"];
        _w1W = w["pre_block.mlp.w1.weight"]; _w1B = w["pre_block.mlp.w1.bias"];
        _w2W = w["pre_block.mlp.w2.weight"]; _w2B = w["pre_block.mlp.w2.bias"];
        _meanProjW = w["mean_proj.weight"]; _meanProjB = w["mean_proj.bias"];
    }

    /// <summary>True when <paramref name="weights"/> carries the encoder half.</summary>
    public static bool Matches(IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        return (weights.ContainsKey("encoder.block.0.weight_v") || weights.ContainsKey("encoder.block.0.weight"))
            && weights.ContainsKey("pre_block.attn.qkv.weight") && weights.ContainsKey("mean_proj.weight");
    }

    /// <summary>Encodes a waveform <c>[1, S, L]</c> in [-1, 1] at the VAE's sample rate to normalized latents
    /// <c>[1, 32, S, T]</c> — the layout <see cref="MiniMaxH3Latents"/> packs into reference audio rows. The tail is
    /// zero-padded up to a whole latent frame.</summary>
    public Tensor Encode(IBackend backend, Tensor waveform)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(waveform);
        if (_stemW is null)
        {
            throw new InvalidOperationException("MiniMaxH3AudioVaeEncoder.LoadWeights must run before Encode.");
        }
        if (waveform.Shape.Rank != 3 || waveform.Shape[0] != 1)
        {
            throw new ArgumentException($"expected a batch-1 [1, S, L] waveform, got {waveform.Shape}.", nameof(waveform));
        }
        int channels = (int)waveform.Shape[1], length = (int)waveform.Shape[2];
        int hop = _config.EncoderHopLength;
        int padded = (length + hop - 1) / hop * hop;

        // Each stereo channel is an independent mono stream, so they ride the stack as the batch.
        Tensor x = new Tensor(new TensorShape(channels, 1, padded), DType.F32);
        float* src = (float*)waveform.DataPointer;
        float* dst = (float*)x.DataPointer;
        for (int s = 0; s < channels; s++)
        {
            Buffer.MemoryCopy(src + (long)s * length, dst + (long)s * padded, (long)length * 4, (long)length * 4);
        }

        try
        {
            Tensor cur = Conv(backend, x, _stemW!, _stemB, stride: 1, pad: 3, dilation: 1);
            x.Dispose();
            x = cur;
            foreach (Stage stage in _stages)
            {
                foreach (ResidualUnit unit in stage.Units)
                {
                    cur = ResidualForward(backend, x, unit);
                    x.Dispose();
                    x = cur;
                }
                cur = Activate(backend, x, stage.Alpha);
                x.Dispose();
                x = cur;
                cur = Conv(backend, x, stage.DownW, stage.DownB, stage.Stride,
                    (stage.Stride + 1) / 2, dilation: 1);
                x.Dispose();
                x = cur;
            }
            cur = Activate(backend, x, _finalAlpha!);
            x.Dispose();
            x = cur;
            cur = Conv(backend, x, _finalW!, _finalB, stride: 1, pad: 1, dilation: 1);
            x.Dispose();
            x = cur;

            Tensor pooled = PosteriorHead(backend, x, channels);
            x.Dispose();
            x = pooled;

            Tensor projected = Conv(backend, x, _meanProjW!, _meanProjB, stride: 1, pad: 0, dilation: 1);
            x.Dispose();
            return NormalizeAndInterleave(projected, channels);
        }
        catch
        {
            x.Dispose();
            throw;
        }
    }

    /// <summary>Snake, dilated conv, Snake, kernel-1 conv, plus the input. The reference crops the residual when the
    /// convolutions shorten the sequence; H3's padding keeps the length, so a mismatch means a bad stride table.</summary>
    private Tensor ResidualForward(IBackend backend, Tensor x, ResidualUnit unit)
    {
        Tensor a1 = Activate(backend, x, unit.Alpha1);
        Tensor c1;
        try
        {
            c1 = Conv(backend, a1, unit.Conv1W, unit.Conv1B, stride: 1, pad: 3 * unit.Dilation, dilation: unit.Dilation);
        }
        finally
        {
            a1.Dispose();
        }
        Tensor a2 = Activate(backend, c1, unit.Alpha2);
        c1.Dispose();
        Tensor c2;
        try
        {
            c2 = Conv(backend, a2, unit.Conv2W, unit.Conv2B, stride: 1, pad: 0, dilation: 1);
        }
        finally
        {
            a2.Dispose();
        }
        if (c2.Shape[2] != x.Shape[2])
        {
            c2.Dispose();
            throw new InvalidOperationException(
                $"MiniMax-H3 audio residual unit changed length {x.Shape[2]} -> {c2.Shape[2]}.");
        }
        try
        {
            Tensor sum = new Tensor(c2.Shape, DType.F32);
            backend.Add(sum, c2, x);
            return sum;
        }
        finally
        {
            c2.Dispose();
        }
    }

    /// <summary>The causal-attention posterior head: it pools the 2048-wide convolutional features down to the DiT's
    /// 32 latent channels, so it is what sets the latent's scale, not just its width.</summary>
    private Tensor PosteriorHead(IBackend backend, Tensor x, int batch)
    {
        int width = (int)x.Shape[1], frames = (int)x.Shape[2];
        int latent = (int)_meanProjW!.Shape[0], heads = _config.AttnHeads, headDim = width / heads;
        using Tensor tokens = ChannelsToTokens(x);

        using Tensor normed1 = new Tensor(tokens.Shape, DType.F32);
        backend.LayerNorm(normed1, tokens, _norm1W!, _norm1B!, _config.NormEps);
        using Tensor qkv = new Tensor(new TensorShape((long)batch * frames, 3L * width), DType.F32);
        backend.Linear(qkv, normed1, _qkvW!, _qkvBias);

        using Tensor q = new Tensor(new TensorShape(batch, heads, frames, headDim), DType.F32);
        using Tensor k = new Tensor(new TensorShape(batch, heads, frames, headDim), DType.F32);
        using Tensor v = new Tensor(new TensorShape(batch, heads, frames, headDim), DType.F32);
        SplitHeads(qkv, q, k, v, batch, frames, heads, headDim);
        using Tensor mask = CausalMask(frames);
        using Tensor attn = new Tensor(new TensorShape(batch, heads, frames, headDim), DType.F32);
        backend.ScaledDotProductAttention(attn, q, k, v, mask, 1f / MathF.Sqrt(headDim));

        // Mean over heads, then an adaptive average pool down to the latent width — this is the reference's whole
        // channel reduction, not a projection.
        using Tensor reduced = MeanHeadsAndPool(attn, batch, heads, frames, headDim, latent);
        using Tensor attnOut = new Tensor(new TensorShape((long)batch * frames, latent), DType.F32);
        backend.Linear(attnOut, reduced, _attnProjW!, _attnProjB);

        using Tensor normed3 = new Tensor(tokens.Shape, DType.F32);
        backend.LayerNorm(normed3, tokens, _norm3W!, _norm3B!, _config.NormEps);
        Tensor stream = new Tensor(new TensorShape((long)batch * frames, latent), DType.F32);
        try
        {
            backend.Linear(stream, normed3, _projW!, _projB);
            using (Tensor sum = new Tensor(stream.Shape, DType.F32))
            {
                backend.Add(sum, stream, attnOut);
                Copy(sum, stream);
            }

            using Tensor normed2 = new Tensor(stream.Shape, DType.F32);
            backend.LayerNorm(normed2, stream, _norm2W!, _norm2B!, _config.NormEps);
            using Tensor mlp = GatedMlp(backend, normed2, latent);
            using (Tensor sum = new Tensor(stream.Shape, DType.F32))
            {
                backend.Add(sum, stream, mlp);
                Copy(sum, stream);
            }
            return TokensToChannels(stream, batch, frames, latent);
        }
        finally
        {
            stream.Dispose();
        }
    }

    /// <summary>GeGLU: <c>w2(gelu(w0(norm(x))) * w1(norm(x)))</c>. The head's own norm is on top of the block's.</summary>
    private Tensor GatedMlp(IBackend backend, Tensor x, int latent)
    {
        long rows = x.Shape[0];
        int hidden = latent * _config.MlpRatio;
        using Tensor normed = new Tensor(x.Shape, DType.F32);
        backend.LayerNorm(normed, x, _mlpNormW!, _mlpNormB!, _config.NormEps);
        using Tensor gate = new Tensor(new TensorShape(rows, hidden), DType.F32);
        backend.Linear(gate, normed, _w0W!, _w0B);
        using Tensor activated = new Tensor(gate.Shape, DType.F32);
        backend.Gelu(activated, gate);
        using Tensor up = new Tensor(new TensorShape(rows, hidden), DType.F32);
        backend.Linear(up, normed, _w1W!, _w1B);
        using Tensor gated = new Tensor(gate.Shape, DType.F32);
        backend.Mul(gated, activated, up);
        Tensor outT = new Tensor(new TensorShape(rows, latent), DType.F32);
        backend.Linear(outT, gated, _w2W!, _w2B);
        return outT;
    }

    private Tensor Activate(IBackend backend, Tensor x, Tensor alpha)
    {
        Tensor outT = new Tensor(x.Shape, DType.F32);
        backend.Snake(outT, x, alpha, null);
        return outT;
    }

    private static Tensor Conv(IBackend backend, Tensor x, Tensor weight, Tensor? bias, int stride, int pad, int dilation)
    {
        int inLen = (int)x.Shape[2], k = (int)weight.Shape[2];
        int outLen = (inLen + 2 * pad - dilation * (k - 1) - 1) / stride + 1;
        Tensor outT = new Tensor(new TensorShape(x.Shape[0], weight.Shape[0], outLen), DType.F32);
        backend.Conv1d(outT, x, weight, bias, stride, pad, pad, dilation, 1);
        return outT;
    }

    /// <summary><c>[B, C, T]</c> to the <c>[B·T, C]</c> row-major form the linear layers consume.</summary>
    private static Tensor ChannelsToTokens(Tensor x)
    {
        int b = (int)x.Shape[0], c = (int)x.Shape[1], t = (int)x.Shape[2];
        Tensor outT = new Tensor(new TensorShape((long)b * t, c), DType.F32);
        float* src = (float*)x.DataPointer;
        float* dst = (float*)outT.DataPointer;
        for (int bi = 0; bi < b; bi++)
        {
            for (int ti = 0; ti < t; ti++)
            {
                for (int ci = 0; ci < c; ci++)
                {
                    dst[((long)bi * t + ti) * c + ci] = src[((long)bi * c + ci) * t + ti];
                }
            }
        }
        return outT;
    }

    private static Tensor TokensToChannels(Tensor x, int batch, int frames, int channels)
    {
        Tensor outT = new Tensor(new TensorShape(batch, channels, frames), DType.F32);
        float* src = (float*)x.DataPointer;
        float* dst = (float*)outT.DataPointer;
        for (int bi = 0; bi < batch; bi++)
        {
            for (int ci = 0; ci < channels; ci++)
            {
                for (int ti = 0; ti < frames; ti++)
                {
                    dst[((long)bi * channels + ci) * frames + ti] = src[((long)bi * frames + ti) * channels + ci];
                }
            }
        }
        return outT;
    }

    private static void SplitHeads(Tensor qkv, Tensor q, Tensor k, Tensor v, int batch, int frames, int heads, int headDim)
    {
        float* src = (float*)qkv.DataPointer;
        float* qp = (float*)q.DataPointer, kp = (float*)k.DataPointer, vp = (float*)v.DataPointer;
        int width = heads * headDim;
        for (int b = 0; b < batch; b++)
        {
            for (int t = 0; t < frames; t++)
            {
                float* row = src + ((long)b * frames + t) * 3 * width;
                for (int h = 0; h < heads; h++)
                {
                    long dstOff = (((long)b * heads + h) * frames + t) * headDim;
                    for (int d = 0; d < headDim; d++)
                    {
                        qp[dstOff + d] = row[h * headDim + d];
                        kp[dstOff + d] = row[width + h * headDim + d];
                        vp[dstOff + d] = row[2 * width + h * headDim + d];
                    }
                }
            }
        }
    }

    private static Tensor MeanHeadsAndPool(Tensor attn, int batch, int heads, int frames, int headDim, int latent)
    {
        Tensor outT = new Tensor(new TensorShape((long)batch * frames, latent), DType.F32);
        float* src = (float*)attn.DataPointer;
        float* dst = (float*)outT.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            for (int t = 0; t < frames; t++)
            {
                for (int l = 0; l < latent; l++)
                {
                    int start = l * headDim / latent, stop = (l + 1) * headDim / latent;
                    double sum = 0;
                    for (int d = start; d < stop; d++)
                    {
                        double acrossHeads = 0;
                        for (int h = 0; h < heads; h++)
                        {
                            acrossHeads += src[(((long)b * heads + h) * frames + t) * headDim + d];
                        }
                        sum += acrossHeads / heads;
                    }
                    dst[((long)b * frames + t) * latent + l] = (float)(sum / (stop - start));
                }
            }
        }
        return outT;
    }

    private static Tensor CausalMask(int frames)
    {
        Tensor mask = new Tensor(new TensorShape(1, 1, frames, frames), DType.F32);
        float* p = (float*)mask.DataPointer;
        for (int i = 0; i < frames; i++)
        {
            for (int j = 0; j < frames; j++)
            {
                p[(long)i * frames + j] = j <= i ? 0f : float.NegativeInfinity;
            }
        }
        return mask;
    }

    /// <summary>Normalizes by the stored latent statistics and lifts <c>[S, C, T]</c> to the DiT's <c>[1, C, S, T]</c>.</summary>
    private Tensor NormalizeAndInterleave(Tensor x, int channels)
    {
        try
        {
            int c = (int)x.Shape[1], t = (int)x.Shape[2];
            Tensor outT = new Tensor(new TensorShape(1, c, channels, t), DType.F32);
            float* src = (float*)x.DataPointer;
            float* dst = (float*)outT.DataPointer;
            for (int s = 0; s < channels; s++)
            {
                for (int ci = 0; ci < c; ci++)
                {
                    float mean = _config.LatentsMean[ci], std = _config.LatentsStd[ci];
                    for (int ti = 0; ti < t; ti++)
                    {
                        dst[(((long)ci * channels) + s) * t + ti] =
                            (src[((long)s * c + ci) * t + ti] - mean) / std;
                    }
                }
            }
            return outT;
        }
        finally
        {
            x.Dispose();
        }
    }

    private static void Copy(Tensor source, Tensor destination) =>
        Buffer.MemoryCopy((void*)source.DataPointer, (void*)destination.DataPointer,
            destination.ElementCount * 4, source.ElementCount * 4);

    private static Tensor ConcatBias(Tensor q, Tensor zeroK, Tensor v)
    {
        Tensor qf = AudioWeightNorm.AsF32(q, out bool ownQ);
        Tensor kf = AudioWeightNorm.AsF32(zeroK, out bool ownK);
        Tensor vf = AudioWeightNorm.AsF32(v, out bool ownV);
        try
        {
            long n = qf.ElementCount;
            Tensor outT = new Tensor(new TensorShape(3 * n), DType.F32);
            float* dst = (float*)outT.DataPointer;
            Buffer.MemoryCopy((void*)qf.DataPointer, dst, n * 4, n * 4);
            Buffer.MemoryCopy((void*)kf.DataPointer, dst + n, n * 4, n * 4);
            Buffer.MemoryCopy((void*)vf.DataPointer, dst + 2 * n, n * 4, n * 4);
            return outT;
        }
        finally
        {
            if (ownQ) { qf.Dispose(); }
            if (ownK) { kf.Dispose(); }
            if (ownV) { vf.Dispose(); }
        }
    }

    private Tensor Fuse(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        Tensor fused = AudioWeightNorm.Fuse(w, prefix, out bool allocated);
        if (allocated)
        {
            _owned.Add(fused);
        }
        return fused;
    }

    /// <summary>Every encoder tensor, for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_stemW is not null) { yield return _stemW; }
        if (_stemB is not null) { yield return _stemB; }
        foreach (Stage s in _stages)
        {
            foreach (ResidualUnit u in s.Units)
            {
                yield return u.Alpha1; yield return u.Conv1W; yield return u.Conv1B;
                yield return u.Alpha2; yield return u.Conv2W; yield return u.Conv2B;
            }
            yield return s.Alpha; yield return s.DownW; yield return s.DownB;
        }
        foreach (Tensor? t in new[] { _finalAlpha, _finalW, _finalB, _norm1W, _norm1B, _norm2W, _norm2B, _norm3W,
            _norm3B, _qkvW, _qkvBias, _attnProjW, _attnProjB, _projW, _projB, _mlpNormW, _mlpNormB, _w0W, _w0B,
            _w1W, _w1B, _w2W, _w2B, _meanProjW, _meanProjB })
        {
            if (t is not null) { yield return t; }
        }
    }

    public void Dispose()
    {
        foreach (Tensor t in _owned)
        {
            t.Dispose();
        }
        _owned.Clear();
    }
}
