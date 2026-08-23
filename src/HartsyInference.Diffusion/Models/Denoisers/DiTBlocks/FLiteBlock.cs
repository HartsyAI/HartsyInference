using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>F-Lite single transformer block. Three sub-paths in sequence: self-attention → cross-attention → MLP. Each sub-path uses an AdaLN-Zero modulation (shift/scale before, gate after). Modulation produces 9 outputs from a single Linear(SiLU(temb)) — chunked into <c>(shift_sa, scale_sa, gate_sa, shift_ca, scale_ca, gate_ca, shift_mlp, scale_mlp, gate_mlp)</c>.
///
/// <para><b>Norms</b>: F-Lite's RMSNorm has no learned scale when <see cref="FLiteConfig.TrainBiasAndRms"/> is false (i.e. on the public 10B). The norm is then "RMSNorm without affine" — pure normalization. We honor this by treating the norm scale as null and skipping the multiplication.</para>
///
/// <para><b>MLP</b>: Linear(hidden → mlp_dim) + GELU + Linear(mlp_dim → hidden). Biases on these two linears follow <see cref="FLiteConfig.TrainBiasAndRms"/>.</para></summary>
public sealed unsafe class FLiteBlock
{
    private readonly FLiteConfig _config;
    private readonly FLiteAttention _selfAttn;
    private readonly FLiteAttention _crossAttn;

    private Tensor? _norm1Weight, _norm2Weight, _norm3Weight;
    private Tensor? _mlpUpWeight, _mlpUpBias;
    private Tensor? _mlpDownWeight, _mlpDownBias;
    private Tensor? _adaLNWeight, _adaLNBias;

    public FLiteBlock(FLiteConfig config)
    {
        _config = config;
        _selfAttn = FLiteAttention.SelfAttn(config);
        _crossAttn = FLiteAttention.CrossAttn(config);
    }

    /// <summary>Loads weights for one block. <paramref name="prefix"/> = "blocks.{i}".</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _selfAttn.LoadWeights(weights, $"{prefix}.self_attn");
        _crossAttn.LoadWeights(weights, $"{prefix}.cross_attn");

        if (_config.TrainBiasAndRms)
        {
            weights.TryGetValue($"{prefix}.norm1.weight", out _norm1Weight);
            weights.TryGetValue($"{prefix}.norm2.weight", out _norm2Weight);
            weights.TryGetValue($"{prefix}.norm3.weight", out _norm3Weight);
        }

        _mlpUpWeight = weights[$"{prefix}.mlp.0.weight"];
        weights.TryGetValue($"{prefix}.mlp.0.bias", out _mlpUpBias);
        _mlpDownWeight = weights[$"{prefix}.mlp.2.weight"];
        weights.TryGetValue($"{prefix}.mlp.2.bias", out _mlpDownBias);

        _adaLNWeight = weights[$"{prefix}.adaLN_modulation.1.weight"];
        weights.TryGetValue($"{prefix}.adaLN_modulation.1.bias", out _adaLNBias);
    }

    /// <summary>Yields all weight tensors for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor w in _selfAttn.EnumerateWeights()) yield return w;
        foreach (Tensor w in _crossAttn.EnumerateWeights()) yield return w;
        if (_norm1Weight is not null) yield return _norm1Weight;
        if (_norm2Weight is not null) yield return _norm2Weight;
        if (_norm3Weight is not null) yield return _norm3Weight;
        if (_mlpUpWeight is not null) yield return _mlpUpWeight;
        if (_mlpUpBias is not null) yield return _mlpUpBias;
        if (_mlpDownWeight is not null) yield return _mlpDownWeight;
        if (_mlpDownBias is not null) yield return _mlpDownBias;
        if (_adaLNWeight is not null) yield return _adaLNWeight;
        if (_adaLNBias is not null) yield return _adaLNBias;
    }

    /// <summary>Block forward. Inputs: <paramref name="x"/> [B, S, hidden] (image tokens including 16 register tokens); <paramref name="context"/> [B, S_ctx, cross_input] (T5); <paramref name="temb"/> [B, hidden] (timestep embedding); <paramref name="vPrev"/> [B, H, S, headDim] from block 0 or null for block 0; <paramref name="cosRope"/>/<paramref name="sinRope"/> [1, 1, S, headDim/2] precomputed RoPE tables. Returns (x_out, v) where v is the self-attn V (caller captures from block 0).</summary>
    public (Tensor x, Tensor v) Forward(IBackend backend, Tensor x, Tensor context, Tensor temb,
        Tensor? vPrev, Tensor? cosRope, Tensor? sinRope)
    {
        int batch = (int)x.Shape[0];
        int seqLen = (int)x.Shape[1];
        int hidden = _config.HiddenSize;

        Tensor[] mod = ComputeAdaLN(backend, temb);

        Tensor saIn = ModulateRmsNorm(backend, x, _norm1Weight, mod[0], mod[1], batch, seqLen, hidden);
        (Tensor saOut, Tensor v) = _selfAttn.ForwardSelf(backend, saIn, vPrev, cosRope, sinRope);
        saIn.Dispose();
        Tensor xAfterSa = GatedResidual(backend, x, saOut, mod[2], batch, seqLen, hidden);
        saOut.Dispose();

        Tensor caIn = ModulateRmsNorm(backend, xAfterSa, _norm2Weight, mod[3], mod[4], batch, seqLen, hidden);
        Tensor caOut = _crossAttn.ForwardCross(backend, caIn, context);
        caIn.Dispose();
        Tensor xAfterCa = GatedResidual(backend, xAfterSa, caOut, mod[5], batch, seqLen, hidden);
        xAfterSa.Dispose();
        caOut.Dispose();

        Tensor mlpIn = ModulateRmsNorm(backend, xAfterCa, _norm3Weight, mod[6], mod[7], batch, seqLen, hidden);
        Tensor mlpOut = ApplyMlp(backend, mlpIn, batch, seqLen, hidden);
        mlpIn.Dispose();
        Tensor xFinal = GatedResidual(backend, xAfterCa, mlpOut, mod[8], batch, seqLen, hidden);
        xAfterCa.Dispose();
        mlpOut.Dispose();

        for (int i = 0; i < mod.Length; i++) mod[i].Dispose();

        return (xFinal, v);
    }

    private Tensor[] ComputeAdaLN(IBackend backend, Tensor temb)
    {
        if (_adaLNWeight is null) throw new InvalidOperationException("AdaLN weight not loaded.");
        int batch = (int)temb.Shape[0];
        int hidden = _config.HiddenSize;

        Tensor activated = new Tensor(new TensorShape(batch, hidden), DType.F32);
        backend.Silu(activated, temb);

        Tensor projected = new Tensor(new TensorShape(batch, 9 * hidden), DType.F32);
        backend.Linear(projected, activated, _adaLNWeight, _adaLNBias);
        activated.Dispose();

        // Device chunk: 9 × [B, hidden] slices along the last dim — no host round-trip (the old
        // DataPointer split was both the per-block stall and the async-pool AV hazard).
        Tensor[] result = new Tensor[9];
        for (int i = 0; i < 9; i++)
        {
            Tensor part = new Tensor(new TensorShape(batch, hidden), DType.F32);
            backend.SliceLastDim(part, projected, i * hidden);
            result[i] = part;
        }
        projected.Dispose();
        return result;
    }

    private Tensor ModulateRmsNorm(IBackend backend, Tensor x, Tensor? scaleWeight,
        Tensor shift, Tensor scale, int batch, int seqLen, int hidden)
    {
        const float Eps = 1e-6f;
        Tensor normed = new Tensor(new TensorShape(batch, seqLen, hidden), DType.F32);
        backend.RmsNorm(normed, x, scaleWeight ?? OnesHidden(backend, hidden), Eps);

        Tensor scalePlus1 = new Tensor(scale.Shape, DType.F32);
        backend.AddScalar(scalePlus1, scale, 1.0f);
        Tensor result = new Tensor(new TensorShape(batch, seqLen, hidden), DType.F32);
        backend.AffineBroadcastLastDim(result, normed, scalePlus1, shift);
        scalePlus1.Dispose();
        normed.Dispose();
        return result;
    }

    // Constant all-ones RMSNorm scale for the public 10B's affine-free norms (TrainBiasAndRms=false).
    private Tensor? _onesHidden;
    private Tensor OnesHidden(IBackend backend, int hidden)
    {
        if (_onesHidden is null)
        {
            _onesHidden = new Tensor(new TensorShape(hidden), DType.F32);
            backend.Fill(_onesHidden, 1.0f);
        }
        return _onesHidden;
    }

    private static Tensor GatedResidual(IBackend backend, Tensor residual, Tensor delta, Tensor gate,
        int batch, int seqLen, int hidden)
    {
        Tensor result = new Tensor(new TensorShape(batch, seqLen, hidden), DType.F32);
        backend.GatedResidualLastDim(result, residual, delta, gate);
        return result;
    }

    private Tensor ApplyMlp(IBackend backend, Tensor x, int batch, int seqLen, int hidden)
    {
        if (_mlpUpWeight is null || _mlpDownWeight is null) throw new InvalidOperationException("MLP weights not loaded.");
        int mlpDim = _config.MlpDim;
        Tensor up = new Tensor(new TensorShape(batch, seqLen, mlpDim), DType.F32);
        backend.Linear(up, x, _mlpUpWeight, _mlpUpBias);
        Tensor activated = new Tensor(up.Shape, DType.F32);
        backend.Gelu(activated, up);
        up.Dispose();
        Tensor result = new Tensor(new TensorShape(batch, seqLen, hidden), DType.F32);
        backend.Linear(result, activated, _mlpDownWeight, _mlpDownBias);
        activated.Dispose();
        return result;
    }
}
