using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Codecs.Mimi;

/// <summary>Mimi's transformer-of-codecs. Small causal multi-head-attention stack with
/// RoPE positional encoding, run BETWEEN the SEANet encoder and the RVQ. Adds full
/// temporal context to every frame's latent before quantization — much higher recon
/// quality than EnCodec at the same bitrate.
///
/// <para>Lightweight: 8 layers × 512 dim × 8 heads × 64 head_dim, FFN 2048. Per-token
/// math at 12.5 Hz makes the transformer a tiny fraction of total codec compute.</para>
///
/// <para>State-dict layout matches <c>kyutai/mimi</c>: per-layer Linear projections for
/// q/k/v/out, RMSNorm pre-attention and pre-FFN, two-Linear FFN (GeLU).</para></summary>
internal sealed unsafe class MimiTransformer
{
    private readonly MimiConfig _cfg;
    private readonly string _prefix;
    private readonly int _layers;
    private readonly int _heads;
    private readonly int _headDim;

    private readonly Tensor?[] _normAttnW;
    private readonly Tensor?[] _normFfnW;
    private readonly Tensor?[] _qW;
    private readonly Tensor?[] _kW;
    private readonly Tensor?[] _vW;
    private readonly Tensor?[] _oW;
    private readonly Tensor?[] _fc1W;
    private readonly Tensor?[] _fc2W;

    public MimiTransformer(MimiConfig cfg, string prefix)
    {
        _cfg = cfg;
        _prefix = prefix;
        _layers = cfg.TransformerLayers;
        _heads = cfg.TransformerHeads;
        _headDim = cfg.TransformerDim / cfg.TransformerHeads;
        _normAttnW = new Tensor?[_layers];
        _normFfnW = new Tensor?[_layers];
        _qW = new Tensor?[_layers];
        _kW = new Tensor?[_layers];
        _vW = new Tensor?[_layers];
        _oW = new Tensor?[_layers];
        _fc1W = new Tensor?[_layers];
        _fc2W = new Tensor?[_layers];
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        for (int i = 0; i < _layers; i++)
        {
            string p = $"{_prefix}.layers.{i}";
            _normAttnW[i] = WhisperOps.EnsureF32(w[$"{p}.norm_attn.weight"]);
            _normFfnW[i] = WhisperOps.EnsureF32(w[$"{p}.norm_ffn.weight"]);
            _qW[i] = WhisperOps.EnsureF32(w[$"{p}.attn.q_proj.weight"]);
            _kW[i] = WhisperOps.EnsureF32(w[$"{p}.attn.k_proj.weight"]);
            _vW[i] = WhisperOps.EnsureF32(w[$"{p}.attn.v_proj.weight"]);
            _oW[i] = WhisperOps.EnsureF32(w[$"{p}.attn.o_proj.weight"]);
            _fc1W[i] = WhisperOps.EnsureF32(w[$"{p}.ffn.fc1.weight"]);
            _fc2W[i] = WhisperOps.EnsureF32(w[$"{p}.ffn.fc2.weight"]);
        }
    }

    /// <summary>Forward — <paramref name="x"/> channels-last <c>[B, T, dim]</c>.
    /// Returns a fresh tensor same shape. Causal mask applied during attention.</summary>
    public Tensor Forward(IBackend backend, Tensor x, int batch, int t)
    {
        Tensor current = x;
        bool ownsCurrent = false;

        // Pre-build causal mask: shape [T, T], lower-triangular = 0, upper-triangular = -inf.
        Tensor causalMask = BuildCausalMask(t);

        for (int i = 0; i < _layers; i++)
        {
            // RMSNorm pre-attn.
            Tensor normed = new(current.Shape, DType.F32);
            backend.RmsNorm(normed, current, _normAttnW[i]!, 1e-5f);

            // Q, K, V projections.
            Tensor qProj = WhisperOps.ProjectLinear(backend, normed, _qW[i]!, null, batch, t, _cfg.TransformerDim, _cfg.TransformerDim);
            Tensor kProj = WhisperOps.ProjectLinear(backend, normed, _kW[i]!, null, batch, t, _cfg.TransformerDim, _cfg.TransformerDim);
            Tensor vProj = WhisperOps.ProjectLinear(backend, normed, _vW[i]!, null, batch, t, _cfg.TransformerDim, _cfg.TransformerDim);
            normed.Dispose();

            // Reshape to [B, H, T, D] for multi-head + RoPE + SDPA.
            Tensor qMh = new(new TensorShape(batch, _heads, t, _headDim), DType.F32);
            Tensor kMh = new(new TensorShape(batch, _heads, t, _headDim), DType.F32);
            Tensor vMh = new(new TensorShape(batch, _heads, t, _headDim), DType.F32);
            WhisperOps.ReshapeToMultiHead4D(qMh, qProj, batch, t, _heads, _headDim);
            WhisperOps.ReshapeToMultiHead4D(kMh, kProj, batch, t, _heads, _headDim);
            WhisperOps.ReshapeToMultiHead4D(vMh, vProj, batch, t, _heads, _headDim);
            qProj.Dispose(); kProj.Dispose(); vProj.Dispose();

            // Apply RoPE to Q and K (in-place).
            ApplyRoPE(qMh, batch, _heads, t, _headDim, _cfg.TransformerRopeTheta);
            ApplyRoPE(kMh, batch, _heads, t, _headDim, _cfg.TransformerRopeTheta);

            // Causal SDPA.
            Tensor attnOut = new(qMh.Shape, DType.F32);
            backend.ScaledDotProductAttention(attnOut, qMh, kMh, vMh, causalMask, 1f / MathF.Sqrt(_headDim));
            qMh.Dispose(); kMh.Dispose(); vMh.Dispose();

            // Reshape back to [B, T, dim].
            Tensor attnFlat = new(new TensorShape(batch, t, _cfg.TransformerDim), DType.F32);
            float* ap = (float*)attnOut.DataPointer;
            float* fp = (float*)attnFlat.DataPointer;
            for (int b = 0; b < batch; b++)
                for (int ti = 0; ti < t; ti++)
                    for (int h = 0; h < _heads; h++)
                        for (int d = 0; d < _headDim; d++)
                            fp[(b * t + ti) * _cfg.TransformerDim + h * _headDim + d]
                                = ap[((b * _heads + h) * t + ti) * _headDim + d];
            attnOut.Dispose();

            // Output projection + residual.
            Tensor oProj = WhisperOps.ProjectLinear(backend, attnFlat, _oW[i]!, null, batch, t, _cfg.TransformerDim, _cfg.TransformerDim);
            attnFlat.Dispose();
            Tensor afterAttn = new(current.Shape, DType.F32);
            backend.Add(afterAttn, current, oProj);
            oProj.Dispose();
            if (ownsCurrent) current.Dispose();
            current = afterAttn;
            ownsCurrent = true;

            // RMSNorm pre-FFN.
            Tensor normedFfn = new(current.Shape, DType.F32);
            backend.RmsNorm(normedFfn, current, _normFfnW[i]!, 1e-5f);

            // FFN: Linear → GeLU → Linear.
            Tensor up = WhisperOps.ProjectLinear(backend, normedFfn, _fc1W[i]!, null, batch, t, _cfg.TransformerDim, _cfg.TransformerFfnDim);
            normedFfn.Dispose();
            Tensor act = new(up.Shape, DType.F32);
            backend.Gelu(act, up);
            up.Dispose();
            Tensor down = WhisperOps.ProjectLinear(backend, act, _fc2W[i]!, null, batch, t, _cfg.TransformerFfnDim, _cfg.TransformerDim);
            act.Dispose();

            // FFN residual.
            Tensor afterFfn = new(current.Shape, DType.F32);
            backend.Add(afterFfn, current, down);
            down.Dispose();
            current.Dispose();
            current = afterFfn;
        }

        causalMask.Dispose();
        return ownsCurrent ? current : x;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        for (int i = 0; i < _layers; i++)
        {
            Tensor?[] all = [_normAttnW[i], _normFfnW[i], _qW[i], _kW[i], _vW[i], _oW[i], _fc1W[i], _fc2W[i]];
            foreach (Tensor? t in all) if (t is not null) yield return t;
        }
    }

    private static Tensor BuildCausalMask(int t)
    {
        Tensor mask = new(new TensorShape(t, t), DType.F32);
        float* mp = (float*)mask.DataPointer;
        for (int i = 0; i < t; i++)
        {
            for (int j = 0; j < t; j++)
                mp[i * t + j] = j <= i ? 0f : float.NegativeInfinity;
        }
        return mask;
    }

    /// <summary>Apply rotary positional embedding to <c>[B, H, T, D]</c> in place. Pairs
    /// up (d_2i, d_2i+1) and rotates each pair by angle <c>pos * (theta^(-2i/D))</c>.</summary>
    private static void ApplyRoPE(Tensor x, int batch, int heads, int t, int headDim, float theta)
    {
        float* xp = (float*)x.DataPointer;
        int halfDim = headDim / 2;
        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < heads; h++)
            {
                for (int ti = 0; ti < t; ti++)
                {
                    int rowBase = ((b * heads + h) * t + ti) * headDim;
                    for (int k = 0; k < halfDim; k++)
                    {
                        float freq = MathF.Pow(theta, -2f * k / headDim);
                        float angle = ti * freq;
                        float cos = MathF.Cos(angle);
                        float sin = MathF.Sin(angle);
                        float x0 = xp[rowBase + k];
                        float x1 = xp[rowBase + halfDim + k];
                        xp[rowBase + k] = x0 * cos - x1 * sin;
                        xp[rowBase + halfDim + k] = x0 * sin + x1 * cos;
                    }
                }
            }
        }
    }
}
