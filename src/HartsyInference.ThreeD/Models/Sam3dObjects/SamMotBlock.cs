using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Norm;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Sam3dObjects;

/// <summary>
/// One block of the mixture-of-transformers structure flow (upstream <c>MOTModulatedTransformerCrossBlock</c>, <c>MOTMultiHeadSelfAttention</c>):
/// every stream has its own norms, attention projections, QK RMS norms, cross-attention and MLP, while the timestep modulation (<c>adaLN_modulation</c>,
/// six chunks of the model width) is shared. Self-attention joins the streams: a protected stream (the shape grid) attends only to itself, every
/// other stream attends to all other-stream tokens and to the protected stream's tokens. All streams cross-attend to the condition tokens.
/// Weight keys under <c>blocks.{i}.</c>, with <c>{s}</c> the stream name: <c>adaLN_modulation.1.{weight,bias}</c>, <c>norm2.{s}.{weight,bias}</c>,
/// <c>self_attn.to_qkv.{s}.*</c>, <c>self_attn.{q,k}_rms_norm.{s}.gamma</c> (<c>[heads, head_dim]</c>), <c>self_attn.to_out.{s}.*</c>,
/// <c>cross_attn.{s}.{to_q,to_kv,to_out}.*</c> and <c>mlp.{s}.mlp.{0,2}.*</c>; <c>norm1</c> and <c>norm3</c> have no parameters. Key names follow the module
/// structure in the source and are unverified against a checkpoint.
/// </summary>
internal sealed class SamMotBlock
{
    private const float NormEps = 1e-6f;
    private const float RmsNoEps = 0f;

    private readonly SamMotFlowConfig _config;
    private readonly IReadOnlyList<string> _streams;
    private readonly AttentionLayer _attention;
    private readonly Tensor _modulationWeight;
    private readonly Tensor _modulationBias;
    private readonly Tensor _onesHeadDim;
    private readonly Dictionary<string, StreamWeights> _weights = new(StringComparer.Ordinal);

    public SamMotBlock(SamMotFlowConfig config, IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _config = config;
        _streams = config.StreamNames;
        int c = config.ModelChannels, heads = config.Heads, headDim = config.HeadDim, hidden = config.MlpHidden, cond = config.CondChannels;
        _attention = new AttentionLayer(new AttentionLayerConfig { Heads = heads, KvHeads = heads, HeadDim = headDim, ComputeDType = DType.F32, AllowF16 = false });
        _modulationWeight = SamWeights.Require(weights, prefix + "adaLN_modulation.1.weight", 6 * c, c);
        _modulationBias = SamWeights.Require(weights, prefix + "adaLN_modulation.1.bias", 6 * c);
        _onesHeadDim = new Tensor(new TensorShape(headDim), DType.F32);
        _onesHeadDim.AsSpan<float>().Fill(1f);
        foreach (string s in _streams)
        {
            string self = prefix + "self_attn.", cross = $"{prefix}cross_attn.{s}.", mlp = $"{prefix}mlp.{s}.mlp.";
            _weights[s] = new StreamWeights(
                Norm2Weight: SamWeights.Require(weights, $"{prefix}norm2.{s}.weight", c),
                Norm2Bias: SamWeights.Require(weights, $"{prefix}norm2.{s}.bias", c),
                QkvWeight: SamWeights.Require(weights, $"{self}to_qkv.{s}.weight", 3 * c, c),
                QkvBias: SamWeights.Require(weights, $"{self}to_qkv.{s}.bias", 3 * c),
                QGamma: SamWeights.Require(weights, $"{self}q_rms_norm.{s}.gamma", heads, headDim).Reshape(new TensorShape(c)),
                KGamma: SamWeights.Require(weights, $"{self}k_rms_norm.{s}.gamma", heads, headDim).Reshape(new TensorShape(c)),
                OutWeight: SamWeights.Require(weights, $"{self}to_out.{s}.weight", c, c),
                OutBias: SamWeights.Require(weights, $"{self}to_out.{s}.bias", c),
                CrossQWeight: SamWeights.Require(weights, cross + "to_q.weight", c, c),
                CrossQBias: SamWeights.Require(weights, cross + "to_q.bias", c),
                CrossKvWeight: SamWeights.Require(weights, cross + "to_kv.weight", 2 * c, cond),
                CrossKvBias: SamWeights.Require(weights, cross + "to_kv.bias", 2 * c),
                CrossOutWeight: SamWeights.Require(weights, cross + "to_out.weight", c, c),
                CrossOutBias: SamWeights.Require(weights, cross + "to_out.bias", c),
                Mlp0Weight: SamWeights.Require(weights, mlp + "0.weight", hidden, c),
                Mlp0Bias: SamWeights.Require(weights, mlp + "0.bias", hidden),
                Mlp2Weight: SamWeights.Require(weights, mlp + "2.weight", c, hidden),
                Mlp2Bias: SamWeights.Require(weights, mlp + "2.bias", c));
        }
    }

    /// <summary>Runs the block over the stream tokens <paramref name="x"/> (F32 <c>[1, tokens, model]</c> per stream, taken over and disposed) with the
    /// timestep embedding <paramref name="timeEmbedding"/> <c>[1, model]</c> and the condition tokens <paramref name="cond"/> <c>[1, n, cond]</c>;
    /// returns the new stream tokens, owned by the caller.</summary>
    public Dictionary<string, Tensor> Forward(IBackend backend, Dictionary<string, Tensor> x, Tensor timeEmbedding, Tensor cond)
    {
        Tensor[] m = Modulation(backend, timeEmbedding);
        try
        {
            // Self-attention: x += gate_msa · attn(modulate(norm1(x))).
            Dictionary<string, Tensor> h = [];
            foreach (string s in _streams)
                h[s] = Modulated(backend, x[s], m[1], m[0]);
            Dictionary<string, Tensor> attended = SelfAttention(backend, h);
            foreach (string s in _streams)
                x[s] = GatedAdd(backend, x[s], attended[s], m[2]);

            // Cross-attention to the condition tokens: x += cross_attn(norm2(x)), ungated.
            foreach (string s in _streams)
            {
                Tensor normed = new(x[s].Shape, DType.F32);
                backend.LayerNorm(normed, x[s], _weights[s].Norm2Weight, _weights[s].Norm2Bias, NormEps);
                Tensor crossed = CrossAttention(backend, _weights[s], normed, cond);
                Tensor sum = new(x[s].Shape, DType.F32);
                backend.Add(sum, x[s], crossed);
                crossed.Dispose();
                x[s].Dispose();
                x[s] = sum;
            }

            // MLP: x += gate_mlp · mlp(modulate(norm3(x))).
            foreach (string s in _streams)
            {
                Tensor modulated = Modulated(backend, x[s], m[4], m[3]);
                StreamWeights w = _weights[s];
                Tensor up = new(new TensorShape(1, x[s].Shape[1], _config.MlpHidden), DType.F32);
                backend.Linear(up, modulated, w.Mlp0Weight, w.Mlp0Bias);
                modulated.Dispose();
                Tensor activated = new(up.Shape, DType.F32);
                backend.Gelu(activated, up);
                up.Dispose();
                Tensor down = new(x[s].Shape, DType.F32);
                backend.Linear(down, activated, w.Mlp2Weight, w.Mlp2Bias);
                activated.Dispose();
                x[s] = GatedAdd(backend, x[s], down, m[5]);
            }

            return x;
        }
        finally
        {
            foreach (Tensor chunk in m)
                chunk.Dispose();
        }
    }

    /// <summary>The six <c>[1, model]</c> chunks of <c>Linear(SiLU(t))</c>: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp.</summary>
    private Tensor[] Modulation(IBackend backend, Tensor timeEmbedding)
    {
        int c = _config.ModelChannels;
        using Tensor activated = new(timeEmbedding.Shape, DType.F32);
        backend.Silu(activated, timeEmbedding);
        using Tensor all = new(new TensorShape(1, 6 * c), DType.F32);
        backend.Linear(all, activated, _modulationWeight, _modulationBias);
        Tensor[] chunks = new Tensor[6];
        for (int i = 0; i < 6; i++)
        {
            chunks[i] = new Tensor(new TensorShape(1, c), DType.F32);
            backend.SliceLastDim(chunks[i], all, i * c);
        }

        return chunks;
    }

    /// <summary><c>layer_norm(x) · (1 + scale) + shift</c> without affine; <paramref name="x"/> is kept.</summary>
    private static Tensor Modulated(IBackend backend, Tensor x, Tensor scale, Tensor shift)
    {
        Tensor result = new(x.Shape, DType.F32);
        backend.NormModulate(NormModulateCall.Rows(result, x, NormKind.LayerNorm, NormEps).WithModulation(ModulationSpec.Broadcast(x, scale, shift, ScaleMode.OnePlus)));
        return result;
    }

    /// <summary><c>x + gate · value</c>; disposes <paramref name="x"/> and <paramref name="value"/>.</summary>
    private static Tensor GatedAdd(IBackend backend, Tensor x, Tensor value, Tensor gate)
    {
        Tensor result = new(x.Shape, DType.F32);
        backend.GatedResidualLastDim(result, x, value, gate);
        x.Dispose();
        value.Dispose();
        return result;
    }

    private Dictionary<string, Tensor> SelfAttention(IBackend backend, Dictionary<string, Tensor> h)
    {
        int c = _config.ModelChannels;
        Dictionary<string, Tensor> q = [], k = [], v = [];
        foreach (string s in _streams)
        {
            StreamWeights w = _weights[s];
            using Tensor packed = new(new TensorShape(1, h[s].Shape[1], 3 * c), DType.F32);
            backend.Linear(packed, h[s], w.QkvWeight, w.QkvBias);
            h[s].Dispose();
            q[s] = QkNorm(backend, SamMotOps.Slice(backend, packed, 0, c), w.QGamma);
            k[s] = QkNorm(backend, SamMotOps.Slice(backend, packed, c, c), w.KGamma);
            v[s] = SamMotOps.Slice(backend, packed, 2 * c, c);
        }

        Dictionary<string, Tensor> attended = [];
        List<string> others = _streams.Where(s => !_config.Protected.Contains(s)).ToList();
        if (others.Count > 0)
        {
            // The other streams' queries see every other-stream token plus the protected streams' (kept before they are consumed below).
            List<string> keyOrder = [.. others, .. _config.Protected];
            Tensor allQ = SamMotOps.Concat(others.Select(s => q[s]).ToList());
            Tensor allK = SamMotOps.Concat(keyOrder.Select(s => k[s]).ToList());
            Tensor allV = SamMotOps.Concat(keyOrder.Select(s => v[s]).ToList());
            using Tensor joint = Attend(backend, allQ, allK, allV);
            int start = 0;
            foreach (string s in others)
            {
                int count = (int)q[s].Shape[1];
                attended[s] = SamMotOps.Tokens(joint, start, count);
                start += count;
                q[s].Dispose();
                k[s].Dispose();
                v[s].Dispose();
            }
        }

        foreach (string s in _config.Protected)
            attended[s] = Attend(backend, q[s], k[s], v[s]);

        foreach (string s in _streams)
        {
            StreamWeights w = _weights[s];
            Tensor projected = new(attended[s].Shape, DType.F32);
            backend.Linear(projected, attended[s], w.OutWeight, w.OutBias);
            attended[s].Dispose();
            attended[s] = projected;
        }

        return attended;
    }

    private Tensor CrossAttention(IBackend backend, StreamWeights w, Tensor normed, Tensor cond)
    {
        int c = _config.ModelChannels;
        Tensor queries = new(normed.Shape, DType.F32);
        backend.Linear(queries, normed, w.CrossQWeight, w.CrossQBias);
        normed.Dispose();
        using Tensor kv = new(new TensorShape(1, cond.Shape[1], 2 * c), DType.F32);
        backend.Linear(kv, cond, w.CrossKvWeight, w.CrossKvBias);
        Tensor attended = Attend(backend, queries, SamMotOps.Slice(backend, kv, 0, c), SamMotOps.Slice(backend, kv, c, c));
        Tensor result = new(attended.Shape, DType.F32);
        backend.Linear(result, attended, w.CrossOutWeight, w.CrossOutBias);
        attended.Dispose();
        return result;
    }

    /// <summary>Full attention over token-major <c>[1, n, model]</c> operands (all three disposed); returns token-major output.</summary>
    private Tensor Attend(IBackend backend, Tensor q, Tensor k, Tensor v)
    {
        int heads = _config.Heads, headDim = _config.HeadDim;
        Tensor output = _attention.AttendHeadMajorOperands(
            backend,
            SamMotOps.ToHeadMajor(backend, q, heads, headDim),
            SamMotOps.ToHeadMajor(backend, k, heads, headDim),
            SamMotOps.ToHeadMajor(backend, v, heads, headDim),
            default);
        return SamMotOps.ToTokenMajor(backend, output);
    }

    /// <summary>Upstream <c>MultiHeadRMSNorm</c>: per head, <c>x / rms(x) · gamma</c> (<c>normalize(x) · sqrt(head_dim) · gamma</c>); <paramref name="flat"/> is disposed.</summary>
    private Tensor QkNorm(IBackend backend, Tensor flat, Tensor gammaFlat)
    {
        int tokens = (int)flat.Shape[1];
        Tensor normed = new(new TensorShape(1, tokens, _config.Heads, _config.HeadDim), DType.F32);
        backend.RmsNorm(normed, flat.Reshape(normed.Shape), _onesHeadDim, RmsNoEps);
        flat.Dispose();
        Tensor scaled = new(new TensorShape(1, tokens, _config.ModelChannels), DType.F32);
        backend.AffineBroadcastLastDim(scaled, normed.Reshape(scaled.Shape), gammaFlat, null);
        normed.Dispose();
        return scaled;
    }

    private sealed record StreamWeights(
        Tensor Norm2Weight, Tensor Norm2Bias,
        Tensor QkvWeight, Tensor QkvBias, Tensor QGamma, Tensor KGamma, Tensor OutWeight, Tensor OutBias,
        Tensor CrossQWeight, Tensor CrossQBias, Tensor CrossKvWeight, Tensor CrossKvBias, Tensor CrossOutWeight, Tensor CrossOutBias,
        Tensor Mlp0Weight, Tensor Mlp0Bias, Tensor Mlp2Weight, Tensor Mlp2Bias);
}
