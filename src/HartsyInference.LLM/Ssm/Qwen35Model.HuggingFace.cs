using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Ssm;

public sealed unsafe partial class Qwen35Model
{
    /// <summary>Builds the text trunk from HuggingFace-keyed weights (<c>model.language_model.*</c>, as shipped in the
    /// Qwen3.5 and Clef checkpoints). Norm weights are stored as <c>w</c> and applied as <c>1 + w</c>, so they are converted
    /// once; everything else is used as stored, quantized or not. The MoE tier's routed experts load from per-expert
    /// <c>mlp.experts.{e}.*_proj.weight</c> tensors or from the fused <c>mlp.experts.gate_up_proj</c> / <c>down_proj</c>
    /// stacks; an untied model's head is <paramref name="lmHeadKey"/>, or none when it is null — for a caller that projects the
    /// hidden states itself (<see cref="ProjectLogits"/> then refuses).</summary>
    public static Qwen35Model FromHuggingFace(IReadOnlyDictionary<string, Tensor> hf, Qwen35HfConfig cfg, int maxSequenceLength,
        string prefix = "model.language_model", string? lmHeadKey = "lm_head.weight")
    {
        ArgumentNullException.ThrowIfNull(hf);
        ArgumentNullException.ThrowIfNull(cfg);
        if (cfg.LinearNumValueHeads % cfg.LinearNumKeyHeads != 0)
        {
            throw new ArgumentException("Value heads must be a multiple of key heads.", nameof(cfg));
        }
        List<Tensor> owned = [];
        MoeFeedForward?[] moe = new MoeFeedForward?[cfg.NumLayers];
        try
        {
            Dictionary<string, Tensor> w = new(StringComparer.Ordinal)
            {
                ["token_embd.weight"] = Require(hf, $"{prefix}.embed_tokens.weight"),
                ["output_norm.weight"] = OnePlus(Require(hf, $"{prefix}.norm.weight"), owned),
            };
            if (!cfg.TieWordEmbeddings && lmHeadKey is not null)
            {
                w["output.weight"] = Require(hf, lmHeadKey);
            }
            bool[] linear = new bool[cfg.NumLayers];
            for (int i = 0; i < cfg.NumLayers; i++)
            {
                string src = $"{prefix}.layers.{i}", dst = $"blk.{i}";
                linear[i] = cfg.IsLinearLayer(i);
                w[$"{dst}.attn_norm.weight"] = OnePlus(Require(hf, $"{src}.input_layernorm.weight"), owned);
                w[$"{dst}.post_attention_norm.weight"] = OnePlus(Require(hf, $"{src}.post_attention_layernorm.weight"), owned);
                if (cfg.IsMoe)
                {
                    moe[i] = BuildHfMoe(hf, cfg, src);
                }
                else
                {
                    w[$"{dst}.ffn_gate.weight"] = Require(hf, $"{src}.mlp.gate_proj.weight");
                    w[$"{dst}.ffn_up.weight"] = Require(hf, $"{src}.mlp.up_proj.weight");
                    w[$"{dst}.ffn_down.weight"] = Require(hf, $"{src}.mlp.down_proj.weight");
                }
                if (linear[i])
                {
                    string la = $"{src}.linear_attn";
                    w[$"{dst}.attn_qkv.weight"] = Require(hf, $"{la}.in_proj_qkv.weight");
                    w[$"{dst}.attn_gate.weight"] = Require(hf, $"{la}.in_proj_z.weight");
                    w[$"{dst}.ssm_beta.weight"] = Require(hf, $"{la}.in_proj_b.weight");
                    w[$"{dst}.ssm_alpha.weight"] = Require(hf, $"{la}.in_proj_a.weight");
                    w[$"{dst}.ssm_out.weight"] = Require(hf, $"{la}.out_proj.weight");
                    w[$"{dst}.ssm_conv1d.weight"] = ToF32(Require(hf, $"{la}.conv1d.weight"), owned);
                    w[$"{dst}.ssm_dt.bias"] = ToF32(Require(hf, $"{la}.dt_bias"), owned);
                    w[$"{dst}.ssm_norm.weight"] = ToF32(Require(hf, $"{la}.norm.weight"), owned);
                    w[$"{dst}.ssm_a"] = NegExp(Require(hf, $"{la}.A_log"), owned);
                }
                else
                {
                    string sa = $"{src}.self_attn";
                    w[$"{dst}.attn_q.weight"] = Require(hf, $"{sa}.q_proj.weight");
                    w[$"{dst}.attn_k.weight"] = Require(hf, $"{sa}.k_proj.weight");
                    w[$"{dst}.attn_v.weight"] = Require(hf, $"{sa}.v_proj.weight");
                    w[$"{dst}.attn_output.weight"] = Require(hf, $"{sa}.o_proj.weight");
                    w[$"{dst}.attn_q_norm.weight"] = OnePlus(Require(hf, $"{sa}.q_norm.weight"), owned);
                    w[$"{dst}.attn_k_norm.weight"] = OnePlus(Require(hf, $"{sa}.k_norm.weight"), owned);
                }
            }
            Qwen35Model model = new(null, w, cfg.HiddenSize, cfg.NumLayers, cfg.VocabSize, cfg.RmsNormEps, cfg.FullAttentionInterval,
                linear, cfg.NumHeads, cfg.NumKvHeads, cfg.HeadDim, cfg.RotaryDim, cfg.RopeTheta, cfg.LinearConvKernel,
                cfg.LinearKeyHeadDim, cfg.LinearValueHeadDim, cfg.LinearNumKeyHeads, cfg.LinearNumValueHeads, moe, maxSequenceLength);
            model._ownedTensors.AddRange(owned);
            model._headless = !cfg.TieWordEmbeddings && lmHeadKey is null;
            model.MropeSection = cfg.MropeSection ?? DefaultMropeSection;
            model.MropeInterleaved = cfg.MropeInterleaved;
            return model;
        }
        catch
        {
            foreach (Tensor t in owned) t.Dispose();
            throw;
        }
    }

    /// <summary>One MoE layer of a HuggingFace Qwen3.5-MoE checkpoint: the router, the routed experts (per-expert tensors, or the fused
    /// <c>gate_up_proj</c> <c>[E, 2I, H]</c> and <c>down_proj</c> <c>[E, H, I]</c> stacks split into per-expert views), and the
    /// sigmoid-gated shared expert.</summary>
    private static MoeFeedForward BuildHfMoe(IReadOnlyDictionary<string, Tensor> hf, Qwen35HfConfig cfg, string layer)
    {
        int e = cfg.NumExperts, inter = cfg.MoeIntermediateSize, hidden = cfg.HiddenSize;
        string mlp = $"{layer}.mlp";
        Dictionary<string, Tensor> lw = new(StringComparer.Ordinal) { [$"{layer}.mlp.gate.weight"] = Require(hf, $"{mlp}.gate.weight") };
        if (hf.TryGetValue($"{mlp}.experts.gate_up_proj", out Tensor? gateUp))
        {
            StackedExperts.SplitGateUp(lw, gateUp, $"{mlp}.experts.{{0}}.gate_proj.weight", $"{mlp}.experts.{{0}}.up_proj.weight", e, inter, hidden);
            StackedExperts.Split(lw, Require(hf, $"{mlp}.experts.down_proj"), $"{mlp}.experts.{{0}}.down_proj.weight", e, hidden, inter);
        }
        else
        {
            for (int x = 0; x < e; x++)
            {
                foreach (string proj in ExpertProjections)
                {
                    string key = $"{mlp}.experts.{x}.{proj}.weight";
                    lw[key] = Require(hf, key);
                }
            }
        }
        if (cfg.SharedExpertIntermediateSize > 0)
        {
            foreach (string proj in ExpertProjections)
            {
                string key = $"{mlp}.shared_expert.{proj}.weight";
                lw[key] = Require(hf, key);
            }
            Tensor gate = Require(hf, $"{mlp}.shared_expert_gate.weight");
            lw[$"{mlp}.shared_expert_gate.weight"] = gate.Shape.Rank == 1 ? gate.Reshape(new TensorShape(1, (int)gate.Shape[0])) : gate;
        }
        MoeConfig moeCfg = new()
        {
            NumExperts = e,
            NumExpertsPerTok = cfg.NumExpertsPerTok,
            MoeIntermediateSize = inter,
            SharedExpertIntermediateSize = cfg.SharedExpertIntermediateSize,
            Scoring = MoeScoring.Softmax,
            NormTopKProb = cfg.NormTopKProb,
            Activation = ActivationKind.Silu,
        };
        MoeFeedForward ff = new(moeCfg, hidden, lowVram: false);
        ff.LoadWeights(lw, layer);
        return ff;
    }

    private static readonly string[] ExpertProjections = ["gate_proj", "up_proj", "down_proj"];

    private static Tensor Require(IReadOnlyDictionary<string, Tensor> hf, string key) =>
        hf.TryGetValue(key, out Tensor? t) ? t : throw new KeyNotFoundException($"The Qwen3.5 checkpoint has no tensor '{key}'.");

    /// <summary>Runs the whole sequence from a fresh state and returns every position's final-norm hidden state, <c>[ids.Count, DModel]</c> row-major.</summary>
    public float[] ForwardHiddenStates(IBackend backend, IReadOnlyList<int> ids)
    {
        int seq = ids.Count, d = DModel;
        if (seq == 0)
        {
            throw new ArgumentException("At least one token is required.", nameof(ids));
        }
        ResetState();
        Tensor h = new(new TensorShape(1, seq, d), DType.F32);
        for (int s = 0; s < seq; s++)
        {
            CopyEmbeddingRow(W("token_embd.weight"), ids[s], d, (float*)h.DataPointer + (long)s * d);
        }
        Tensor normed = RunTrunk(backend, h, seq);
        float[] result = new Span<float>((void*)normed.DataPointer, seq * d).ToArray();
        normed.Dispose();
        return result;
    }

    /// <summary>Copies one row of a <c>[rows, cols]</c> F32/BF16/F16 matrix (an embedding or <c>lm_head</c> table) into F32.</summary>
    public static void CopyEmbeddingRow(Tensor table, int row, int cols, float* destination)
    {
        if (table.DType == DType.F32)
        {
            Buffer.MemoryCopy((float*)table.DataPointer + (long)row * cols, destination, (long)cols * 4, (long)cols * 4);
        }
        else if (table.DType == DType.BF16)
        {
            ushort* src = (ushort*)table.DataPointer + (long)row * cols;
            for (int c = 0; c < cols; c++)
            {
                destination[c] = BitConverter.Int32BitsToSingle(src[c] << 16);
            }
        }
        else if (table.DType == DType.F16)
        {
            Half* src = (Half*)table.DataPointer + (long)row * cols;
            for (int c = 0; c < cols; c++)
            {
                destination[c] = (float)src[c];
            }
        }
        else
        {
            throw new NotSupportedException($"Embedding rows are read from F32, BF16 or F16 tables, not {table.DType}.");
        }
    }

    private static Tensor ToF32(Tensor t, List<Tensor> owned)
    {
        if (t.DType == DType.F32)
        {
            return t;
        }
        Tensor c = t.CastTo(DType.F32);
        owned.Add(c);
        return c;
    }

    private static Tensor OnePlus(Tensor t, List<Tensor> owned)
    {
        Tensor f = NormWeights.OnePlus(t);
        owned.Add(f);
        return f;
    }

    private static Tensor NegExp(Tensor t, List<Tensor> owned)
    {
        Tensor f = new(t.Shape, DType.F32);
        Tensor converted = t.DType == DType.F32 ? t : t.CastTo(DType.F32);
        Buffer.MemoryCopy((void*)converted.DataPointer, (void*)f.DataPointer, f.ElementCount * 4, f.ElementCount * 4);
        if (!ReferenceEquals(converted, t))
        {
            converted.Dispose();
        }
        float* p = (float*)f.DataPointer;
        for (long i = 0; i < f.ElementCount; i++)
        {
            p[i] = -MathF.Exp(p[i]);
        }
        owned.Add(f);
        return f;
    }
}
