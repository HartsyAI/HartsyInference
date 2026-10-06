using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Ssm;

public sealed unsafe partial class Qwen35Model
{
    /// <summary>Builds the text trunk from HuggingFace-keyed weights (<c>model.language_model.*</c>, as shipped in the
    /// Qwen3.5 and Clef checkpoints). Norm weights are stored as <c>w</c> and applied as <c>1 + w</c>, so they are converted
    /// once; everything else is used as stored.</summary>
    public static Qwen35Model FromHuggingFace(IReadOnlyDictionary<string, Tensor> hf, Qwen35HfConfig cfg, int maxSequenceLength,
        string prefix = "model.language_model")
    {
        ArgumentNullException.ThrowIfNull(hf);
        ArgumentNullException.ThrowIfNull(cfg);
        if (cfg.LinearNumValueHeads % cfg.LinearNumKeyHeads != 0)
        {
            throw new ArgumentException("Value heads must be a multiple of key heads.", nameof(cfg));
        }
        List<Tensor> owned = [];
        Dictionary<string, Tensor> w = new(StringComparer.Ordinal)
        {
            ["token_embd.weight"] = hf[$"{prefix}.embed_tokens.weight"],
            ["output_norm.weight"] = OnePlus(hf[$"{prefix}.norm.weight"], owned),
        };
        int rotaryDim = (int)(cfg.HeadDim * cfg.PartialRotaryFactor);
        for (int i = 0; i < cfg.NumLayers; i++)
        {
            string src = $"{prefix}.layers.{i}", dst = $"blk.{i}";
            w[$"{dst}.attn_norm.weight"] = OnePlus(hf[$"{src}.input_layernorm.weight"], owned);
            w[$"{dst}.post_attention_norm.weight"] = OnePlus(hf[$"{src}.post_attention_layernorm.weight"], owned);
            w[$"{dst}.ffn_gate.weight"] = hf[$"{src}.mlp.gate_proj.weight"];
            w[$"{dst}.ffn_up.weight"] = hf[$"{src}.mlp.up_proj.weight"];
            w[$"{dst}.ffn_down.weight"] = hf[$"{src}.mlp.down_proj.weight"];
            if ((i + 1) % cfg.FullAttentionInterval != 0)
            {
                string la = $"{src}.linear_attn";
                w[$"{dst}.attn_qkv.weight"] = hf[$"{la}.in_proj_qkv.weight"];
                w[$"{dst}.attn_gate.weight"] = hf[$"{la}.in_proj_z.weight"];
                w[$"{dst}.ssm_beta.weight"] = hf[$"{la}.in_proj_b.weight"];
                w[$"{dst}.ssm_alpha.weight"] = hf[$"{la}.in_proj_a.weight"];
                w[$"{dst}.ssm_out.weight"] = hf[$"{la}.out_proj.weight"];
                w[$"{dst}.ssm_conv1d.weight"] = ToF32(hf[$"{la}.conv1d.weight"], owned);
                w[$"{dst}.ssm_dt.bias"] = ToF32(hf[$"{la}.dt_bias"], owned);
                w[$"{dst}.ssm_norm.weight"] = ToF32(hf[$"{la}.norm.weight"], owned);
                w[$"{dst}.ssm_a"] = NegExp(hf[$"{la}.A_log"], owned);
            }
            else
            {
                string sa = $"{src}.self_attn";
                w[$"{dst}.attn_q.weight"] = hf[$"{sa}.q_proj.weight"];
                w[$"{dst}.attn_k.weight"] = hf[$"{sa}.k_proj.weight"];
                w[$"{dst}.attn_v.weight"] = hf[$"{sa}.v_proj.weight"];
                w[$"{dst}.attn_output.weight"] = hf[$"{sa}.o_proj.weight"];
                w[$"{dst}.attn_q_norm.weight"] = OnePlus(hf[$"{sa}.q_norm.weight"], owned);
                w[$"{dst}.attn_k_norm.weight"] = OnePlus(hf[$"{sa}.k_norm.weight"], owned);
            }
        }
        Qwen35Model model = new(null, w, cfg.HiddenSize, cfg.NumLayers, cfg.VocabSize, cfg.RmsNormEps, cfg.FullAttentionInterval, null,
            cfg.NumHeads, cfg.NumKvHeads, cfg.HeadDim, rotaryDim, cfg.RopeTheta, cfg.LinearConvKernel, cfg.LinearKeyHeadDim,
            cfg.LinearValueHeadDim, cfg.LinearNumKeyHeads, cfg.LinearNumValueHeads, new MoeFeedForward?[cfg.NumLayers], maxSequenceLength);
        model._ownedTensors.AddRange(owned);
        return model;
    }

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

    /// <summary>Copies one row of a <c>[rows, cols]</c> F32/BF16 matrix (an embedding or <c>lm_head</c> table) into F32.</summary>
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
        else
        {
            throw new NotSupportedException($"Embedding rows are read from F32 or BF16 tables, not {table.DType}.");
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
            p[i] += 1f;
        }
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
