using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Models.Shared;
using HartsyInference.LLM.Ssm;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.LLM.Models.Spark;

/// <summary>Spark-X decoder from HuggingFace-keyed weights: pre-norm layers of fused-QKV head-gated attention (sliding or full, partial RoPE per type) and a GELU-gated MLP. Carries its own KV cache, so callers pass only the new tokens.</summary>
public sealed unsafe class SparkModel : ISsmModel
{
    private const string Prefix = "model";

    private readonly IReadOnlyDictionary<string, Tensor> _weights;
    private readonly List<Tensor> _owned = [];
    private readonly FixedKvCache _kv;
    private readonly AttentionLayer _attention;
    private readonly IDisposable? _owner;
    private int _pos;
    private int _disposed;

    public SparkConfig Config { get; }

    public int VocabSize => Config.VocabSize;

    /// <summary>Empty: a Hugging Face checkpoint carries no GGUF metadata; its tokenizer and template are built from the directory's files.</summary>
    public GgufMetadata Metadata { get; } = new();

    /// <summary>Tokens consumed since the last <see cref="ResetState"/>.</summary>
    public int Position => _pos;

    /// <summary>Wraps <paramref name="weights"/>. Norm weights are widened to F32 once; projections are used as stored.</summary>
    /// <param name="maxSequenceLength">Longest sequence (prompt plus generation) the KV cache holds.</param>
    /// <param name="owner">What backs <paramref name="weights"/> (a safetensors loader); disposed with the model. Null leaves them with the caller, who must keep them alive.</param>
    /// <exception cref="KeyNotFoundException">A tensor the config needs is missing.</exception>
    public SparkModel(IReadOnlyDictionary<string, Tensor> weights, SparkConfig config, int maxSequenceLength, IDisposable? owner = null)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(config);
        Config = config;
        _owner = owner;
        Dictionary<string, Tensor> staged = new(StringComparer.Ordinal);
        try
        {
            Stage(weights, staged, $"{Prefix}.embedding.weight", asF32: false);
            Stage(weights, staged, $"{Prefix}.norm.weight", asF32: true);
            if (!config.TieWordEmbeddings)
            {
                Stage(weights, staged, "lm_head.weight", asF32: false);
            }
            for (int i = 0; i < config.NumLayers; i++)
            {
                StageLayer(weights, staged, i);
            }
            _weights = staged;
            _kv = new FixedKvCache(config.NumLayers, 1, config.NumKvHeads, config.HeadDim, maxSequenceLength);
            _attention = new AttentionLayer(new AttentionLayerConfig
            {
                Heads = config.NumHeads, KvHeads = config.NumKvHeads, HeadDim = config.HeadDim, ComputeDType = DType.F32,
            });
        }
        catch
        {
            foreach (Tensor t in _owned)
            {
                t.Dispose();
            }
            throw;
        }
    }

    /// <summary>Forgets the KV cache; call before an unrelated sequence.</summary>
    public void ResetState()
    {
        _kv.Reset();
        _pos = 0;
    }

    /// <summary>Runs the new tokens <paramref name="ids"/> through the stack and returns the next-token logits of the last one. A call that throws resets the model, so the next call starts a new sequence.</summary>
    public float[] ForwardLastLogits(IBackend backend, IReadOnlyList<int> ids)
    {
        Tensor normed = ForwardHidden(backend, ids);
        int seq = ids.Count, d = Config.HiddenSize, vocab = Config.VocabSize;
        Tensor last = new(new TensorShape(1, 1, 1, d), DType.F32);
        Tensor logits = new(new TensorShape(1, 1, vocab), DType.F32);
        try
        {
            backend.SliceTimeRange(last, normed.Reshape(new TensorShape(1, 1, seq, d)), seq - 1, 1);
            Tensor head = Config.TieWordEmbeddings ? W($"{Prefix}.embedding.weight") : W("lm_head.weight");
            backend.Linear(logits, last, head, null);
            backend.Sync();
            return new ReadOnlySpan<float>((void*)logits.DataPointer, vocab).ToArray();
        }
        finally
        {
            normed.Dispose();
            last.Dispose();
            logits.Dispose();
        }
    }

    /// <summary>Runs the new tokens and returns every position's final-norm hidden state <c>[1, ids.Count, hidden]</c>; the caller owns it.</summary>
    public Tensor ForwardHidden(IBackend backend, IReadOnlyList<int> ids)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        int seq = ids.Count, d = Config.HiddenSize;
        if (seq == 0)
        {
            throw new ArgumentException("At least one token is required.", nameof(ids));
        }
        if (_pos + seq > _kv.MaxSequenceLength)
        {
            throw new InvalidOperationException(
                $"{_pos + seq} tokens exceed the {_kv.MaxSequenceLength}-token limit this model was loaded with (its KV cache size, not the checkpoint's own context limit).");
        }
        Tensor h = new(new TensorShape(1, seq, d), DType.F32);
        Tensor table = W($"{Prefix}.embedding.weight");
        for (int s = 0; s < seq; s++)
        {
            if ((uint)ids[s] >= (uint)Config.VocabSize)
            {
                h.Dispose();
                throw new ArgumentOutOfRangeException(nameof(ids), $"Token id {ids[s]} is outside the {Config.VocabSize}-entry vocabulary.");
            }
            Qwen35Model.CopyEmbeddingRow(table, ids[s], d, (float*)h.DataPointer + (long)s * d);
        }
        (Tensor Cos, Tensor Sin)?[] ropes = new (Tensor, Tensor)?[2];
        try
        {
            for (int i = 0; i < Config.NumLayers; i++)
            {
                h = RunLayer(backend, h, i, seq, ropes);
            }
        }
        catch
        {
            ResetState();
            throw;
        }
        finally
        {
            foreach ((Tensor Cos, Tensor Sin)? entry in ropes)
            {
                entry?.Cos.Dispose();
                entry?.Sin.Dispose();
            }
        }
        Tensor normed = new(h.Shape, DType.F32);
        backend.RmsNorm(normed, h, W($"{Prefix}.norm.weight"), Config.NormEps);
        h.Dispose();
        _kv.AdvanceLength(seq);
        _pos += seq;
        return normed;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }
        foreach (Tensor t in _owned)
        {
            t.Dispose();
        }
        _kv.Dispose();
        _owner?.Dispose();
    }

    /// <summary>Loads the Hugging Face directory <paramref name="directory"/> (<c>config.json</c> and the <c>model-*.safetensors</c> shards); the model owns the mapped weights.</summary>
    public static SparkModel Load(string directory, int maxSequenceLength)
    {
        SparkConfig config = SparkConfig.FromJson(File.ReadAllText(Path.Combine(directory, "config.json")));
        SafeTensorsShardLoader loader = new();
        try
        {
            loader.LoadDirectory(directory);
            return new SparkModel(loader.GetAllTensors(), config, maxSequenceLength, loader);
        }
        catch
        {
            loader.Dispose();
            throw;
        }
    }

    private Tensor RunLayer(IBackend backend, Tensor h, int i, int seq, (Tensor Cos, Tensor Sin)?[] ropes)
    {
        string p = $"{Prefix}.layers.{i}";
        TensorShape shape = h.Shape;
        Tensor xn = new(shape, DType.F32);
        backend.RmsNorm(xn, h, W($"{p}.input_layernorm.weight"), Config.NormEps);
        Tensor attended = Attention(backend, xn, i, seq, ropes);
        xn.Dispose();
        Tensor afterAttention = new(shape, DType.F32);
        backend.Add(afterAttention, h, attended);
        h.Dispose();
        attended.Dispose();

        Tensor ffnIn = new(shape, DType.F32);
        backend.RmsNorm(ffnIn, afterAttention, W($"{p}.post_attention_layernorm.weight"), Config.NormEps);
        int inner = Config.IntermediateSize;
        Tensor gate = new(new TensorShape(1, seq, inner), DType.F32);
        Tensor up = new(gate.Shape, DType.F32);
        backend.Linear(gate, ffnIn, W($"{p}.mlp.gate_proj.weight"), null);
        backend.Linear(up, ffnIn, W($"{p}.mlp.up_proj.weight"), null);
        ffnIn.Dispose();
        backend.GeluErf(gate, gate);
        backend.Mul(gate, gate, up);
        up.Dispose();
        Tensor down = new(shape, DType.F32);
        backend.Linear(down, gate, W($"{p}.mlp.down_proj.weight"), null);
        gate.Dispose();
        Tensor result = new(shape, DType.F32);
        backend.Add(result, afterAttention, down);
        afterAttention.Dispose();
        down.Dispose();
        return result;
    }

    /// <summary>Fused-QKV grouped-query attention with partial RoPE, an optional sliding window and a sigmoid output gate per head.</summary>
    private Tensor Attention(IBackend backend, Tensor xn, int i, int seq, (Tensor Cos, Tensor Sin)?[] ropes)
    {
        string p = $"{Prefix}.layers.{i}.self_attn";
        int hq = Config.NumHeads, hkv = Config.NumKvHeads, hd = Config.HeadDim;
        int qWidth = hq * hd, kvWidth = hkv * hd;
        Tensor qkv = new(new TensorShape(1, seq, qWidth + 2 * kvWidth), DType.F32);
        backend.Linear(qkv, xn, W($"{p}.q_k_v_proj.weight"), null);
        Tensor qFlat = new(new TensorShape(1, seq, qWidth), DType.F32);
        Tensor kFlat = new(new TensorShape(1, seq, kvWidth), DType.F32);
        Tensor vFlat = new(kFlat.Shape, DType.F32);
        backend.SliceLastDim(qFlat, qkv, 0);
        backend.SliceLastDim(kFlat, qkv, qWidth);
        backend.SliceLastDim(vFlat, qkv, qWidth + kvWidth);
        qkv.Dispose();
        Tensor q = qFlat.Reshape(new TensorShape(1, seq, hq, hd));
        Tensor k = kFlat.Reshape(new TensorShape(1, seq, hkv, hd));
        Tensor v = vFlat.Reshape(new TensorShape(1, seq, hkv, hd));
        Tensor gateLogits = new(new TensorShape(1, seq, hq), DType.F32);
        backend.Linear(gateLogits, xn, W($"{p}.g_proj.weight"), null);

        // The two layer types share their tables for the whole pass, so they are built once per type.
        LayerRopeParameters rope = Config.RopeOf(i);
        int rotaryDim = rope.RotaryDim(hd), slot = Config.SlidingLayers[i] ? 1 : 0;
        if (ropes[slot] is null)
        {
            Tensor cos = new(new TensorShape(1, seq, hd), DType.F32);
            Tensor sin = new(cos.Shape, DType.F32);
            RopeTables.BuildRope(cos, sin, seq, _pos, hd, rotaryDim, rope.Theta, RopeScaling.None);
            ropes[slot] = (cos, sin);
        }
        (Tensor ropeCos, Tensor ropeSin) = ropes[slot]!.Value;
        backend.ApplyRopeSingle(q, ropeCos, ropeSin, rotaryDim);
        backend.ApplyRopeSingle(k, ropeCos, ropeSin, rotaryDim);

        int window = Config.SlidingLayers[i] ? Config.SlidingWindow : 0;
        Tensor output = HeadGatedAttention.Attend(backend, _attention, _kv, i, q, k, v, gateLogits, _pos, window, W($"{p}.out_proj.weight"));
        qFlat.Dispose();
        kFlat.Dispose();
        vFlat.Dispose();
        return output;
    }

    private Tensor W(string key) => _weights[key];

    private void StageLayer(IReadOnlyDictionary<string, Tensor> src, Dictionary<string, Tensor> dst, int i)
    {
        string p = $"{Prefix}.layers.{i}";
        Stage(src, dst, $"{p}.input_layernorm.weight", asF32: true);
        Stage(src, dst, $"{p}.post_attention_layernorm.weight", asF32: true);
        foreach (string proj in (string[])["gate_proj", "up_proj", "down_proj"])
        {
            Stage(src, dst, $"{p}.mlp.{proj}.weight", asF32: false);
        }
        foreach (string proj in (string[])["q_k_v_proj", "g_proj", "out_proj"])
        {
            Stage(src, dst, $"{p}.self_attn.{proj}.weight", asF32: false);
        }
    }

    private void Stage(IReadOnlyDictionary<string, Tensor> src, Dictionary<string, Tensor> dst, string key, bool asF32)
    {
        if (!src.TryGetValue(key, out Tensor? t))
        {
            throw new KeyNotFoundException($"The Spark-X checkpoint has no tensor '{key}'.");
        }
        if (asF32 && t.DType != DType.F32)
        {
            t = t.CastTo(DType.F32);
            _owned.Add(t);
        }
        dst[key] = t;
    }
}
