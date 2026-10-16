using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Moe;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.Gguf;

namespace HartsyInference.LLM.Ssm;

/// <summary>Qwen3.5 (<c>qwen35</c> GGUF arch) — a hybrid decoder, NOT a transformer or a pure recurrent model: most layers are Gated DeltaNet (a delta-rule linear attention), interleaved every <see cref="FullAttentionInterval"/>-th layer with a REGULAR causal self-attention layer (GQA + partial RoPE + KV cache, Qwen3-style QK-norm, plus a fused query+gate projection unique to this arch).</summary>
/// <remarks>Text-only: the GGUF's 4-section M-RoPE degenerates to a single partial-rotary RoPE (<see cref="RotaryDim"/> = sum of the 4 sections × 2) since every section gets the same position for non-multimodal input. Ported from llama.cpp's <c>src/models/{qwen35.cpp, delta-net-base.cpp}</c> (no local reference model existed at implementation time). Linear projections run through <see cref="IBackend"/>; the conv/delta-rule/gated-norm run host-side, matching every other recurrent model in this namespace (<see cref="Mamba2Model"/> et al.).</remarks>
public sealed unsafe partial class Qwen35Model : IDisposable, ISsmModel, ISsmGraphDecodable, IExpertOffloadTarget
{
    private readonly GgufModelLoader.LoadedGgufModel? _handle;
    private readonly List<Tensor> _ownedTensors = [];
    private readonly IReadOnlyDictionary<string, Tensor> _w;
    private bool _headless;   // an untied HuggingFace build whose caller projects the hidden states itself

    public IEnumerable<Tensor> EnumerateWeights()
    {
        // Skip the host-only F32 embed copy and the stacked `_exps` tensors (their per-expert views, yielded via the
        // MoE blocks below, are what the Linear path actually uses — preloading the stacked form too would double the
        // upload). Dense models have no `_exps` keys and no MoE blocks, so this reduces to the original set. With a
        // separate output head the embedding table is only gathered on the host, so it stays off the device too.
        bool untiedHead = _headless || Has("output.weight");
        foreach (KeyValuePair<string, Tensor> kv in _w)
        {
            if (kv.Key == "__embed_f32" || kv.Key.Contains("_exps", StringComparison.Ordinal)) continue;
            if (untiedHead && kv.Key == "token_embd.weight") continue;
            yield return kv.Value;
        }
        if (_moe is not null)
            foreach (MoeFeedForward? ff in _moe)
                if (ff is not null)
                    foreach (Tensor t in ff.EnumerateWeights()) yield return t;
    }

    /// <summary><see cref="EnumerateWeights"/> arranged for <see cref="IBackend.PreloadWeightGroups"/>: each MoE layer's routed
    /// experts as one group per projection, so the device-routed expert kernels find them resident, and every other weight
    /// as a group of one.</summary>
    /// <param name="includeExperts">Include the routed experts; false when an expert offload owns their residency.</param>
    public IEnumerable<IReadOnlyList<Tensor>> EnumerateWeightGroups(bool includeExperts = true)
    {
        HashSet<Tensor> grouped = new(ReferenceEqualityComparer.Instance);
        foreach (MoeFeedForward? ff in _moe)
        {
            if (ff is null) continue;
            foreach (IReadOnlyList<Tensor> group in ff.EnumerateExpertGroups())
            {
                foreach (Tensor t in group) grouped.Add(t);
                if (includeExperts) yield return group;
            }
        }
        foreach (Tensor t in EnumerateWeights())
        {
            if (!grouped.Contains(t)) yield return [t];
        }
    }

    /// <inheritdoc/>
    public int ExpertHiddenSize => DModel;

    /// <inheritdoc/>
    public int ExpertIntermediateSize => Array.Find(_moe, static ff => ff is not null)?.IntermediateSize ?? 0;

    /// <summary>Every routed expert projection of every MoE layer.</summary>
    public IEnumerable<Tensor> EnumerateExpertWeights()
    {
        foreach (MoeFeedForward? ff in _moe)
        {
            if (ff is null) continue;
            foreach (IReadOnlyList<Tensor> group in ff.EnumerateExpertGroups())
                foreach (Tensor t in group) yield return t;
        }
    }

    /// <summary>Routes every MoE layer's experts through <paramref name="offload"/> (see <see cref="MoeFeedForward.AttachOffload"/>);
    /// returns each MoE layer with its expert count, for seeding the cache.</summary>
    public IReadOnlyList<(int Layer, int ExpertCount)> AttachExpertOffload(MoeExpertOffload offload)
    {
        ArgumentNullException.ThrowIfNull(offload);
        List<(int, int)> layers = [];
        for (int i = 0; i < _moe.Length; i++)
        {
            if (_moe[i] is not MoeFeedForward ff) continue;
            ff.AttachOffload(offload, i);
            layers.Add((i, ff.NumExperts));
        }
        return layers;
    }

    private readonly bool[] _isLinear;   // per-layer: true = Gated DeltaNet, false = regular attention
    // qwen35moe: per-(trunk-)layer MoE FFN (null on a dense-tier model). _isMoe keeps MoE models off graph decode: the
    // host-routed experts' top-k read cannot live inside a captured CUDA graph.
    private readonly MoeFeedForward?[] _moe;
    private readonly bool _isMoe;
    /// <inheritdoc/>
    public bool GraphDecodeReady => !_isMoe;

    // Gated DeltaNet per-layer recurrent state (null for regular-attention layers). Carried across calls so
    // ForwardLastLogits can be fed only the NEW tokens each call, same contract as every other ISsmModel.
    // _convHistory[i]: last (ConvKernel-1) pre-conv rows of the fused QKV projection (causal conv left context).
    // _deltaState[i]: the [S_v, S_v] per-head state matrix, flattened [NumVHeads * S_v * S_v].
    private readonly float[]?[] _convHistory;
    private readonly float[]?[] _deltaState;
    // The state tensors every prefill and decode pass reads and advances in place (lazily created, then resident on
    // the backend); the host arrays above hold the reset state they are seeded from — see EnsureStateOnDevice.
    private readonly Tensor?[] _convHistT;
    private readonly Tensor?[] _deltaStateT;
    private readonly bool[] _stateOnDevice;
    // Regular-attention layers' KV cache (unused slots for linear-attn layers are simply never touched).
    // ISsmModel has no "expected max length" input, so this is a practical fixed cap, not a general solution.
    private const int MaxAttnSeqLen = 8192;
    private readonly FixedKvCache _kvCache;
    private int _pos;
    private int _disposed;

    public GgufMetadata Metadata => _handle?.Metadata ?? throw new NotSupportedException("A model built from HuggingFace weights has no GGUF metadata.");
    public int DModel { get; }
    public int NumLayers { get; }
    public int VocabSize { get; }
    public float Eps { get; }
    public int FullAttentionInterval { get; }

    // Regular-attention layer dims.
    public int NumHeads { get; }
    public int NumKvHeads { get; }
    public int AttnHeadDim { get; }      // n_embd_head_k / n_embd_head_v
    public int RotaryDim { get; }        // sum(rope.dimension_sections) * 2 — text-only M-RoPE degenerate case
    public float RopeTheta { get; }

    // Gated DeltaNet layer dims.
    public int ConvKernel { get; }
    public int HeadKDim { get; }         // ssm.state_size (S_k == S_v for every known qwen35 checkpoint)
    public int HeadVDim { get; }
    public int NumKHeads { get; }        // ssm.group_count
    public int NumVHeads { get; }        // ssm.time_step_rank
    private int KeyDim => HeadKDim * NumKHeads;
    private int ValueDim => HeadVDim * NumVHeads;
    private int ConvDim => KeyDim * 2 + ValueDim;
    private int KHeadsPerVHead => NumVHeads / NumKHeads;   // repeat factor when NumKHeads < NumVHeads

    private Qwen35Model(GgufModelLoader.LoadedGgufModel? handle, IReadOnlyDictionary<string, Tensor> w,
        int dModel, int layers, int vocab, float eps, int fullAttnInterval, bool[]? recrOverride,
        int numHeads, int numKvHeads, int attnHeadDim, int rotaryDim, float ropeTheta,
        int convK, int headKDim, int headVDim, int numKHeads, int numVHeads, MoeFeedForward?[] moe, int maxAttnSeqLen = MaxAttnSeqLen)
    {
        _handle = handle; _w = w;
        _moe = moe; _isMoe = Array.Exists(moe, m => m is not null);
        DModel = dModel; NumLayers = layers; VocabSize = vocab; Eps = eps <= 0f ? 1e-6f : eps;
        FullAttentionInterval = fullAttnInterval;
        NumHeads = numHeads; NumKvHeads = numKvHeads; AttnHeadDim = attnHeadDim;
        RotaryDim = rotaryDim; RopeTheta = ropeTheta;
        ConvKernel = convK; HeadKDim = headKDim; HeadVDim = headVDim; NumKHeads = numKHeads; NumVHeads = numVHeads;

        _isLinear = new bool[layers];
        for (int i = 0; i < layers; i++)
            _isLinear[i] = recrOverride is { Length: > 0 } ? recrOverride[i] : (i + 1) % fullAttnInterval != 0;

        _convHistory = new float[layers][];
        _deltaState = new float[layers][];
        _convHistT = new Tensor?[layers];
        _deltaStateT = new Tensor?[layers];
        _stateOnDevice = new bool[layers];
        for (int i = 0; i < layers; i++)
        {
            if (!_isLinear[i]) continue;
            _convHistory[i] = new float[(convK - 1) * ConvDim];
            _deltaState[i] = new float[NumVHeads * HeadVDim * HeadVDim];
        }
        _kvCache = new FixedKvCache(layers, 1, numKvHeads, attnHeadDim, maxAttnSeqLen);
    }

    /// <summary>Zeroes every layer's carried state (Gated DeltaNet matrices + conv history) and the KV cache — call before the first <see cref="ForwardLastLogits"/> of a new generation.</summary>
    public void ResetState()
    {
        foreach (float[]? h in _convHistory) if (h is not null) Array.Clear(h);
        foreach (float[]? s in _deltaState) if (s is not null) Array.Clear(s);
        // The zeroed host arrays are now the state; a device mirror from the last generation must not be read back over them.
        Array.Clear(_stateOnDevice);
        _kvCache.Reset();
        _pos = 0;
    }

    /// <summary>Gathers token embeddings for <paramref name="ids"/> into <paramref name="dst"/> (<c>[1, ids.Count, DModel]</c>, host F32) from the F32 embedding copy — the text-token half of a VLM's spliced <c>[pre][image][post]</c> embedding sequence (image rows are memcpy'd in by the caller).</summary>
    public void EmbedLookup(Tensor dst, IReadOnlyList<int> ids)
    {
        int d = DModel;
        float* dp = (float*)dst.DataPointer;
        // A GGUF load keeps an F32 copy for the gather; a HuggingFace load reads the stored table's rows directly.
        Tensor table = _w.TryGetValue("__embed_f32", out Tensor? f32) ? f32 : W("token_embd.weight");
        for (int s = 0; s < ids.Count; s++)
        {
            if ((uint)ids[s] >= (uint)VocabSize) throw new ArgumentOutOfRangeException(nameof(ids), $"Token id {ids[s]} is outside the {VocabSize}-token vocabulary.");
            CopyEmbeddingRow(table, ids[s], d, dp + (long)s * d);
        }
    }

    /// <summary>Runs <paramref name="embeds"/> (<c>[1, seq, DModel]</c> host F32, not consumed) through every layer, continuing the
    /// carried state, and returns every position's final-norm hidden state <c>[1, seq, DModel]</c> (caller owns).</summary>
    public Tensor ForwardEmbedsHidden(IBackend backend, Tensor embeds, int seq)
    {
        ArgumentNullException.ThrowIfNull(embeds);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(seq);
        if (embeds.ElementCount < (long)seq * DModel) throw new ArgumentException($"Embeddings hold fewer than {seq} rows of {DModel}.", nameof(embeds));
        int d = DModel;
        Tensor h = new(new TensorShape(1, seq, d), DType.F32);
        Buffer.MemoryCopy((void*)embeds.DataPointer, (void*)h.DataPointer, (long)seq * d * 4, (long)seq * d * 4);
        return RunTrunk(backend, h, seq);
    }

    /// <summary>Projects <paramref name="rows"/> hidden rows (<c>[1, rows, DModel]</c>) to vocabulary logits <c>[1, rows, VocabSize]</c> (caller owns).</summary>
    public Tensor ProjectLogits(IBackend backend, Tensor hidden, int rows)
    {
        Tensor logits = new(new TensorShape(1, rows, VocabSize), DType.F32);
        backend.Linear(logits, hidden, HeadWeight, null);
        return logits;
    }

    /// <summary>Maps one qwen35moe layer's raw GGUF MoE tensors to the HF-style names <see cref="MoeFeedForward.LoadWeights"/> expects, splitting the stacked experts into per-expert quant views (reuses the same [E·out, in] flatten + byte-range slice as the transformer path).</summary>
    private static Dictionary<string, Tensor> BuildMoeLayerWeights(IReadOnlyDictionary<string, Tensor> w, int li, int e, int inter, int hidden)
    {
        string b = $"blk.{li}";
        Dictionary<string, Tensor> lw = new(StringComparer.Ordinal) { [$"{b}.mlp.gate.weight"] = w[$"{b}.ffn_gate_inp.weight"] };
        SplitExpertsInto(lw, w, $"{b}.ffn_gate_exps.weight", $"{b}.mlp.experts.{{0}}.gate_proj.weight", e, inter, hidden);
        SplitExpertsInto(lw, w, $"{b}.ffn_up_exps.weight", $"{b}.mlp.experts.{{0}}.up_proj.weight", e, inter, hidden);
        SplitExpertsInto(lw, w, $"{b}.ffn_down_exps.weight", $"{b}.mlp.experts.{{0}}.down_proj.weight", e, hidden, inter);
        if (w.ContainsKey($"{b}.ffn_up_shexp.weight"))
        {
            lw[$"{b}.mlp.shared_expert.gate_proj.weight"] = w[$"{b}.ffn_gate_shexp.weight"];
            lw[$"{b}.mlp.shared_expert.up_proj.weight"] = w[$"{b}.ffn_up_shexp.weight"];
            lw[$"{b}.mlp.shared_expert.down_proj.weight"] = w[$"{b}.ffn_down_shexp.weight"];
            // Shared-expert sigmoid gate: a 1-logit-per-token projection stored as a rank-1 [hidden] (dequantized to
            // F32 above); MoeFeedForward's router Project wants a [1, hidden] weight, so relabel the vector.
            if (w.TryGetValue($"{b}.ffn_gate_inp_shexp.weight", out Tensor? sg))
                lw[$"{b}.mlp.shared_expert_gate.weight"] = sg.Shape.Rank == 1 ? sg.Reshape(new TensorShape(1, (int)sg.Shape[0])) : sg;
        }
        return lw;
    }

    /// <summary>Splits one stacked GGUF expert tensor into per-expert views (<see cref="StackedExperts.Split"/>); the stacked source stays in <paramref name="w"/>.</summary>
    private static void SplitExpertsInto(Dictionary<string, Tensor> dst, IReadOnlyDictionary<string, Tensor> w,
        string stackedKey, string targetFormat, int e, int outDim, int inDim)
    {
        if (w.TryGetValue(stackedKey, out Tensor? stacked)) StackedExperts.Split(dst, stacked, targetFormat, e, outDim, inDim);
    }

    public static Qwen35Model Load(string ggufPath)
    {
        // QUANT-RESIDENT load (2026-07-24): weights stay in their GGUF quantization so every
        // backend.Linear dispatches the fused quant GEMV / dp4a decode paths — the previous
        // LoadDequantized(F32) build ran all Linears on F32 weights (6× the bytes; no dp4a).
        // RelabelRank2ToPyTorchOrder is a pure metadata swap (GGUF rank-2 bytes are already
        // [out, in]), so it is quant-safe — unlike the old memcpy Relabel, which was F32-only.
        GgufModelLoader.LoadedGgufModel handle = GgufModelLoader.Load(ggufPath);
        try
        {
            Dictionary<string, Tensor> w = GgufModelLoader.RelabelRank2ToPyTorchOrder(handle.Weights);
            // Host-consumed / non-Linear tensors must be F32:
            //  - every rank-1 tensor (norm weights, ssm_a, ssm_dt.bias — RmsNorm and the host SSM
            //    math read them as float*),
            //  - ssm_conv1d.weight (host conv loop; ALSO restored to its original GGUF labeling,
            //    which the conv indexing comments assume — taken from the pre-relabel dict).
            foreach (string key in w.Keys.ToList())
            {
                if (key.Contains("ssm_conv1d"))
                {
                    Tensor conv = handle.Weights[key];
                    w[key] = conv.DType == DType.F32 ? conv
                        : conv.DType.IsQuantized ? GgufDequantizer.Dequantize(conv, DType.F32)
                        : conv.CastTo(DType.F32);
                }
                else if (w[key].Shape.Rank != 2 && w[key].DType != DType.F32 && !key.Contains("_exps", StringComparison.Ordinal))
                {
                    // Stacked MoE expert tensors (`ffn_*_exps`, rank-3) are the ONE exception: they stay quantized so
                    // the per-expert Linears keep the dp4a path — dequantizing 256 experts to F32 would be ruinous
                    // (35B → ~140GB). They are split into per-expert quant views below (quant-safe byte-range slices).
                    Tensor t = w[key];
                    w[key] = t.DType.IsQuantized ? GgufDequantizer.Dequantize(t, DType.F32) : t.CastTo(DType.F32);
                }
            }
            // Embedding gather runs host-side over F32 rows: keep a dedicated F32 copy; the dict
            // entry stays QUANT so the (tied) lm-head Linear uses the fused quant GEMV.
            Tensor embQ = w["token_embd.weight"];
            Tensor embF32 = embQ.DType == DType.F32 ? embQ
                : embQ.DType.IsQuantized ? GgufDequantizer.Dequantize(embQ, DType.F32)
                : embQ.CastTo(DType.F32);
            w["__embed_f32"] = embF32;
            GgufMetadata m = handle.Metadata;
            // "qwen35" (dense) or "qwen35moe" — the same hybrid GDN/full-attention trunk; qwen35moe adds a routed MoE
            // FFN + gated shared expert on every trunk layer (built below) and appends an MTP/NextN draft block.
            string arch = (m.GetString("general.architecture") ?? "qwen35").ToLowerInvariant();
            int dModel = (int)m.GetUInt32($"{arch}.embedding_length");
            // block_count includes the appended MTP/NextN block(s); exclude them from the executed trunk.
            int nextn = (int)m.GetUInt32($"{arch}.nextn.predict_layers", 0u);
            int layers = (int)m.GetUInt32($"{arch}.block_count") - nextn;
            int vocab = (int)w["token_embd.weight"].Shape[0];   // relabeled to [vocab, dModel]
            float eps = m.GetFloat32($"{arch}.attention.layer_norm_rms_epsilon", 1e-6f);
            int fullAttnInterval = (int)m.GetUInt32($"{arch}.full_attention_interval", 4u);
            bool[]? recrOverride = m.GetBoolArray($"{arch}.attention.recurrent_layers");

            int numHeads = (int)m.GetUInt32($"{arch}.attention.head_count");
            int numKvHeads = (int)m.GetUInt32($"{arch}.attention.head_count_kv", (uint)numHeads);
            int attnHeadDim = (int)m.GetUInt32($"{arch}.attention.key_length", 0u);
            if (attnHeadDim == 0) attnHeadDim = dModel / numHeads;
            // Text-only M-RoPE degenerate case: every section gets the same position, so the 4-section split
            // collapses into one ordinary partial-rotary table spanning rope.dimension_count dims.
            int rotaryDim = (int)m.GetUInt32($"{arch}.rope.dimension_count", 0u);
            float ropeTheta = m.GetFloat32($"{arch}.rope.freq_base", 10_000_000f);

            int convK = (int)m.GetUInt32($"{arch}.ssm.conv_kernel");
            int headKvDim = (int)m.GetUInt32($"{arch}.ssm.state_size");   // S_k == S_v for every known checkpoint
            int numKHeads = (int)m.GetUInt32($"{arch}.ssm.group_count");
            int numVHeads = (int)m.GetUInt32($"{arch}.ssm.time_step_rank");
            int dInner = (int)m.GetUInt32($"{arch}.ssm.inner_size");
            int headVDim = dInner / numVHeads;

            // qwen35moe: every trunk layer carries a routed MoE FFN (256 experts, top-8, SOFTMAX + top-k renorm —
            // hardcoded in llama.cpp, not emitted as metadata) plus an always-on sigmoid-gated shared expert. The
            // hybrid attention path is identical to the dense tier, so only the FFN differs.
            MoeFeedForward?[] moe = new MoeFeedForward?[layers];
            int expertCount = (int)m.GetUInt32($"{arch}.expert_count", 0u);
            if (expertCount > 0)
            {
                int expertFfn = (int)m.GetUInt32($"{arch}.expert_feed_forward_length", 0u);
                if (expertFfn == 0 && w.TryGetValue("blk.0.ffn_up_exps.weight", out Tensor? ux0) && ux0.Shape.Rank == 3)
                    expertFfn = (int)ux0.Shape[1];
                MoeConfig moeCfg = new MoeConfig
                {
                    NumExperts = expertCount,
                    NumExpertsPerTok = (int)m.GetUInt32($"{arch}.expert_used_count", 8u),
                    MoeIntermediateSize = expertFfn,
                    SharedExpertIntermediateSize = (int)m.GetUInt32($"{arch}.expert_shared_feed_forward_length", 0u),
                    Scoring = MoeScoring.Softmax,
                    NormTopKProb = true,
                    FirstDenseLayers = 0,
                    ExpertGroupCount = 0,
                    ExpertGroupUsedCount = 0,
                    RoutedScalingFactor = 1f,
                    Activation = ActivationKind.Silu,
                };
                for (int li = 0; li < layers; li++)
                {
                    MoeFeedForward ff = new MoeFeedForward(moeCfg, dModel, lowVram: false);
                    ff.LoadWeights(BuildMoeLayerWeights(w, li, expertCount, expertFfn, dModel), $"blk.{li}");
                    moe[li] = ff;
                }
            }

            return new Qwen35Model(handle, w, dModel, layers, vocab, eps, fullAttnInterval, recrOverride,
                numHeads, numKvHeads, attnHeadDim, rotaryDim, ropeTheta,
                convK, headKvDim, headVDim, numKHeads, numVHeads, moe);
        }
        catch { handle.Dispose(); throw; }
    }

    private Tensor W(string key) => _w[key];
    private bool Has(string key) => _w.ContainsKey(key);
    private Tensor HeadWeight => _headless ? throw new InvalidOperationException("This Qwen3.5 model was built without an output head.")
        : Has("output.weight") ? W("output.weight") : W("token_embd.weight");   // tied by default

    /// <inheritdoc/>
    public float[] ForwardLastLogits(IBackend backend, IReadOnlyList<int> ids)
    {
        int seq = ids.Count, d = DModel;
        Tensor h = new(new TensorShape(1, seq, d), DType.F32);
        EmbedLookup(h, ids);
        return ForwardFromHidden(backend, h, seq);
    }

    /// <summary>VLM prefill entry point: runs the decoder over a precomputed embedding sequence <paramref name="embeds"/> (text-token embeddings with image embeddings already spliced in) and returns last-position logits, carrying Gated-DeltaNet state/KV/<c>_pos</c> forward identically to the ids-in path. Call <see cref="ResetState"/> first.</summary>
    public float[] ForwardEmbedsLastLogits(IBackend backend, Tensor embeds, int seq)
    {
        int d = DModel;
        Tensor h = new(new TensorShape(1, seq, d), DType.F32);
        Buffer.MemoryCopy((void*)embeds.DataPointer, (void*)h.DataPointer, (long)seq * d * 4, (long)seq * d * 4);
        return ForwardFromHidden(backend, h, seq);
    }

    /// <summary>Shared decode core: consumes an initial hidden state <paramref name="h"/> (owned — disposed here), runs every layer (Gated-DeltaNet or full-attention), final RMSNorm, and projects the last position to vocab logits; the ids-in and embeds-in entry points differ only in how <paramref name="h"/> is seeded.</summary>
    private float[] ForwardFromHidden(IBackend backend, Tensor h, int seq) => ProjectLast(backend, RunTrunk(backend, h, seq), seq);

    private Tensor RunTrunk(IBackend backend, Tensor h, int seq)
    {
        int d = DModel;
        // seq==1 decode: one rope table for the token position, shared by every attention layer
        // (previously rebuilt host-side per layer per token).
        Tensor? stepCos = null, stepSin = null;
        if (seq == 1)
        {
            stepCos = new Tensor(new TensorShape(1, 1, AttnHeadDim), DType.F32);
            stepSin = new Tensor(new TensorShape(1, 1, AttnHeadDim), DType.F32);
            BuildRope(stepCos, stepSin, 1, _pos, AttnHeadDim, RotaryDim, RopeTheta);
        }
        for (int i = 0; i < NumLayers; i++)
        {
            Tensor next = _isLinear[i] ? BlockLinear(backend, h, i, seq) : BlockAttn(backend, h, i, seq, stepCos, stepSin);
            h.Dispose(); h = next;
        }
        stepCos?.Dispose(); stepSin?.Dispose();
        _pos += seq;
        _kvCache.AdvanceLength(seq);

        Tensor normed = new(new TensorShape(1, seq, d), DType.F32);
        backend.RmsNorm(normed, h, W("output_norm.weight"), Eps);
        h.Dispose();
        backend.Sync();
        return normed;
    }

    private float[] ProjectLast(IBackend backend, Tensor normed, int seq)
    {
        int d = DModel;
        using Tensor last = new(new TensorShape(1, 1, d), DType.F32);
        Buffer.MemoryCopy((byte*)normed.DataPointer + (long)(seq - 1) * d * 4, (void*)last.DataPointer, (long)d * 4, (long)d * 4);
        normed.Dispose();
        using Tensor logits = new(new TensorShape(1, 1, VocabSize), DType.F32);
        backend.Linear(logits, last, HeadWeight, null);
        backend.Sync();
        float[] outv = new float[VocabSize];
        fixed (float* o = outv) Buffer.MemoryCopy((void*)logits.DataPointer, o, (long)VocabSize * 4, (long)VocabSize * 4);
        return outv;
    }

    /// <summary>Device-side seq=1 attention step: replaces the host q|gate de-interleave (a per-layer D2H/H2D round-trip) with a SliceTimeRange over the [1,hq,2,hd]-shaped fused projection, and the per-layer host rope-table rebuild with the shared per-token table. Rope runs on [1,1,H,hd]-labeled copies (ApplyRopeSingle's contract); the tiny relabel copies stay on-device.</summary>
    private Tensor BlockAttnDeviceStep(IBackend backend, Tensor hIn, int i, Tensor stepCos, Tensor stepSin)
    {
        int d = DModel, hq = NumHeads, hkv = NumKvHeads, hd = AttnHeadDim, group = hq / hkv;
        string p = $"blk.{i}";
        TensorShape flat = new(1, 1, d);

        Tensor xn = new(flat, DType.F32);
        backend.RmsNormEmitQ8(xn, hIn, W($"{p}.attn_norm.weight"), Eps);

        // Fused [query | gate] projection, shaped [1, hq, 2, hd] so the per-head de-interleave is a
        // device SliceTimeRange (each head's row is [q(hd) | g(hd)] = two time-steps of width hd).
        Tensor qgFull = new(new TensorShape(1, hq, 2, hd), DType.F32);
        backend.Linear(qgFull, xn, W($"{p}.attn_q.weight"), null);
        Tensor qHm = new(new TensorShape(1, hq, 1, hd), DType.F32);
        Tensor gateHm = new(new TensorShape(1, hq, 1, hd), DType.F32);
        backend.SliceTimeRange(qHm, qgFull, 0, 1);
        backend.SliceTimeRange(gateHm, qgFull, 1, 1);
        qgFull.Dispose();

        Tensor k = new(new TensorShape(1, hkv, 1, hd), DType.F32);
        backend.Linear(k, xn, W($"{p}.attn_k.weight"), null);
        Tensor vMh = new(new TensorShape(1, hkv, 1, hd), DType.F32);
        backend.Linear(vMh, xn, W($"{p}.attn_v.weight"), null);
        xn.Dispose();

        // Per-head QK-norm (reduces over the last dim — shape-agnostic), then rope on [1,1,H,hd] labels.
        Tensor qN = new(new TensorShape(1, hq, 1, hd), DType.F32);
        backend.RmsNorm(qN, qHm, W($"{p}.attn_q_norm.weight"), Eps);
        qHm.Dispose();
        Tensor kN = new(new TensorShape(1, hkv, 1, hd), DType.F32);
        backend.RmsNorm(kN, k, W($"{p}.attn_k_norm.weight"), Eps);
        k.Dispose();

        Tensor qRope = new(new TensorShape(1, 1, hq, hd), DType.F32);
        backend.CopyTo(qRope, qN);
        backend.ApplyRopeSingle(qRope, stepCos, stepSin, RotaryDim);
        Tensor qMh = qN;                      // reuse the head-major tensor as the rope destination
        backend.CopyTo(qMh, qRope);
        qRope.Dispose();
        Tensor kRope = new(new TensorShape(1, 1, hkv, hd), DType.F32);
        backend.CopyTo(kRope, kN);
        backend.ApplyRopeSingle(kRope, stepCos, stepSin, RotaryDim);
        Tensor kMh = kN;
        backend.CopyTo(kMh, kRope);
        kRope.Dispose();

        _kvCache.AppendStep(backend, i, kMh, vMh);
        kMh.Dispose(); vMh.Dispose();
        int kvLen = _pos + 1;
        Tensor attn = new(new TensorShape(1, hq, 1, hd), DType.F32);
        backend.FlashAttention(attn, qMh, _kvCache.KeyPrefix(i), _kvCache.ValuePrefix(i), kvLen, group, causal: true, qOffset: _pos, 1f / MathF.Sqrt(hd));
        qMh.Dispose();

        // [1,hq,1,hd] ≡ [1,1,hq*hd] bytes at seq=1 — gate, then the output projection, no permutes.
        Tensor gateSig = new(new TensorShape(1, hq, 1, hd), DType.F32);
        backend.Sigmoid(gateSig, gateHm);
        gateHm.Dispose();
        Tensor gated = new(new TensorShape(1, 1, hq * hd), DType.F32);
        backend.Mul(gated, attn, gateSig);
        attn.Dispose(); gateSig.Dispose();

        Tensor attnOut = new(flat, DType.F32);
        backend.Linear(attnOut, gated, W($"{p}.attn_output.weight"), null);
        gated.Dispose();
        Tensor afterAttn = new(flat, DType.F32);
        Tensor preFfn = new(flat, DType.F32);
        backend.AddRmsNormEmitQ8(afterAttn, preFfn, hIn, attnOut, W($"{p}.post_attention_norm.weight"), Eps);
        attnOut.Dispose();
        Tensor ffnOut = Ffn(backend, preFfn, p, i, 1);
        preFfn.Dispose();
        Tensor result = new(flat, DType.F32);
        backend.Add(result, afterAttn, ffnOut);
        afterAttn.Dispose(); ffnOut.Dispose();
        return result;
    }

    /// <summary>Regular causal self-attention layer: GQA + partial RoPE + KV cache, Qwen3-style per-head QK-norm, and a fused query+gate projection (wq outputs <c>[query | gate]</c>, double width — the gate sigmoid-gates the attention output before <c>wo</c>, unique to this arch's full-attention layers).</summary>
    private Tensor BlockAttn(IBackend backend, Tensor hIn, int i, int seq, Tensor? stepCos = null, Tensor? stepSin = null)
    {
        if (seq == 1 && stepCos is not null && backend.GraphDecodeSupported && EngineKnobs.SsmDeviceStep.Value)
            return BlockAttnDeviceStep(backend, hIn, i, stepCos, stepSin!);
        int d = DModel, hq = NumHeads, hkv = NumKvHeads, hd = AttnHeadDim, group = hq / hkv;
        string p = $"blk.{i}";
        TensorShape flat = new(1, seq, d);

        Tensor xn = new(flat, DType.F32);
        backend.RmsNorm(xn, hIn, W($"{p}.attn_norm.weight"), Eps);

        // wq outputs [query | gate] fused, each hq*hd wide (double width); each head's row is
        // [query_0..query_{hd-1}, gate_0..gate_{hd-1}] — a strided de-interleave, not a flat slice — done
        // host-side since it's a one-time small reshuffle per call, not a hot loop.
        Tensor qgFull = new(new TensorShape(1, seq, hq * hd * 2), DType.F32);
        backend.Linear(qgFull, xn, W($"{p}.attn_q.weight"), null);
        Tensor q = new(new TensorShape(1, seq, hq, hd), DType.F32);
        Tensor gate = new(new TensorShape(1, seq, hq, hd), DType.F32);
        SplitInterleavedQueryGate(qgFull, q, gate, seq, hq, hd);
        qgFull.Dispose();

        Tensor qN = new(q.Shape, DType.F32);
        backend.RmsNorm(qN, q, W($"{p}.attn_q_norm.weight"), Eps);
        q.Dispose();

        Tensor k = new(new TensorShape(1, seq, hkv, hd), DType.F32);
        backend.Linear(k, xn, W($"{p}.attn_k.weight"), null);
        Tensor kN = new(k.Shape, DType.F32);
        backend.RmsNorm(kN, k, W($"{p}.attn_k_norm.weight"), Eps);
        k.Dispose();

        Tensor v = new(new TensorShape(1, seq, hkv, hd), DType.F32);
        backend.Linear(v, xn, W($"{p}.attn_v.weight"), null);
        xn.Dispose();

        Tensor cos = new(new TensorShape(1, seq, hd), DType.F32);
        Tensor sin = new(new TensorShape(1, seq, hd), DType.F32);
        BuildRope(cos, sin, seq, _pos, hd, RotaryDim, RopeTheta);
        backend.ApplyRopeSingle(qN, cos, sin, RotaryDim);
        backend.ApplyRopeSingle(kN, cos, sin, RotaryDim);
        cos.Dispose(); sin.Dispose();

        Tensor qMh = new(new TensorShape(1, hq, seq, hd), DType.F32);
        Tensor kMh = new(new TensorShape(1, hkv, seq, hd), DType.F32);
        Tensor vMh = new(new TensorShape(1, hkv, seq, hd), DType.F32);
        backend.Permute0213(qMh, qN, seq, hq, hd);
        backend.Permute0213(kMh, kN, seq, hkv, hd);
        backend.Permute0213(vMh, v, seq, hkv, hd);
        qN.Dispose(); kN.Dispose(); v.Dispose();

        _kvCache.AppendStep(backend, i, kMh, vMh);
        kMh.Dispose(); vMh.Dispose();
        int kvLen = _pos + seq;
        Tensor attn = new(new TensorShape(1, hq, seq, hd), DType.F32);
        backend.FlashAttention(attn, qMh, _kvCache.KeyPrefix(i), _kvCache.ValuePrefix(i), kvLen, group, causal: true, qOffset: _pos, 1f / MathF.Sqrt(hd));
        qMh.Dispose();

        Tensor attnFlat = new(new TensorShape(1, seq, hq * hd), DType.F32);
        backend.Permute0213(attnFlat, attn, hq, seq, hd);
        attn.Dispose();

        Tensor gateFlat = gate.Reshape(new TensorShape(1, seq, hq * hd));   // view, no copy — same layout already
        Tensor gateSig = new(gateFlat.Shape, DType.F32);
        backend.Sigmoid(gateSig, gateFlat);
        gate.Dispose();
        Tensor gated = new(attnFlat.Shape, DType.F32);
        backend.Mul(gated, attnFlat, gateSig);
        attnFlat.Dispose(); gateSig.Dispose();

        Tensor attnOut = new(flat, DType.F32);
        backend.Linear(attnOut, gated, W($"{p}.attn_output.weight"), null);
        gated.Dispose();
        Tensor afterAttn = new(flat, DType.F32);
        backend.Add(afterAttn, hIn, attnOut);
        attnOut.Dispose();

        Tensor preFfn = new(flat, DType.F32);
        backend.RmsNorm(preFfn, afterAttn, W($"{p}.post_attention_norm.weight"), Eps);
        Tensor ffnOut = Ffn(backend, preFfn, p, i, seq);
        preFfn.Dispose();
        Tensor result = new(flat, DType.F32);
        backend.Add(result, afterAttn, ffnOut);
        afterAttn.Dispose(); ffnOut.Dispose();
        return result;
    }

    /// <summary>Splits the fused <c>[query | gate]</c> projection into two <c>[1,seq,hq,hd]</c> tensors — each head's output row is <c>[query_0..query_{hd-1}, gate_0..gate_{hd-1}]</c>, a per-head strided de-interleave, not a contiguous slice.</summary>
    private static void SplitInterleavedQueryGate(Tensor qgFull, Tensor q, Tensor gate, int seq, int hq, int hd)
    {
        float* src = (float*)qgFull.DataPointer;
        float* qp = (float*)q.DataPointer;
        float* gp = (float*)gate.DataPointer;
        int headStride = hd * 2;
        for (int s = 0; s < seq; s++)
        {
            long rowBase = (long)s * hq * headStride;
            long outBase = (long)s * hq * hd;
            for (int hh = 0; hh < hq; hh++)
            {
                float* srcHead = src + rowBase + hh * headStride;
                Buffer.MemoryCopy(srcHead, qp + outBase + (long)hh * hd, hd * 4, hd * 4);
                Buffer.MemoryCopy(srcHead + hd, gp + outBase + (long)hh * hd, hd * 4, hd * 4);
            }
        }
    }

    private static void BuildRope(Tensor cos, Tensor sin, int t, int posStart, int headDim, int rotaryDim, float theta)
    {
        int rdim = rotaryDim > 0 && rotaryDim < headDim ? rotaryDim : headDim;
        int half = rdim / 2;
        (double[] invFreq, double mscale) = RopeFrequencyBuilder.Build(rdim, theta, RopeScaling.None, posStart + t);
        float* pc = (float*)cos.DataPointer;
        float* ps = (float*)sin.DataPointer;
        for (int s = 0; s < t; s++)
        {
            int pos = posStart + s;
            long baseOff = (long)s * headDim;
            for (int i = 0; i < half; i++)
            {
                double angle = pos * invFreq[i];
                float c = (float)(Math.Cos(angle) * mscale);
                float si = (float)(Math.Sin(angle) * mscale);
                pc[baseOff + i] = c; pc[baseOff + i + half] = c;
                ps[baseOff + i] = si; ps[baseOff + i + half] = si;
            }
        }
    }

    // Graph-decode session state (transformer playbook): full-position rope tables + the F32 embed
    // table device-resident, built once per model lifetime.
    private Tensor? _graphCos, _graphSin;
    private int _graphRopeCapacity;

    /// <summary>Uploads the F32 embed table and builds device rope tables for graph decode.</summary>
    public (Tensor cos, Tensor sin, Tensor embed) EnsureGraphDecodeResident(IBackend backend, int minCapacity)
    {
        Tensor embed = W("__embed_f32");
        backend.PreloadWeights([embed]);
        if (_graphCos is null || _graphRopeCapacity < minCapacity)
        {
            _graphCos?.Dispose(); _graphSin?.Dispose();
            (_graphCos, _graphSin) = backend.BuildRopeTableDevice(minCapacity, AttnHeadDim, RotaryDim, RopeTheta, RopeScaling.None, splitHalfPartial: true);
            _graphRopeCapacity = minCapacity;
        }
        return (_graphCos!, _graphSin!, embed);
    }

    /// <summary>One full graph-capturable decode step: device embed gather → every layer → final norm → head → greedy argmax written back into <paramref name="deviceTokenId"/>. The recurrent conv/delta states and the KV cache mutate in place on their captured device buffers, so replaying the captured graph chains decode entirely on-device.</summary>
    public void ForwardGraphDecodeStep(IBackend backend, Tensor embedTable, Tensor cosTable, Tensor sinTable,
        ulong devicePos, ulong deviceTokenId)
    {
        int d = DModel;
        Tensor h = new(new TensorShape(1, 1, d), DType.F32);
        backend.EmbedGatherDecodeStep(h, embedTable, deviceTokenId);

        for (int i = 0; i < NumLayers; i++)
        {
            Tensor next = _isLinear[i] ? BlockLinearDeviceStep(backend, h, i)
                : BlockAttnGraphStep(backend, h, i, cosTable, sinTable, devicePos);
            h.Dispose(); h = next;
        }

        Tensor normed = new(new TensorShape(1, 1, d), DType.F32);
        backend.RmsNormEmitQ8(normed, h, W("output_norm.weight"), Eps);
        h.Dispose();
        Tensor logits = new(new TensorShape(1, 1, VocabSize), DType.F32);
        backend.Linear(logits, normed, HeadWeight, null);
        normed.Dispose();
        backend.ArgMaxInto(deviceTokenId, logits);
        logits.Dispose();
    }

    /// <summary>Attention layer for the captured step: identical to <see cref="BlockAttnDeviceStep"/> except rope/append/attention read the position from the device buffer (the decode rope kernel is head-major, so the relabel copies disappear too).</summary>
    private Tensor BlockAttnGraphStep(IBackend backend, Tensor hIn, int i, Tensor cosTable, Tensor sinTable, ulong devicePos)
    {
        int d = DModel, hq = NumHeads, hkv = NumKvHeads, hd = AttnHeadDim, group = hq / hkv;
        string p = $"blk.{i}";
        TensorShape flat = new(1, 1, d);

        // EmitQ8/fused-add-norm mirror BlockLinearDeviceStep (self-gated, plain semantics when off).
        Tensor xn = new(flat, DType.F32);
        backend.RmsNormEmitQ8(xn, hIn, W($"{p}.attn_norm.weight"), Eps);

        Tensor qgFull = new(new TensorShape(1, hq, 2, hd), DType.F32);
        backend.Linear(qgFull, xn, W($"{p}.attn_q.weight"), null);
        Tensor qHm = new(new TensorShape(1, hq, 1, hd), DType.F32);
        Tensor gateHm = new(new TensorShape(1, hq, 1, hd), DType.F32);
        backend.SliceTimeRange(qHm, qgFull, 0, 1);
        backend.SliceTimeRange(gateHm, qgFull, 1, 1);
        qgFull.Dispose();

        Tensor k = new(new TensorShape(1, hkv, 1, hd), DType.F32);
        backend.Linear(k, xn, W($"{p}.attn_k.weight"), null);
        Tensor vMh = new(new TensorShape(1, hkv, 1, hd), DType.F32);
        backend.Linear(vMh, xn, W($"{p}.attn_v.weight"), null);
        xn.Dispose();

        Tensor qMh = new(new TensorShape(1, hq, 1, hd), DType.F32);
        backend.RmsNorm(qMh, qHm, W($"{p}.attn_q_norm.weight"), Eps);
        qHm.Dispose();
        Tensor kMh = new(new TensorShape(1, hkv, 1, hd), DType.F32);
        backend.RmsNorm(kMh, k, W($"{p}.attn_k_norm.weight"), Eps);
        k.Dispose();

        backend.RopeApplyDecodeStep(qMh, cosTable, sinTable, RotaryDim, interleaved: false, devicePos);
        backend.RopeApplyDecodeStep(kMh, cosTable, sinTable, RotaryDim, interleaved: false, devicePos);

        Tensor kFull = _kvCache.KeyPrefix(i);
        Tensor vFull = _kvCache.ValuePrefix(i);
        backend.KvCacheAppendDev(kFull, kMh, 0, devicePos);
        backend.KvCacheAppendDev(vFull, vMh, 0, devicePos);
        kMh.Dispose(); vMh.Dispose();

        Tensor attn = new(new TensorShape(1, hq, 1, hd), DType.F32);
        backend.FlashAttentionDev(attn, qMh, kFull, vFull, 0, group, causal: true, 0, 1f / MathF.Sqrt(hd), devicePos, 0f, 0);
        qMh.Dispose();

        Tensor gateSig = new(new TensorShape(1, hq, 1, hd), DType.F32);
        backend.Sigmoid(gateSig, gateHm);
        gateHm.Dispose();
        Tensor gated = new(new TensorShape(1, 1, hq * hd), DType.F32);
        backend.Mul(gated, attn, gateSig);
        attn.Dispose(); gateSig.Dispose();

        Tensor attnOut = new(flat, DType.F32);
        backend.Linear(attnOut, gated, W($"{p}.attn_output.weight"), null);
        gated.Dispose();
        Tensor afterAttn = new(flat, DType.F32);
        Tensor preFfn = new(flat, DType.F32);
        backend.AddRmsNormEmitQ8(afterAttn, preFfn, hIn, attnOut, W($"{p}.post_attention_norm.weight"), Eps);
        attnOut.Dispose();
        Tensor ffnOut = DenseSwiGlu(backend, preFfn, p, 1);
        preFfn.Dispose();
        Tensor result = new(flat, DType.F32);
        backend.Add(result, afterAttn, ffnOut);
        afterAttn.Dispose(); ffnOut.Dispose();
        return result;
    }

    /// <summary>Advances the host position counter after a graph-decode session (the replay loop moved the device position; host bookkeeping must follow for any subsequent prefill).</summary>
    public void AdvancePosition(int tokens)
    {
        _pos += tokens;
        _kvCache.AdvanceLength(tokens);
    }

    public int CurrentPosition => _pos;

    private unsafe void EnsureStateOnDevice(int i)
    {
        if (_stateOnDevice[i]) return;
        int histLen = (ConvKernel - 1) * ConvDim, stLen = NumVHeads * HeadVDim * HeadVDim;
        _convHistT[i] ??= new Tensor(new TensorShape(histLen), DType.F32);
        _deltaStateT[i] ??= new Tensor(new TensorShape(stLen), DType.F32);
        fixed (float* src = _convHistory[i])
            Buffer.MemoryCopy(src, (void*)_convHistT[i]!.DataPointer, (long)histLen * 4, (long)histLen * 4);
        fixed (float* src = _deltaState[i])
            Buffer.MemoryCopy(src, (void*)_deltaStateT[i]!.DataPointer, (long)stLen * 4, (long)stLen * 4);
        _stateOnDevice[i] = true;
    }

    /// <summary>Device-side seq=1 decode step for a gated-DeltaNet layer — replaces the per-layer host conv/recurrence/gated-norm round-trip (a per-layer Sync + PCIe ping-pong that capped decode at ~26 tok/s). Reductions reorder float sums vs the host loops, so outputs are tolerance-equivalent (dp4a numerics class), not byte-identical.</summary>
    private Tensor BlockLinearDeviceStep(IBackend backend, Tensor hIn, int i)
    {
        int d = DModel;
        string p = $"blk.{i}";
        TensorShape flat = new(1, 1, d);
        EnsureStateOnDevice(i);

        // EmitQ8: one Q8_1 sidecar from the norm feeds all four projections' dp4a GEMVs
        // (self-gated — plain RmsNorm semantics when quantize-at-producer is off).
        Tensor xn = new(flat, DType.F32);
        backend.RmsNormEmitQ8(xn, hIn, W($"{p}.attn_norm.weight"), Eps);
        Tensor qkvMixed = new(new TensorShape(1, 1, ConvDim), DType.F32);
        backend.Linear(qkvMixed, xn, W($"{p}.attn_qkv.weight"), null);
        Tensor z = new(new TensorShape(1, 1, ValueDim), DType.F32);
        backend.Linear(z, xn, W($"{p}.attn_gate.weight"), null);
        Tensor betaRaw = new(new TensorShape(1, 1, NumVHeads), DType.F32);
        backend.Linear(betaRaw, xn, W($"{p}.ssm_beta.weight"), null);
        Tensor alphaRaw = new(new TensorShape(1, 1, NumVHeads), DType.F32);
        backend.Linear(alphaRaw, xn, W($"{p}.ssm_alpha.weight"), null);
        xn.Dispose();

        Tensor q = new(new TensorShape(KeyDim), DType.F32);
        Tensor k = new(new TensorShape(KeyDim), DType.F32);
        Tensor v = new(new TensorShape(ValueDim), DType.F32);
        backend.SsmConvSplitStep(q, k, v, qkvMixed, _convHistT[i]!, W($"{p}.ssm_conv1d.weight"),
            ConvKernel, NumKHeads, HeadKDim, NumVHeads, HeadVDim, Eps, 1f / MathF.Sqrt(HeadKDim));
        qkvMixed.Dispose();

        Tensor oT = new(new TensorShape(1, 1, ValueDim), DType.F32);
        backend.SsmDeltaStep(oT, _deltaStateT[i]!, q, k, v, z,
            alphaRaw, betaRaw, W($"{p}.ssm_dt.bias"), W($"{p}.ssm_a"), W($"{p}.ssm_norm.weight"),
            NumVHeads, HeadVDim, HeadKDim, KHeadsPerVHead, Eps);
        q.Dispose(); k.Dispose(); v.Dispose(); z.Dispose(); alphaRaw.Dispose(); betaRaw.Dispose();

        Tensor attnOut = new(flat, DType.F32);
        backend.Linear(attnOut, oT, W($"{p}.ssm_out.weight"), null);
        oT.Dispose();
        // Fused residual add + pre-FFN norm, sidecar shared by the gate/up projections.
        Tensor afterAttn = new(flat, DType.F32);
        Tensor preFfn = new(flat, DType.F32);
        backend.AddRmsNormEmitQ8(afterAttn, preFfn, hIn, attnOut, W($"{p}.post_attention_norm.weight"), Eps);
        attnOut.Dispose();
        Tensor mlpOut = Ffn(backend, preFfn, p, i, 1);
        preFfn.Dispose();
        Tensor result = new(flat, DType.F32);
        backend.Add(result, afterAttn, mlpOut);
        afterAttn.Dispose(); mlpOut.Dispose();
        return result;
    }

    /// <summary>Gated DeltaNet layer over a token sequence: fused QKV projection → causal depthwise Conv1d+SiLU → per-head L2-norm on Q/K → delta-rule recurrence per token → gated RMSNorm(o, silu(z)) → output projection + residual. The conv and recurrence run as the backend's <see cref="IBackend.SsmConvSplitPrefill"/>/<see cref="IBackend.SsmDeltaPrefill"/> (device kernels on CUDA and Vulkan, the parallel host reference elsewhere) against the layer's carried state, so a prefill never leaves the device; <c>numerics.ssmDevicePrefill=0</c> forces the host reference.</summary>
    private Tensor BlockLinear(IBackend backend, Tensor hIn, int i, int seq)
    {
        if (seq == 1 && backend.GraphDecodeSupported && EngineKnobs.SsmDeviceStep.Value)   // kill-switch
            return BlockLinearDeviceStep(backend, hIn, i);
        EnsureStateOnDevice(i);
        int d = DModel;
        string p = $"blk.{i}";
        TensorShape flat = new(1, seq, d);

        // Scoped so a throwing backend op leaves no device tensor behind; each scope ends where the old code disposed.
        using Tensor afterAttn = new(flat, DType.F32);
        using (Tensor oT = new(new TensorShape(1, seq, ValueDim), DType.F32))
        {
            using (Tensor qkvMixed = new(new TensorShape(1, seq, ConvDim), DType.F32), z = new(new TensorShape(1, seq, ValueDim), DType.F32),
                betaRaw = new(new TensorShape(1, seq, NumVHeads), DType.F32), alphaRaw = new(new TensorShape(1, seq, NumVHeads), DType.F32))
            {
                using (Tensor xn = new(flat, DType.F32))
                {
                    backend.RmsNorm(xn, hIn, W($"{p}.attn_norm.weight"), Eps);
                    backend.Linear(qkvMixed, xn, W($"{p}.attn_qkv.weight"), null);
                    backend.Linear(z, xn, W($"{p}.attn_gate.weight"), null);
                    backend.Linear(betaRaw, xn, W($"{p}.ssm_beta.weight"), null);
                    backend.Linear(alphaRaw, xn, W($"{p}.ssm_alpha.weight"), null);
                }

                // q *= 1/sqrt(S_k) after the L2 norm — llama.cpp's build_delta_net_autoregressive applies it right before the
                // recurrence. Missing it scales the readout ~sqrt(HeadKDim)x too large, compounding across every linear layer.
                using Tensor q = new(new TensorShape(seq, KeyDim), DType.F32);
                using Tensor k = new(new TensorShape(seq, KeyDim), DType.F32);
                using Tensor v = new(new TensorShape(seq, ValueDim), DType.F32);
                float qScale = 1f / MathF.Sqrt(HeadKDim);
                if (EngineKnobs.SsmDevicePrefill.Value)
                {
                    backend.SsmConvSplitPrefill(q, k, v, qkvMixed, _convHistT[i]!, W($"{p}.ssm_conv1d.weight"),
                        ConvKernel, NumKHeads, HeadKDim, NumVHeads, HeadVDim, Eps, qScale);
                    backend.SsmDeltaPrefill(oT, _deltaStateT[i]!, q, k, v, z, alphaRaw, betaRaw, W($"{p}.ssm_dt.bias"), W($"{p}.ssm_a"),
                        W($"{p}.ssm_norm.weight"), NumVHeads, HeadVDim, HeadKDim, KHeadsPerVHead, Eps);
                }
                else
                {
                    GatedDeltaNetReference.ConvSplit(q, k, v, qkvMixed, _convHistT[i]!, W($"{p}.ssm_conv1d.weight"),
                        ConvKernel, NumKHeads, HeadKDim, NumVHeads, HeadVDim, Eps, qScale);
                    GatedDeltaNetReference.DeltaScan(oT, _deltaStateT[i]!, q, k, v, z, alphaRaw, betaRaw, W($"{p}.ssm_dt.bias"), W($"{p}.ssm_a"),
                        W($"{p}.ssm_norm.weight"), NumVHeads, HeadVDim, HeadKDim, KHeadsPerVHead, Eps);
                }
            }
            using Tensor attnOut = new(flat, DType.F32);
            backend.Linear(attnOut, oT, W($"{p}.ssm_out.weight"), null);
            backend.Add(afterAttn, hIn, attnOut);
        }

        Tensor ffnOut;
        using (Tensor preFfn = new(flat, DType.F32))
        {
            backend.RmsNorm(preFfn, afterAttn, W($"{p}.post_attention_norm.weight"), Eps);
            ffnOut = Ffn(backend, preFfn, p, i, seq);
        }
        using (ffnOut)
        {
            Tensor result = new(flat, DType.F32);
            try
            {
                backend.Add(result, afterAttn, ffnOut);
                return result;
            }
            catch
            {
                result.Dispose();
                throw;
            }
        }
    }

    /// <summary>Per-layer FFN dispatch: the routed MoE block on a qwen35moe layer, else the dense SwiGLU.</summary>
    private Tensor Ffn(IBackend backend, Tensor preFfn, string p, int layer, int seq)
        => _moe[layer] is { } ff ? ff.Forward(backend, preFfn, seq) : DenseSwiGlu(backend, preFfn, p, seq);

    /// <summary>Dense SwiGLU FFN: <c>down(silu(gate(x)) · up(x))</c>; the dense tier (0.8B-9B) uses it on every layer, the MoE tier (<c>qwen35moe</c>) only if a layer is not MoE (none are, currently).</summary>
    private Tensor DenseSwiGlu(IBackend backend, Tensor preFfn, string p, int seq)
    {
        int ffWidth = (int)W($"{p}.ffn_up.weight").Shape[0];   // Shape[0] = outDim (post-relabel [out,in])
        TensorShape ff = new(1, seq, ffWidth);
        Tensor up = new(ff, DType.F32);
        backend.Linear(up, preFfn, W($"{p}.ffn_up.weight"), null);
        Tensor gate = new(ff, DType.F32);
        backend.Linear(gate, preFfn, W($"{p}.ffn_gate.weight"), null);
        Tensor gateAct = new(ff, DType.F32);
        backend.Silu(gateAct, gate);
        gate.Dispose();
        Tensor comb = new(ff, DType.F32);
        backend.Mul(comb, gateAct, up);
        gateAct.Dispose(); up.Dispose();
        Tensor outp = new(new TensorShape(1, seq, DModel), DType.F32);
        backend.Linear(outp, comb, W($"{p}.ffn_down.weight"), null);
        comb.Dispose();
        return outp;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _kvCache.Dispose();
        _handle?.Dispose();
        foreach (Tensor t in _ownedTensors) t.Dispose();
    }
}
