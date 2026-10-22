using System.Collections.Immutable;
using System.Security.Cryptography;
using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.LLM.Looped;
using HartsyInference.LLM.Multimodal;
using HartsyInference.LLM.Ssm;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Tests.Attention;

/// <summary>Forward passes of the LLM attention code (GenericTransformer eager, batched and graph decode, MLA, tensor parallel, looped nGPT, Qwen3.5, Mllama cross-attention) on tiny name-seeded weights, each
/// reduced to SHA-256 digests of its output bits: the per-layer hidden states, the final hidden states, the logits, the greedy token stream and the committed K/V rows of every cache. Everything runs on
/// the CPU backend (graph-decode steps through <see cref="DevicePositionShimBackend"/>), so the same cases produce the golden fixture from the pre-migration source and must reproduce it after the
/// attention refactor. Weights come from the per-case-salted <c>Weights</c> harness, never the static-<c>_rng</c> helpers of the older LLM tests (their weights depend on test order).</summary>
internal static unsafe class AttentionWaveDGoldenCases
{
    private const int Vocab = 24, Hidden = 16, Inter = 24, Capacity = 16, DecodeSteps = 5;
    private static readonly int[] Prompt = [3, 7, 1, 12, 5, 9];
    private static readonly int[] PromptB = [11, 2, 20];

    private enum CacheKind { Fixed, Growing, Paged, Streaming }

    /// <summary>Every case, in fixture order.</summary>
    public static IReadOnlyList<string> Names { get; } =
    [
        "gt_llama_fixed", "gt_mha_kvcache", "gt_mqa_fixed", "gt_gqa4_paged", "gt_qwen3_qknorm_kv", "gt_qwen2_bias_partial_rope", "gt_interleaved_rope_partial",
        "gt_granite_scale", "gt_gemma2_window_softcap", "gt_window1", "gt_window_exceeds_sequence", "gt_gemma3_dual_rope", "gt_gemma4_share_vnorm_swa",
        "gt_olmoe_fullnorm", "gt_stablelm_layernorm_qk", "gt_gptoss_sink_window", "gt_alibi_mha", "gt_alibi_gqa", "gt_cohere_parallel_nope", "gt_muse_gate_headnorm",
        "gt_mixed_dtype_v", "gt_streaming_kv", "gt_chunked_prefill_truncate", "gt_chunked_window_softcap_paged",
        "gt_mla_direct", "gt_mla_qlora",
        "gt_batch_b2_fixed", "gt_batch_b2_paged", "gt_batch_b2_alibi",
        "gt_graph_fused", "gt_graph_qknorm", "gt_graph_fullnorm", "gt_graph_composed_vnorm", "gt_graph_composed_layernorm", "gt_graph_window_softcap",
        "gt_graph_dual_rope", "gt_graph_qonly_swa", "gt_graph_mixed_v", "gt_graph_mixed_v_qknorm", "gt_dual",
        "tp_deg2_qknorm", "tp_deg2_bias", "tp_deg2_mha",
        "looped_perloop", "looped_lastloop", "looped_batch", "looped_graph",
        "mllama_cross_layer", "mllama_cross_in_transformer",
        "qwen35_attn_prefill_decode", "qwen35_device_step", "qwen35_graph_step",
    ];

    /// <summary>Runs <paramref name="name"/> and returns its digests as space-separated <c>label=hex</c> pairs.</summary>
    public static string Run(string name) => name switch
    {
        "gt_mla_direct" or "gt_mla_qlora" => Mla(name),
        "gt_batch_b2_fixed" or "gt_batch_b2_paged" or "gt_batch_b2_alibi" => Batch(name),
        "gt_dual" => Dual(),
        _ when name.StartsWith("gt_graph_", StringComparison.Ordinal) => Graph(name),
        _ when name.StartsWith("gt_", StringComparison.Ordinal) => Eager(name),
        _ when name.StartsWith("tp_", StringComparison.Ordinal) => TensorParallel(name),
        _ when name.StartsWith("looped_", StringComparison.Ordinal) => Looped(name),
        "mllama_cross_layer" => MllamaLayer(),
        "mllama_cross_in_transformer" => MllamaInTransformer(),
        _ when name.StartsWith("qwen35_", StringComparison.Ordinal) => Qwen35(name),
        _ => throw new ArgumentException($"Unknown attention golden case '{name}'.", nameof(name)),
    };

    // ── GenericTransformer configs ─────────────────────────────────────────────────────────────────────────────────

    private static TransformerConfig Base() => new()
    {
        HiddenSize = Hidden, NumLayers = 4, NumHeads = 4, NumKvHeads = 2, HeadDim = 8, IntermediateSize = Inter, VocabSize = Vocab,
        MaxPositionEmbeddings = 64, RopeTheta = 10_000f, TieWordEmbeddings = false,
    };

    /// <summary>The config, cache kind and weight salt of an eager <c>gt_*</c> case.</summary>
    private static (TransformerConfig Cfg, CacheKind Cache, uint Salt) EagerSpec(string name) => name switch
    {
        "gt_llama_fixed" => (Base(), CacheKind.Fixed, 101),
        "gt_mha_kvcache" => (Base() with { NumKvHeads = 4 }, CacheKind.Growing, 102),
        "gt_mqa_fixed" => (Base() with { NumKvHeads = 1 }, CacheKind.Fixed, 103),
        "gt_gqa4_paged" => (Base() with { NumHeads = 8, NumKvHeads = 2 }, CacheKind.Paged, 104),
        "gt_qwen3_qknorm_kv" => (Base() with { QkNorm = true }, CacheKind.Growing, 1105),
        "gt_qwen2_bias_partial_rope" => (Base() with { AttentionBias = true, RotaryDim = 4 }, CacheKind.Fixed, 106),
        "gt_interleaved_rope_partial" => (Base() with { Rope = RopeStyle.Interleaved, RotaryDim = 4, QkNorm = true }, CacheKind.Fixed, 107),
        "gt_granite_scale" => (Base() with { AttentionMultiplier = 0.2f, ResidualMultiplier = 0.5f, LogitScale = 2f }, CacheKind.Fixed, 1108),
        "gt_gemma2_window_softcap" => (Base() with
        {
            SlidingWindowPattern = 2, SlidingWindow = 3, AttnLogitSoftcap = 20f, SandwichNorm = true, QueryPreAttnScalar = 12f,
            FinalLogitSoftcap = 30f, Activation = ActivationKind.GeluTanh,
        }, CacheKind.Fixed, 109),
        "gt_window1" => (Base() with { SlidingWindowPattern = 2, SlidingWindow = 1 }, CacheKind.Fixed, 110),
        "gt_window_exceeds_sequence" => (Base() with { SlidingWindowPattern = 2, SlidingWindow = 64 }, CacheKind.Growing, 111),
        "gt_gemma3_dual_rope" => (Base() with
        {
            NumLayers = 6, SlidingWindowPattern = 3, SlidingWindow = 4, RopeLocalTheta = 1_000f, QkNorm = true, SandwichNorm = true,
            Activation = ActivationKind.GeluTanh,
        }, CacheKind.Fixed, 112),
        "gt_gemma4_share_vnorm_swa" => (Gemma4(), CacheKind.Fixed, 113),
        "gt_olmoe_fullnorm" => (Base() with { QkNorm = true, QkNormFullDim = true }, CacheKind.Fixed, 114),
        "gt_stablelm_layernorm_qk" => (Base() with { UseLayerNorm = true, QkNorm = true, RotaryDim = 4, AttentionBias = true }, CacheKind.Growing, 115),
        "gt_gptoss_sink_window" => (Base() with { AttnSink = true, SlidingWindowPattern = 2, SlidingWindow = 3, AttentionBias = true }, CacheKind.Fixed, 116),
        "gt_alibi_mha" => (Base() with { NumKvHeads = 4, AlibiMaxBias = 8f, AlibiSlopes = TransformerConfig.ComputeAlibiSlopes(4, 8f) }, CacheKind.Fixed, 1117),
        "gt_alibi_gqa" => (Base() with { AlibiMaxBias = 8f, AlibiSlopes = TransformerConfig.ComputeAlibiSlopes(4, 8f) }, CacheKind.Growing, 118),
        "gt_cohere_parallel_nope" => (Base() with
        {
            ParallelResidual = true, UseLayerNorm = true, NoRopeOnGlobalLayers = true, SlidingWindowPattern = 2, SlidingWindow = 3,
        }, CacheKind.Fixed, 119),
        "gt_muse_gate_headnorm" => (Base() with
        {
            AttnOutputGate = true, AttnHeadOutputNorm = true, ResidualGates = true, SandwichNorm = true, PostAttnNormWeightless = true,
            PostFfnNormScalesInput = true,
        }, CacheKind.Fixed, 120),
        "gt_mixed_dtype_v" => (Base() with { AttentionBias = true }, CacheKind.Fixed, 121),
        "gt_streaming_kv" => (Base() with { AttentionBias = true }, CacheKind.Streaming, 1122),
        "gt_chunked_prefill_truncate" => (Base() with { QkNorm = true }, CacheKind.Fixed, 123),
        "gt_chunked_window_softcap_paged" => (Base() with { SlidingWindowPattern = 2, SlidingWindow = 3, AttnLogitSoftcap = 20f }, CacheKind.Paged, 124),
        _ => throw new ArgumentException($"Unknown eager case '{name}'.", nameof(name)),
    };

    /// <summary>Gemma-4 shape: narrower local layers, layers 2 and 3 reading the K/V of layers 1 and 0, weightless V norm.</summary>
    private static TransformerConfig Gemma4() => Base() with
    {
        HeadDimSwa = 4, KvSharedFromLayer = 2, VNorm = true, QkNorm = true, GlobalLayers = new[] { false, true, true, false }, SlidingWindow = 3,
        SandwichNorm = true, RopeLocalTheta = 1_000f, Activation = ActivationKind.GeluTanh,
    };

    // ── GenericTransformer eager prefill + decode ──────────────────────────────────────────────────────────────────

    private static string Eager(string name)
    {
        (TransformerConfig cfg, CacheKind kind, uint salt) = EagerSpec(name);
        using Weights w = ModelWeights(cfg, salt);
        if (name == "gt_mixed_dtype_v") w.ToF16("self_attn.v_proj.weight");
        using CpuBackend cpu = new();
        using GenericTransformer model = new(cfg);
        model.LoadWeights(w, "model");
        using Caches owned = new();
        IKvCache cache = owned.New(kind, cfg);
        Digests d = new();
        if (name.StartsWith("gt_chunked_", StringComparison.Ordinal)) RunChunked(d, model, cpu, cache, cfg);
        else RunEager(d, model, cpu, cache, cfg);
        DigestCache(d, cache, "", cfg);
        return d.Finish();
    }

    /// <summary>Chunked prefill (4 + 2 tokens, so the second chunk has several rows at a non-zero offset), two decodes, a rollback of three tokens, a three-token verify chunk at the rolled-back
    /// offset, then two more decodes: every step passes a <c>posStart</c> that differs from the chunk's row count.</summary>
    private static void RunChunked(Digests d, GenericTransformer model, IBackend backend, IKvCache cache, TransformerConfig cfg)
    {
        List<int> tokens = [];
        int[][] chunks = [Prompt[..4], Prompt[4..], [], [], [], [2, 17, 8], [], []];
        for (int step = 0; step < chunks.Length; step++)
        {
            if (step == 4) cache.Truncate(cache.CurrentLength - 3);
            if (step == 5) cache.Truncate(cache.CurrentLength);   // a no-op rollback to the same length
            int[] ids = chunks[step].Length > 0 ? chunks[step] : [tokens[^1]];
            int t = ids.Length;
            using Tensor embeds = new(new TensorShape(1, t, cfg.HiddenSize), DType.F32);
            model.EmbedLookup(embeds, ids);
            using Tensor hidden = model.ForwardEmbeds(backend, embeds, t, cache.CurrentLength, cache, tokenIds: ids, layerTap: (i, tap) => d.Add($"l{i}", tap));
            using Tensor logits = model.ProjectLogits(backend, hidden, t);
            d.Add("hid", hidden);
            d.Add("logits", logits);
            tokens.Add(ArgMaxLastRow(logits, t));
        }
        d.Plain("toks", string.Join(',', tokens));
    }

    /// <summary>Prefill the prompt, then <see cref="DecodeSteps"/> single-token greedy decodes; digests the layer taps, hidden, logits and tokens.</summary>
    private static void RunEager(Digests d, GenericTransformer model, IBackend backend, IKvCache cache, TransformerConfig cfg)
    {
        List<int> tokens = [];
        int[] ids = Prompt;
        for (int step = 0; step <= DecodeSteps; step++)
        {
            int t = ids.Length;
            using Tensor embeds = new(new TensorShape(1, t, cfg.HiddenSize), DType.F32);
            model.EmbedLookup(embeds, ids);
            using Tensor hidden = model.ForwardEmbeds(backend, embeds, t, cache.CurrentLength, cache, tokenIds: ids, layerTap: (i, tap) => d.Add($"l{i}", tap));
            using Tensor logits = model.ProjectLogits(backend, hidden, t);
            d.Add("hid", hidden);
            d.Add("logits", logits);
            int next = ArgMaxLastRow(logits, t);
            tokens.Add(next);
            ids = [next];
        }
        d.Plain("toks", string.Join(',', tokens));
    }

    // ── MLA ────────────────────────────────────────────────────────────────────────────────────────────────────────

    private static string Mla(string name)
    {
        bool qLora = name == "gt_mla_qlora";
        MlaConfig mla = new() { KvLoraRank = 8, QLoraRank = qLora ? 12 : 0, QkNopeHeadDim = 4, QkRopeHeadDim = 2, VHeadDim = 4, AttnScale = qLora ? 0.3f : 0f };
        TransformerConfig cfg = Base() with { NumLayers = 2, NumKvHeads = 4, HeadDim = mla.QkHeadDim, Mla = mla };
        using Weights w = new(qLora ? 202u : 201u);
        w.Add("model.embed_tokens.weight", Vocab, Hidden);
        w.Add("lm_head.weight", Vocab, Hidden);
        w.AddNorm("model.norm.weight", Hidden);
        for (int i = 0; i < cfg.NumLayers; i++)
        {
            string p = $"model.layers.{i}";
            w.AddNorm($"{p}.input_layernorm.weight", Hidden);
            w.AddNorm($"{p}.post_attention_layernorm.weight", Hidden);
            if (qLora)
            {
                w.Add($"{p}.self_attn.q_a_proj.weight", mla.QLoraRank, Hidden);
                w.AddNorm($"{p}.self_attn.q_a_norm.weight", mla.QLoraRank);
                w.Add($"{p}.self_attn.q_b_proj.weight", cfg.NumHeads * mla.QkHeadDim, mla.QLoraRank);
            }
            else
            {
                w.Add($"{p}.self_attn.q_proj.weight", cfg.NumHeads * mla.QkHeadDim, Hidden);
            }
            w.Add($"{p}.self_attn.kv_a_proj.weight", mla.KvLoraRank + mla.QkRopeHeadDim, Hidden);
            w.AddNorm($"{p}.self_attn.kv_a_norm.weight", mla.KvLoraRank);
            w.Add($"{p}.self_attn.kv_b_proj.weight", cfg.NumHeads * (mla.QkNopeHeadDim + mla.VHeadDim), mla.KvLoraRank);
            w.Add($"{p}.self_attn.o_proj.weight", Hidden, cfg.NumHeads * mla.VHeadDim);
            AddMlp(w, p);
        }
        using CpuBackend cpu = new();
        using GenericTransformer model = new(cfg);
        model.LoadWeights(w, "model");
        using FixedKvCache cache = new(cfg.NumLayers, 1, cfg.NumKvHeads, cfg.HeadDim, Capacity);
        Digests d = new();
        RunEager(d, model, cpu, cache, cfg);
        DigestCache(d, cache, "", cfg);
        return d.Finish();
    }

    // ── Batched decode ─────────────────────────────────────────────────────────────────────────────────────────────

    private static string Batch(string name)
    {
        (TransformerConfig cfg, CacheKind kind, uint salt) = name switch
        {
            "gt_batch_b2_fixed" => (Base() with
            {
                QkNorm = true, AttentionBias = true, SlidingWindowPattern = 2, SlidingWindow = 3, AttnLogitSoftcap = 20f,
            }, CacheKind.Fixed, 301u),
            "gt_batch_b2_paged" => (Base() with { AttnSink = true, SlidingWindowPattern = 2, SlidingWindow = 4 }, CacheKind.Paged, 302u),
            _ => (Base() with { NumKvHeads = 4, AlibiMaxBias = 8f, AlibiSlopes = TransformerConfig.ComputeAlibiSlopes(4, 8f) }, CacheKind.Growing, 303u),
        };
        using Weights w = ModelWeights(cfg, salt);
        using CpuBackend cpu = new();
        using GenericTransformer model = new(cfg);
        model.LoadWeights(w, "model");
        using Caches owned = new();
        IKvCache[] caches = [owned.New(kind, cfg), owned.New(kind, cfg)];
        Digests d = new();
        int[][] prompts = [Prompt[..5], PromptB];
        int[] next = new int[2];
        for (int s = 0; s < 2; s++)
        {
            int t = prompts[s].Length;
            using Tensor embeds = new(new TensorShape(1, t, cfg.HiddenSize), DType.F32);
            model.EmbedLookup(embeds, prompts[s]);
            using Tensor hidden = model.ForwardEmbeds(cpu, embeds, t, 0, caches[s], tokenIds: prompts[s]);
            using Tensor logits = model.ProjectLogits(cpu, hidden, t);
            d.Add($"prefill{s}", logits);
            next[s] = ArgMaxLastRow(logits, t);
        }
        List<string> tokens = [$"{next[0]}/{next[1]}"];
        for (int step = 0; step < 4; step++)
        {
            using Tensor embeds = new(new TensorShape(1, 2, cfg.HiddenSize), DType.F32);
            model.EmbedLookup(embeds, next);
            int[] positions = [caches[0].CurrentLength, caches[1].CurrentLength];
            using Tensor hidden = model.ForwardBatchDecode(cpu, embeds, positions, caches);
            using Tensor logits = model.ProjectLogits(cpu, hidden, 2);
            d.Add("hid", hidden);
            d.Add("logits", logits);
            int vocab = (int)(logits.ElementCount / 2);
            next = [ArgMaxRow(logits, 0, vocab), ArgMaxRow(logits, 1, vocab)];
            tokens.Add($"{next[0]}/{next[1]}");
        }
        d.Plain("toks", string.Join(',', tokens));
        DigestCache(d, caches[0], "a", cfg);
        DigestCache(d, caches[1], "b", cfg);
        return d.Finish();
    }

    // ── Graph decode through the device-position shim ──────────────────────────────────────────────────────────────

    private static (TransformerConfig Cfg, uint Salt, string? F16Weight) GraphSpec(string name) => name switch
    {
        "gt_graph_fused" => (Base() with { AttentionBias = true }, 401, null),
        "gt_graph_qknorm" => (Base() with { QkNorm = true, NumHeads = 8, NumKvHeads = 2 }, 402, null),
        "gt_graph_fullnorm" => (Base() with { QkNorm = true, QkNormFullDim = true }, 403, null),
        "gt_graph_composed_vnorm" => (Base() with { VNorm = true }, 404, null),
        "gt_graph_composed_layernorm" => (Base() with { UseLayerNorm = true, QkNorm = true }, 405, null),
        "gt_graph_window_softcap" => (Base() with
        {
            SlidingWindowPattern = 2, SlidingWindow = 3, AttnLogitSoftcap = 20f, SandwichNorm = true, QueryPreAttnScalar = 12f,
            Activation = ActivationKind.GeluTanh,
        }, 406, null),
        "gt_graph_dual_rope" => (Base() with
        {
            NumLayers = 6, SlidingWindowPattern = 3, SlidingWindow = 4, RopeLocalTheta = 1_000f, QkNorm = true, SandwichNorm = true,
            Activation = ActivationKind.GeluTanh,
        }, 407, null),
        "gt_graph_qonly_swa" => (Gemma4(), 408, null),
        "gt_graph_mixed_v" => (Base() with { RotaryDim = 4 }, 1409, "self_attn.v_proj.weight"),
        "gt_graph_mixed_v_qknorm" => (Base() with { QkNorm = true }, 1410, "self_attn.v_proj.weight"),
        _ => throw new ArgumentException($"Unknown graph case '{name}'.", nameof(name)),
    };

    private static string Graph(string name)
    {
        (TransformerConfig cfg, uint salt, string? f16) = GraphSpec(name);
        using Weights w = ModelWeights(cfg, salt);
        if (f16 is not null) w.ToF16(f16);
        using DevicePositionShimBackend shim = new();
        using GenericTransformer model = new(cfg);
        model.LoadWeights(w, "model");
        int[] tokens = GraphDecode(model, shim, cfg, out string digests);
        return $"{digests} toks={string.Join(',', tokens)}";
    }

    /// <summary>The token stream of graph case <paramref name="name"/> next to the stream the same weights give through the eager forward on the same backend; they agree when the shim is faithful.</summary>
    internal static (int[] Graph, int[] Eager) GraphAndEagerTokens(string name)
    {
        (TransformerConfig cfg, uint salt, string? f16) = GraphSpec(name);
        using Weights w = ModelWeights(cfg, salt);
        if (f16 is not null) w.ToF16(f16);
        using DevicePositionShimBackend shim = new();
        using GenericTransformer model = new(cfg);
        model.LoadWeights(w, "model");
        int[] graph = GraphDecode(model, shim, cfg, out string _);
        using FixedKvCache cache = new(cfg.NumLayers, 1, cfg.NumKvHeads, cfg.HeadDimsPerLayer(), Capacity);
        List<int> eager = [Prefill(model, shim, cache, cfg, Prompt)];
        for (int step = 0; step < DecodeSteps; step++) eager.Add(Prefill(model, shim, cache, cfg, [eager[^1]]));
        return (graph, [.. eager]);
    }

    /// <summary>Eager prefill on the shim, then five chained graph-decode steps (<c>ForwardGraphDecodeStep</c>, the device token id carrying the greedy stream); returns the tokens and the digests of
    /// the committed K/V plus the per-step hidden states of a second pass through <c>ForwardGraphDecodeStepEmbeds</c> over the same token stream.</summary>
    private static int[] GraphDecode(GenericTransformer model, DevicePositionShimBackend shim, TransformerConfig cfg, out string digests)
    {
        Digests d = new();
        using FixedKvCache cache = new(cfg.NumLayers, 1, cfg.NumKvHeads, cfg.HeadDimsPerLayer(), Capacity);
        List<int> tokens = [Prefill(model, shim, cache, cfg, Prompt)];
        Tensor embedTable = model.EnsureEmbedResidentForGraphDecode(shim);
        (Tensor cos, Tensor sin) = model.EnsureRopeTableForGraphDecode(shim, Capacity);
        ulong devicePos = shim.AllocDevicePos(), tokenId = shim.AllocDeviceTokenId();
        for (int step = 0; step < DecodeSteps; step++)
        {
            int pos = cache.CurrentLength;
            shim.WriteDeviceTokenId(tokenId, tokens[^1]);
            shim.WriteDevicePos(devicePos, pos + 1, pos);
            model.ForwardGraphDecodeStep(shim, embedTable, cache, cos, sin, devicePos, tokenId);
            cache.AdvanceLength(1);
            tokens.Add(shim.ReadDeviceTokenId(tokenId));
        }
        DigestCache(d, cache, "", cfg);

        using FixedKvCache second = new(cfg.NumLayers, 1, cfg.NumKvHeads, cfg.HeadDimsPerLayer(), Capacity);
        Prefill(model, shim, second, cfg, Prompt);
        using Tensor inEmbed = new(new TensorShape(1, 1, cfg.HiddenSize), DType.F32);
        using Tensor outHidden = new(new TensorShape(1, 1, cfg.HiddenSize), DType.F32);
        for (int step = 0; step < DecodeSteps; step++)
        {
            int pos = second.CurrentLength;
            model.EmbedLookup(inEmbed, [tokens[step]]);
            shim.WriteDevicePos(devicePos, pos + 1, pos);
            model.ForwardGraphDecodeStepEmbeds(shim, inEmbed, second, cos, sin, devicePos, outHidden);
            second.AdvanceLength(1);
            d.Add("hid", outHidden);
        }
        shim.FreeDevicePos(devicePos);
        shim.FreeDeviceTokenId(tokenId);
        digests = d.Finish();
        return [.. tokens];
    }

    /// <summary>Eager prefill of <paramref name="ids"/> into <paramref name="cache"/>; returns the greedy token after the last row.</summary>
    private static int Prefill(GenericTransformer model, IBackend backend, IKvCache cache, TransformerConfig cfg, int[] ids)
    {
        using Tensor embeds = new(new TensorShape(1, ids.Length, cfg.HiddenSize), DType.F32);
        model.EmbedLookup(embeds, ids);
        using Tensor hidden = model.ForwardEmbeds(backend, embeds, ids.Length, cache.CurrentLength, cache, tokenIds: ids);
        using Tensor logits = model.ProjectLogits(backend, hidden, ids.Length);
        return ArgMaxLastRow(logits, ids.Length);
    }

    /// <summary>Two streams (CSM / HeartMuLa CFG decode) sharing one device position: <c>ForwardGraphDecodeStepDualEmbeds</c> over two caches of equal length.</summary>
    private static string Dual()
    {
        TransformerConfig cfg = Base() with { QkNorm = true, SlidingWindowPattern = 2, SlidingWindow = 3, AttnLogitSoftcap = 20f };
        using Weights w = ModelWeights(cfg, 501);
        using DevicePositionShimBackend shim = new();
        using GenericTransformer model = new(cfg);
        model.LoadWeights(w, "model");
        using FixedKvCache a = new(cfg.NumLayers, 1, cfg.NumKvHeads, cfg.HeadDim, Capacity);
        using FixedKvCache b = new(cfg.NumLayers, 1, cfg.NumKvHeads, cfg.HeadDim, Capacity);
        int[] next = [Prefill(model, shim, a, cfg, Prompt), Prefill(model, shim, b, cfg, [4, 15, 8, 0, 22, 6])];
        (Tensor cos, Tensor sin) = model.EnsureRopeTableForGraphDecode(shim, Capacity);
        ulong devicePos = shim.AllocDevicePos();
        Digests d = new();
        List<string> tokens = [$"{next[0]}/{next[1]}"];
        using Tensor inEmbed = new(new TensorShape(1, 2, cfg.HiddenSize), DType.F32);
        using Tensor outHidden = new(new TensorShape(1, 2, cfg.HiddenSize), DType.F32);
        for (int step = 0; step < 4; step++)
        {
            int pos = a.CurrentLength;
            model.EmbedLookup(inEmbed, next);
            shim.WriteDevicePos(devicePos, pos + 1, pos);
            model.ForwardGraphDecodeStepDualEmbeds(shim, inEmbed, a, b, cos, sin, devicePos, outHidden);
            a.AdvanceLength(1);
            b.AdvanceLength(1);
            using Tensor logits = model.ProjectLogits(shim, outHidden, 2);
            d.Add("hid", outHidden);
            d.Add("logits", logits);
            int vocab = (int)(logits.ElementCount / 2);
            next = [ArgMaxRow(logits, 0, vocab), ArgMaxRow(logits, 1, vocab)];
            tokens.Add($"{next[0]}/{next[1]}");
        }
        shim.FreeDevicePos(devicePos);
        d.Plain("toks", string.Join(',', tokens));
        DigestCache(d, a, "a", cfg);
        DigestCache(d, b, "b", cfg);
        return d.Finish();
    }

    // ── Tensor parallel (two CPU ranks) ────────────────────────────────────────────────────────────────────────────

    private static string TensorParallel(string name)
    {
        (TransformerConfig cfg, uint salt, bool oBias) = name switch
        {
            "tp_deg2_qknorm" => (Base() with { QkNorm = true }, 601u, false),
            "tp_deg2_bias" => (Base() with { AttentionBias = true }, 1602u, true),
            _ => (Base() with { NumKvHeads = 4 }, 603u, false),
        };
        using Weights w = ModelWeights(cfg, salt, oBias);
        using CpuBackend rank0 = new();
        using CpuBackend rank1 = new();
        using HostStagedComm comm = new(2);
        TpPlacement placement = new([rank0, rank1], comm);
        using TensorParallelTransformer tp = new(cfg, placement);
        tp.LoadWeights(w, "model");
        KvCache[] caches = tp.CreateKvCaches();
        try
        {
            Digests d = new();
            List<int> tokens = [];
            int[] ids = Prompt;
            for (int step = 0; step <= DecodeSteps; step++)
            {
                int t = ids.Length;
                using Tensor embeds = new(new TensorShape(1, t, cfg.HiddenSize), DType.F32);
                tp.EmbedLookup(embeds, ids);
                using Tensor embedsCopy = new(embeds.Shape, DType.F32);
                rank0.CopyTo(embedsCopy, embeds);
                using Tensor hidden = tp.ForwardEmbedsTp(embedsCopy, t, caches[0].CurrentLength, caches);
                using Tensor logits = tp.ProjectLogits(hidden, t);
                d.Add("hid", hidden);
                d.Add("logits", logits);
                int next = ArgMaxLastRow(logits, t);
                tokens.Add(next);
                ids = [next];
            }
            d.Plain("toks", string.Join(',', tokens));
            for (int r = 0; r < caches.Length; r++)
                for (int layer = 0; layer < cfg.NumLayers; layer++)
                    DigestKv(d, caches[r], $"r{r}_", layer, cfg.NumKvHeads / 2);
            return d.Finish();
        }
        finally
        {
            foreach (KvCache cache in caches) cache.Dispose();
        }
    }

    // ── Looped (nGPT) ──────────────────────────────────────────────────────────────────────────────────────────────

    private static LoopedTransformerConfig LoopedConfig(LoopedKvCacheMode mode) => new()
    {
        HiddenSize = 16, NumHeads = 2, NumKvHeads = 1, HeadDim = 8, IntermediateSize = 24, VocabSize = 20, MaxPositionEmbeddings = 64,
        InputLayers = 2, RecurrentLayers = 2, OutputLayers = 2, Loops = 2,
        LayerKinds = [LoopedLayerKind.SlidingAttention, LoopedLayerKind.Mlp, LoopedLayerKind.FullAttention, LoopedLayerKind.Mlp,
            LoopedLayerKind.SlidingAttention, LoopedLayerKind.Mlp],
        RopeThetas = [10_000f, 0f, 100_000f, 0f, 10_000f, 0f],
        SlidingWindow = 3, KvCacheMode = mode, InitStd = 0.02f, AlphaInit = 0.05f, Architecture = "dory", LowVramQuant = false,
    };

    private static Weights LoopedWeights(LoopedTransformerConfig cfg, uint salt)
    {
        Weights w = new(salt);
        int h = cfg.HiddenSize, nq = cfg.QDim, nk = cfg.KvDim, ff = cfg.IntermediateSize, v = cfg.VocabSize, d = cfg.HeadDim;
        w.Add("model.embed_tokens.weight", v, h, 1f / MathF.Sqrt(h));
        w.Add("lm_head.weight", v, h, 1f / MathF.Sqrt(h));
        w.AddScaleVector("lm_head.logit_scale.weight", v, cfg.InitStd, 0.5f);
        for (int p = 0; p < cfg.PhysicalLayers; p++)
        {
            string prefix = $"model.layers.{p}";
            w.AddScaleVector($"{prefix}.residual_alpha.weight", h, cfg.InitStd / cfg.AlphaInit * 0.5f, 0.8f);
            if (cfg.IsAttention(p))
            {
                w.Add($"{prefix}.self_attn.q_proj.weight", nq, h, 1f / MathF.Sqrt(h));
                w.Add($"{prefix}.self_attn.k_proj.weight", nk, h, 1f / MathF.Sqrt(h));
                w.Add($"{prefix}.self_attn.v_proj.weight", nk, h, 1f / MathF.Sqrt(h));
                w.Add($"{prefix}.self_attn.o_proj.weight", h, nq, 1f / MathF.Sqrt(nq));
                w.Add($"{prefix}.self_attn.gate_proj.weight", nq, h, 1f / MathF.Sqrt(h));
                w.AddScaleVector($"{prefix}.self_attn.qk_scale.weight", nq, cfg.InitStd / MathF.Sqrt(d), 0.3f);
                w.AddScaleVector($"{prefix}.self_attn.gate_scale.weight", nq, cfg.InitStd / MathF.Sqrt(cfg.EffectiveGateInit), 0.3f);
            }
            else
            {
                w.Add($"{prefix}.mlp.gate_proj.weight", ff, h, 1f / MathF.Sqrt(h));
                w.Add($"{prefix}.mlp.up_proj.weight", ff, h, 1f / MathF.Sqrt(h));
                w.Add($"{prefix}.mlp.down_proj.weight", h, ff, 1f / MathF.Sqrt(ff));
                w.AddScaleVector($"{prefix}.mlp.gate_scale.weight", ff, 1f / MathF.Sqrt(h), 0.3f);
                w.AddScaleVector($"{prefix}.mlp.up_scale.weight", ff, 1f / MathF.Sqrt(h), 0.3f);
            }
        }
        return w;
    }

    private static LoopedSequenceState LoopedState(LoopedTransformerConfig cfg) =>
        new(new FixedKvCache(LoopedSchedule.MaxSlotCount(cfg), 1, cfg.NumKvHeads, cfg.HeadDim, Capacity), cfg.Loops);

    private static string Looped(string name)
    {
        LoopedTransformerConfig cfg = LoopedConfig(name == "looped_lastloop" ? LoopedKvCacheMode.LastLoop : LoopedKvCacheMode.PerLoop);
        using Weights w = LoopedWeights(cfg, 701);
        using LoopedTransformer model = new(cfg);
        model.LoadWeights(w, "model");
        Digests d = new();
        return name switch
        {
            "looped_batch" => LoopedBatch(model, cfg, d),
            "looped_graph" => LoopedGraph(model, cfg, d),
            _ => LoopedPrefillDecode(model, cfg, d),
        };
    }

    /// <summary>A 4-token prefill chunk, a 3-token chunk, then four single-token decodes.</summary>
    private static string LoopedPrefillDecode(LoopedTransformer model, LoopedTransformerConfig cfg, Digests d)
    {
        using CpuBackend cpu = new();
        using LoopedSequenceState state = LoopedState(cfg);
        List<int> tokens = [];
        int[] ids = [3, 7, 1, 12];
        int[][] chunks = [ids, [5, 9, 14], [], [], [], []];
        for (int step = 0; step < chunks.Length; step++)
        {
            int[] chunk = chunks[step].Length > 0 ? chunks[step] : [tokens[^1]];
            using Tensor hidden = model.Forward(cpu, chunk, state.CurrentLength, state, cfg.Loops);
            using Tensor logits = model.ProjectLogits(cpu, hidden, chunk.Length);
            d.Add("hid", hidden);
            d.Add("logits", logits);
            if (step >= 1) tokens.Add(ArgMaxLastRow(logits, chunk.Length));
        }
        d.Plain("toks", string.Join(',', tokens));
        DigestSlots(d, state, "", cfg);
        return d.Finish();
    }

    private static string LoopedBatch(LoopedTransformer model, LoopedTransformerConfig cfg, Digests d)
    {
        using CpuBackend cpu = new();
        using LoopedSequenceState a = LoopedState(cfg);
        using LoopedSequenceState b = LoopedState(cfg);
        int[][] prompts = [[3, 7, 1, 12, 5], [11, 2, 9]];
        int[] next = new int[2];
        IKvCache[] states = [a, b];
        for (int s = 0; s < 2; s++)
        {
            using Tensor hidden = model.Forward(cpu, prompts[s], 0, states[s], cfg.Loops);
            using Tensor logits = model.ProjectLogits(cpu, hidden, prompts[s].Length);
            d.Add($"prefill{s}", logits);
            next[s] = ArgMaxLastRow(logits, prompts[s].Length);
        }
        List<string> tokens = [$"{next[0]}/{next[1]}"];
        for (int step = 0; step < 4; step++)
        {
            using Tensor embeds = new(new TensorShape(1, 2, cfg.HiddenSize), DType.F32);
            model.EmbedLookup(embeds, next);
            using Tensor hidden = model.ForwardBatchDecode(cpu, embeds, [a.CurrentLength, b.CurrentLength], states, cfg.Loops);
            using Tensor logits = model.ProjectLogits(cpu, hidden, 2);
            d.Add("hid", hidden);
            d.Add("logits", logits);
            int vocab = (int)(logits.ElementCount / 2);
            next = [ArgMaxRow(logits, 0, vocab), ArgMaxRow(logits, 1, vocab)];
            tokens.Add($"{next[0]}/{next[1]}");
        }
        d.Plain("toks", string.Join(',', tokens));
        DigestSlots(d, a, "a", cfg);
        DigestSlots(d, b, "b", cfg);
        return d.Finish();
    }

    private static string LoopedGraph(LoopedTransformer model, LoopedTransformerConfig cfg, Digests d)
    {
        using DevicePositionShimBackend shim = new();
        using LoopedSequenceState state = LoopedState(cfg);
        int[] ids = [3, 7, 1, 12, 5, 9];
        using (Tensor hidden = model.Forward(shim, ids, 0, state, cfg.Loops))
        using (Tensor logits = model.ProjectLogits(shim, hidden, ids.Length))
        {
            d.Add("prefill", logits);
            ids = [ArgMaxLastRow(logits, ids.Length)];
        }
        List<int> tokens = [ids[0]];
        Tensor embedTable = model.EnsureEmbedResidentForGraphDecode(shim);
        model.EnsureRopeTablesForGraphDecode(shim, Capacity);
        ulong devicePos = shim.AllocDevicePos(), tokenId = shim.AllocDeviceTokenId();
        for (int step = 0; step < DecodeSteps; step++)
        {
            int pos = state.CurrentLength;
            shim.WriteDeviceTokenId(tokenId, tokens[^1]);
            shim.WriteDevicePos(devicePos, pos + 1, pos);
            model.ForwardGraphDecodeStep(shim, embedTable, state, cfg.Loops, devicePos, tokenId);
            state.AdvanceLength(1);
            tokens.Add(shim.ReadDeviceTokenId(tokenId));
        }
        shim.FreeDevicePos(devicePos);
        shim.FreeDeviceTokenId(tokenId);
        d.Plain("toks", string.Join(',', tokens));
        DigestSlots(d, state, "", cfg);
        return d.Finish();
    }

    // ── Mllama cross-attention ─────────────────────────────────────────────────────────────────────────────────────

    private const int VisionLen = 5;

    private static void AddCrossLayer(Weights w, string p)
    {
        w.AddNorm($"{p}.input_layernorm.weight", Hidden);
        w.AddNorm($"{p}.post_attention_layernorm.weight", Hidden);
        w.Add($"{p}.cross_attn.q_proj.weight", 32, Hidden);
        w.Add($"{p}.cross_attn.k_proj.weight", 16, Hidden);
        w.Add($"{p}.cross_attn.v_proj.weight", 16, Hidden);
        w.Add($"{p}.cross_attn.o_proj.weight", Hidden, 32);
        w.AddNorm($"{p}.cross_attn.q_norm.weight", 8);
        w.AddNorm($"{p}.cross_attn.k_norm.weight", 8);
        AddMlp(w, p);
        w.AddScalar($"{p}.cross_attn_attn_gate", 0.7f);
        w.AddScalar($"{p}.cross_attn_mlp_gate", -0.4f);
    }

    /// <summary>The cross-attention layer over one constant vision tensor: a 3-token step, then three 1-token steps.</summary>
    private static string MllamaLayer()
    {
        using Weights w = new(801);
        AddCrossLayer(w, "blk.0");
        using CpuBackend cpu = new();
        MllamaCrossAttentionLayer layer = new(Hidden, 4, 2, 8, Inter, 1e-6f);
        layer.LoadWeights(w, "blk.0");
        using Tensor vision = Seeded(new TensorShape(1, VisionLen, Hidden), 8001, 1f);
        Digests d = new();
        int[] steps = [3, 1, 1, 1];
        for (int s = 0; s < steps.Length; s++)
        {
            using Tensor text = Seeded(new TensorShape(1, steps[s], Hidden), (uint)(8100 + s), 1f);
            using Tensor outp = layer.Forward(cpu, text, steps[s], vision, VisionLen);
            d.Add("out", outp);
        }
        return d.Finish();
    }

    /// <summary>A self / cross / self stack: the cross layer reads the same vision tensor on every prefill and decode call.</summary>
    private static string MllamaInTransformer()
    {
        TransformerConfig cfg = Base() with { NumLayers = 3, CrossAttnLayers = ImmutableHashSet.Create(1) };
        using Weights w = new(802);
        w.Add("model.embed_tokens.weight", Vocab, Hidden);
        w.Add("lm_head.weight", Vocab, Hidden);
        w.AddNorm("model.norm.weight", Hidden);
        for (int i = 0; i < cfg.NumLayers; i++)
        {
            string p = $"model.layers.{i}";
            if (i == 1)
            {
                AddCrossLayer(w, p);
                continue;
            }
            AddSelfAttention(w, cfg, p, i);
            w.AddNorm($"{p}.input_layernorm.weight", Hidden);
            w.AddNorm($"{p}.post_attention_layernorm.weight", Hidden);
            AddMlp(w, p);
        }
        using CpuBackend cpu = new();
        using GenericTransformer model = new(cfg);
        model.LoadWeights(w, "model");
        using FixedKvCache cache = new(cfg.NumLayers, 1, cfg.NumKvHeads, cfg.HeadDim, Capacity);
        using Tensor vision = Seeded(new TensorShape(1, VisionLen, Hidden), 8002, 1f);
        Digests d = new();
        List<int> tokens = [];
        int[] ids = Prompt;
        for (int step = 0; step <= DecodeSteps; step++)
        {
            int t = ids.Length;
            using Tensor embeds = new(new TensorShape(1, t, Hidden), DType.F32);
            model.EmbedLookup(embeds, ids);
            using Tensor hidden = model.ForwardEmbeds(cpu, embeds, t, cache.CurrentLength, cache, crossStates: vision, crossLen: VisionLen, tokenIds: ids,
                layerTap: (i, tap) => d.Add($"l{i}", tap));
            using Tensor logits = model.ProjectLogits(cpu, hidden, t);
            d.Add("hid", hidden);
            d.Add("logits", logits);
            int next = ArgMaxLastRow(logits, t);
            tokens.Add(next);
            ids = [next];
        }
        d.Plain("toks", string.Join(',', tokens));
        DigestCache(d, cache, "", cfg);
        return d.Finish();
    }

    // ── Qwen3.5 (hybrid Gated-DeltaNet / gated full attention) ─────────────────────────────────────────────────────

    private const int QHidden = 32, QHeads = 4, QKvHeads = 2, QHeadDim = 8, QLinValueHeads = 4, QLinKeyHeads = 2, QLinDim = 4, QConv = 4, QInter = 48;

    private static Qwen35HfConfig QwenConfig(bool allAttention) => new()
    {
        HiddenSize = QHidden, NumLayers = 4, VocabSize = Vocab, NumHeads = QHeads, NumKvHeads = QKvHeads, HeadDim = QHeadDim, FullAttentionInterval = 0,
        LinearLayers = allAttention ? new[] { false, false, false, false } : new[] { true, false, true, false },
        LinearConvKernel = QConv, LinearKeyHeadDim = QLinDim, LinearValueHeadDim = QLinDim, LinearNumKeyHeads = QLinKeyHeads, LinearNumValueHeads = QLinValueHeads,
        PartialRotaryFactor = 0.5f, RopeTheta = 10_000f, TieWordEmbeddings = false, MaxPositionEmbeddings = 64, MropeSection = [2, 1, 1], MropeInterleaved = true,
    };

    private static Weights QwenWeights(Qwen35HfConfig cfg, uint salt)
    {
        Weights w = new(salt);
        const string root = "model.language_model";
        w.Add($"{root}.embed_tokens.weight", Vocab, QHidden);
        w.Add("lm_head.weight", Vocab, QHidden);
        w.AddBias($"{root}.norm.weight", QHidden);
        for (int i = 0; i < cfg.NumLayers; i++)
        {
            string p = $"{root}.layers.{i}";
            w.AddBias($"{p}.input_layernorm.weight", QHidden);
            w.AddBias($"{p}.post_attention_layernorm.weight", QHidden);
            w.Add($"{p}.mlp.gate_proj.weight", QInter, QHidden);
            w.Add($"{p}.mlp.up_proj.weight", QInter, QHidden);
            w.Add($"{p}.mlp.down_proj.weight", QHidden, QInter);
            if (cfg.IsLinearLayer(i))
            {
                int convDim = QLinDim * QLinKeyHeads * 2 + QLinDim * QLinValueHeads;
                string la = $"{p}.linear_attn";
                w.Add($"{la}.in_proj_qkv.weight", convDim, QHidden);
                w.Add($"{la}.in_proj_z.weight", QLinDim * QLinValueHeads, QHidden);
                w.Add($"{la}.in_proj_b.weight", QLinValueHeads, QHidden);
                w.Add($"{la}.in_proj_a.weight", QLinValueHeads, QHidden);
                w.Add($"{la}.out_proj.weight", QHidden, QLinDim * QLinValueHeads);
                w.AddShaped($"{la}.conv1d.weight", new TensorShape(convDim, 1, QConv), 0.3f);
                w.AddBias($"{la}.dt_bias", QLinValueHeads);
                w.AddNorm($"{la}.norm.weight", QLinDim);
                w.AddBias($"{la}.A_log", QLinValueHeads);
            }
            else
            {
                string sa = $"{p}.self_attn";
                w.Add($"{sa}.q_proj.weight", QHeads * QHeadDim * 2, QHidden);
                w.Add($"{sa}.k_proj.weight", QKvHeads * QHeadDim, QHidden);
                w.Add($"{sa}.v_proj.weight", QKvHeads * QHeadDim, QHidden);
                w.Add($"{sa}.o_proj.weight", QHidden, QHeads * QHeadDim);
                w.AddBias($"{sa}.q_norm.weight", QHeadDim);
                w.AddBias($"{sa}.k_norm.weight", QHeadDim);
            }
        }
        return w;
    }

    private static string Qwen35(string name)
    {
        bool shimmed = name != "qwen35_attn_prefill_decode";
        Qwen35HfConfig cfg = QwenConfig(allAttention: shimmed);   // the Gated-DeltaNet device step needs CUDA
        using Weights w = QwenWeights(cfg, 901);
        using DevicePositionShimBackend shim = new(graphDecodeSupported: shimmed);
        using CpuBackend cpu = new();
        IBackend backend = shimmed ? shim : cpu;
        using Qwen35Model model = Qwen35Model.FromHuggingFace(w, cfg, maxSequenceLength: Capacity);
        Digests d = new();
        List<int> tokens = [];
        int[] ids = Prompt;
        int steps = name == "qwen35_graph_step" ? 0 : DecodeSteps;
        for (int step = 0; step <= steps; step++)
        {
            int t = ids.Length;
            using Tensor embeds = new(new TensorShape(1, t, QHidden), DType.F32);
            model.EmbedLookup(embeds, ids);
            using Tensor hidden = model.ForwardEmbedsHidden(backend, embeds, t);
            using Tensor logits = model.ProjectLogits(backend, hidden, t);
            d.Add("hid", hidden);
            d.Add("logits", logits);
            int next = ArgMaxLastRow(logits, t);
            tokens.Add(next);
            ids = [next];
        }
        if (name == "qwen35_graph_step")
        {
            (Tensor cos, Tensor sin) = shim.BuildRopeTableDevice(Capacity, QHeadDim, cfg.RotaryDim, cfg.RopeTheta, HartsyInference.Core.Rope.RopeScaling.None, splitHalfPartial: true);
            Tensor embed = w["model.language_model.embed_tokens.weight"];
            ulong devicePos = shim.AllocDevicePos(), tokenId = shim.AllocDeviceTokenId();
            for (int step = 0; step < DecodeSteps; step++)
            {
                int pos = model.CurrentPosition;
                shim.WriteDeviceTokenId(tokenId, tokens[^1]);
                shim.WriteDevicePos(devicePos, pos + 1, pos);
                model.ForwardGraphDecodeStep(shim, embed, cos, sin, devicePos, tokenId);
                model.AdvancePosition(1);
                tokens.Add(shim.ReadDeviceTokenId(tokenId));
            }
            shim.FreeDevicePos(devicePos);
            shim.FreeDeviceTokenId(tokenId);
            cos.Dispose();
            sin.Dispose();
        }
        d.Plain("toks", string.Join(',', tokens));
        return d.Finish();
    }

    // ── weights ────────────────────────────────────────────────────────────────────────────────────────────────────

    private static void AddMlp(Weights w, string p)
    {
        w.Add($"{p}.mlp.gate_proj.weight", Inter, Hidden);
        w.Add($"{p}.mlp.up_proj.weight", Inter, Hidden);
        w.Add($"{p}.mlp.down_proj.weight", Hidden, Inter);
    }

    private static void AddSelfAttention(Weights w, TransformerConfig c, string p, int layer)
    {
        int d = c.HeadDimFor(layer), qDim = c.NumHeads * d, kvDim = c.NumKvHeads * d;
        bool own = c.HasOwnKv(layer);
        w.Add($"{p}.self_attn.q_proj.weight", qDim, Hidden);
        if (own)
        {
            w.Add($"{p}.self_attn.k_proj.weight", kvDim, Hidden);
            w.Add($"{p}.self_attn.v_proj.weight", kvDim, Hidden);
        }
        w.Add($"{p}.self_attn.o_proj.weight", Hidden, qDim);
        if (c.AttentionBias)
        {
            w.AddBias($"{p}.self_attn.q_proj.bias", qDim);
            if (own)
            {
                w.AddBias($"{p}.self_attn.k_proj.bias", kvDim);
                w.AddBias($"{p}.self_attn.v_proj.bias", kvDim);
            }
        }
        if (c.QkNorm)
        {
            w.AddNorm($"{p}.self_attn.q_norm.weight", c.QkNormFullDim ? qDim : d);
            w.AddNorm($"{p}.self_attn.k_norm.weight", c.QkNormFullDim ? kvDim : d);
            if (c.UseLayerNorm)
            {
                w.AddBias($"{p}.self_attn.q_norm.bias", d);
                w.AddBias($"{p}.self_attn.k_norm.bias", d);
            }
        }
        if (c.AttnSink) w.AddBias($"{p}.self_attn.sinks.weight", c.NumHeads, 1f);
        if (c.AttnOutputGate) w.Add($"{p}.self_attn.gate_proj.weight", qDim, Hidden);
    }

    /// <summary>Every tensor a <see cref="GenericTransformer"/> of <paramref name="c"/> loads, seeded from its own name plus <paramref name="salt"/>.</summary>
    private static Weights ModelWeights(TransformerConfig c, uint salt, bool outBias = false)
    {
        Weights w = new(salt);
        w.Add("model.embed_tokens.weight", c.VocabSize, c.HiddenSize);
        w.Add("lm_head.weight", c.VocabSize, c.HiddenSize);
        w.AddNorm("model.norm.weight", c.HiddenSize);
        if (c.UseLayerNorm) w.AddBias("model.norm.bias", c.HiddenSize);
        for (int i = 0; i < c.NumLayers; i++)
        {
            string p = $"model.layers.{i}";
            w.AddNorm($"{p}.input_layernorm.weight", c.HiddenSize);
            if (c.UseLayerNorm) w.AddBias($"{p}.input_layernorm.bias", c.HiddenSize);
            if (c.SandwichNorm)
            {
                if (!c.PostAttnNormWeightless) w.AddNorm($"{p}.post_attention_layernorm.weight", c.HiddenSize);
                w.AddNorm($"{p}.pre_feedforward_layernorm.weight", c.HiddenSize);
                w.AddNorm($"{p}.post_feedforward_layernorm.weight", c.HiddenSize);
            }
            else if (!c.ParallelResidual)
            {
                w.AddNorm($"{p}.post_attention_layernorm.weight", c.HiddenSize);
                if (c.UseLayerNorm) w.AddBias($"{p}.post_attention_layernorm.bias", c.HiddenSize);
            }
            AddSelfAttention(w, c, p, i);
            if (outBias) w.AddBias($"{p}.self_attn.o_proj.bias", c.HiddenSize);
            if (c.ResidualGates)
            {
                w.AddNorm($"{p}.post_attention_residual_gate.alpha", c.HiddenSize);
                w.AddBias($"{p}.post_attention_residual_gate.beta", c.HiddenSize, 1f);
                w.AddNorm($"{p}.post_feedforward_residual_gate.alpha", c.HiddenSize);
                w.AddBias($"{p}.post_feedforward_residual_gate.beta", c.HiddenSize, 1f);
            }
            AddMlp(w, p);
        }
        return w;
    }

    // ── helpers ────────────────────────────────────────────────────────────────────────────────────────────────────

    private static int ArgMaxLastRow(Tensor logits, int rows) => ArgMaxRow(logits, rows - 1, (int)(logits.ElementCount / rows));

    private static int ArgMaxRow(Tensor logits, int row, int vocab)
    {
        float* p = (float*)logits.DataPointer + (long)row * vocab;
        int best = 0;
        for (int i = 1; i < vocab; i++)
        {
            if (p[i] > p[best]) best = i;
        }
        return best;
    }

    /// <summary>Digests the committed K/V rows of every layer that owns a cache slot.</summary>
    private static void DigestCache(Digests d, IKvCache cache, string tag, TransformerConfig cfg)
    {
        for (int layer = 0; layer < cfg.NumLayers; layer++)
        {
            if (cfg.IsCrossAttnLayer(layer) || !cfg.HasOwnKv(layer)) continue;
            DigestKv(d, cache, tag, layer, cfg.NumKvHeads);
        }
    }

    private static void DigestSlots(Digests d, IKvCache cache, string tag, LoopedTransformerConfig cfg)
    {
        for (int slot = 0; slot < LoopedSchedule.MaxSlotCount(cfg); slot++)
            DigestKv(d, cache, tag, slot, cfg.NumKvHeads);
    }

    /// <summary>The first <c>CurrentLength</c> rows of each head of the layer's K and V (a fixed cache's buffer is capacity-strided).</summary>
    private static void DigestKv(Digests d, IKvCache cache, string tag, int layer, int kvHeads)
    {
        int rows = cache.CurrentLength;
        foreach ((string label, Tensor t) in new[] { ("k", cache.KeyPrefix(layer)), ("v", cache.ValuePrefix(layer)) })
        {
            int stride = (int)t.Shape[2], dim = (int)t.Shape[3];
            for (int h = 0; h < kvHeads; h++)
                d.AddFloats($"{label}{tag}{layer}", (float*)t.DataPointer + (long)h * stride * dim, (long)rows * dim);
        }
    }

    /// <summary>Running SHA-256 per label (appended in call order) plus plain labels; non-finite values are rejected.</summary>
    private sealed class Digests
    {
        private readonly List<(string Label, IncrementalHash Hash)> _hashes = [];
        private readonly List<string> _plain = [];

        public void Add(string label, Tensor t) => AddFloats(label, (float*)t.DataPointer, t.ElementCount);

        public void AddFloats(string label, float* data, long count)
        {
            for (long i = 0; i < count; i++)
                if (!float.IsFinite(data[i]))
                    throw new InvalidOperationException($"Non-finite value in '{label}' at {i}.");
            int at = _hashes.FindIndex(entry => entry.Label == label);
            if (at < 0)
            {
                _hashes.Add((label, IncrementalHash.CreateHash(HashAlgorithmName.SHA256)));
                at = _hashes.Count - 1;
            }
            _hashes[at].Hash.AppendData(new ReadOnlySpan<byte>(data, checked((int)(count * sizeof(float)))));
        }

        public void Plain(string label, string value) => _plain.Add($"{label}={value}");

        public string Finish()
        {
            string digests = string.Join(' ', _hashes.Select(entry => $"{entry.Label}={Convert.ToHexStringLower(entry.Hash.GetHashAndReset())}"));
            foreach ((string _, IncrementalHash hash) in _hashes) hash.Dispose();
            return _plain.Count == 0 ? digests : $"{digests} {string.Join(' ', _plain)}";
        }
    }

    /// <summary>Caches (and the pool a paged one draws on) a case created, disposed in reverse order.</summary>
    private sealed class Caches : IDisposable
    {
        private readonly List<IDisposable> _owned = [];
        private PagedKvPool? _pool;

        public IKvCache New(CacheKind kind, TransformerConfig cfg)
        {
            IKvCache cache;
            switch (kind)
            {
                case CacheKind.Fixed:
                    cache = new FixedKvCache(cfg.NumLayers, 1, cfg.NumKvHeads, cfg.HeadDimsPerLayer(), Capacity);
                    break;
                case CacheKind.Growing:
                    cache = new KvCache(cfg.NumLayers, 1, cfg.NumKvHeads, cfg.HeadDim);
                    break;
                case CacheKind.Streaming:
                    cache = new StreamingKvCache(cfg.NumLayers, 1, cfg.NumKvHeads, Capacity, cfg.HeadDim);
                    break;
                default:
                    if (_pool is null)
                    {
                        _pool = new PagedKvPool(cfg.NumLayers, cfg.NumKvHeads, cfg.HeadDimsPerLayer(), 4, 16);
                        _owned.Add(_pool);
                    }
                    cache = new PagedKvCache(_pool);
                    break;
            }
            _owned.Add(cache);
            return cache;
        }

        public void Dispose()
        {
            for (int i = _owned.Count - 1; i >= 0; i--) _owned[i].Dispose();
        }
    }

    private static Tensor Seeded(TensorShape shape, uint seed, float scale)
    {
        Tensor tensor = new(shape, DType.F32);
        float* data = (float*)tensor.DataPointer;
        uint state = seed * 2654435761u + 0x9E3779B9u;
        for (long i = 0; i < tensor.ElementCount; i++)
        {
            state = state * 1664525u + 1013904223u;
            data[i] = (((state >> 8) & 0xFFFF) / 32768f - 1f) * scale;
        }
        return tensor;
    }

    /// <summary>A checkpoint stand-in: each tensor is seeded from its own name (FNV-1a) plus a per-case salt, and owned (disposed) here.</summary>
    private sealed class Weights(uint salt) : Dictionary<string, Tensor>, IDisposable
    {
        public void Add(string name, long rows, long cols, float scale = 0.5f) => base.Add(name, Seeded(new TensorShape(rows, cols), Seed(name), scale));

        public void AddShaped(string name, TensorShape shape, float scale) => base.Add(name, Seeded(shape, Seed(name), scale));

        public void AddBias(string name, long n, float scale = 0.1f) => base.Add(name, Seeded(new TensorShape(n), Seed(name), scale));

        public void AddScalar(string name, float value)
        {
            Tensor t = new(new TensorShape(1), DType.F32);
            *(float*)t.DataPointer = value;
            base.Add(name, t);
        }

        /// <summary>Norm scale near 1.</summary>
        public void AddNorm(string name, long n)
        {
            Tensor t = Seeded(new TensorShape(n), Seed(name), 0.25f);
            float* p = (float*)t.DataPointer;
            for (long i = 0; i < n; i++) p[i] += 1f;
            base.Add(name, t);
        }

        /// <summary>A per-channel scale vector around <paramref name="center"/> with relative spread <paramref name="spread"/>.</summary>
        public void AddScaleVector(string name, long n, float center, float spread)
        {
            Tensor t = Seeded(new TensorShape(n), Seed(name), 1f);
            float* p = (float*)t.DataPointer;
            for (long i = 0; i < n; i++) p[i] = center * (1f + spread * p[i]);
            base.Add(name, t);
        }

        /// <summary>Stores every tensor named <c>*.<paramref name="suffix"/></c> as F16 (a mixed-dtype projection that defeats the fused QKV weight).</summary>
        public void ToF16(string suffix)
        {
            foreach (string name in Keys.Where(k => k.EndsWith("." + suffix, StringComparison.Ordinal)).ToList())
            {
                Tensor original = this[name];
                this[name] = original.CastTo(DType.F16);
                original.Dispose();
            }
        }

        public void Dispose()
        {
            foreach (Tensor t in Values) t.Dispose();
        }

        private uint Seed(string name)
        {
            uint h = 2166136261u ^ salt;
            foreach (char c in name) h = (h ^ c) * 16777619u;
            return h;
        }
    }
}
