using HartsyInference.Audio.Frontends;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Sampling;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.Audio.Models.Music;

/// <summary>Per-call knobs for the ACE-Step 5 Hz LM planner (upstream <c>llm_inference.py</c> defaults).</summary>
public sealed record AceStepPlannerOptions
{
    public bool Thinking { get; init; } = true;
    public float Temperature { get; init; } = 0.85f;
    /// <summary>LM-level CFG over logits (cond vs negative-prompt uncond stream); ≤1 disables.</summary>
    public float CfgScale { get; init; } = 2.0f;
    public int TopK { get; init; }
    public float TopP { get; init; } = 0.9f;
    public string NegativePrompt { get; init; } = "";
    public int Seed { get; init; }
}

/// <summary>ACE-Step 5 Hz LM planner that plans a song as FSQ audio codes using a stock Qwen3 causal LM.</summary>
/// <remarks>Vocab is 217,204 = Qwen 151,936 + 65,535 <c>&lt;|audio_code_N|&gt;</c> added tokens (id = 151,669 + N).
/// Runs on the LLM lib's <see cref="GenericTransformer"/> (no bespoke transformer): phase 1 samples the
/// <c>&lt;think&gt;…&lt;/think&gt;</c> CoT (no CFG), phase 2 samples exactly <c>duration·5</c> audio codes with
/// logits masked to code ids (+EOS) and optional dual-KV-stream CFG, mirroring upstream
/// <c>_generate_with_cfg_custom</c>. Codes then feed <c>AceStep15AudioDetokenizer</c> → 25 Hz lmHints.</remarks>
public sealed unsafe class AceStepLmPlanner : IDisposable
{
    private const int ImStart = 151_644;
    private const int ImEnd = 151_645;          // also EOS
    private const int ThinkOpen = 151_667;
    private const int ThinkClose = 151_668;
    private const int AudioCodeBase = 151_669;
    private const int NumAudioTokens = 65_535;
    private const string Instruction = "Generate audio semantic tokens based on the given conditions:";

    private readonly GenericTransformer _lm;
    private readonly TransformerConfig _cfg;
    private int _disposed;

    /// <summary>acestep-5Hz-lm-0.6B (Qwen3-0.6B body, 217,204 vocab, tied embeddings).</summary>
    public static TransformerConfig Config0_6B => TransformerConfig.Qwen3_0_6B with { VocabSize = 217_204 };

    /// <summary>acestep-5Hz-lm-4B (Qwen3-4B body: 36L / 2560h / 32H / 8KV / hd128 / ff9728, tied).</summary>
    public static TransformerConfig Config4B => TransformerConfig.Qwen3_0_6B with
    {
        VocabSize = 217_204, HiddenSize = 2560, NumLayers = 36, NumHeads = 32, NumKvHeads = 8,
        HeadDim = 128, IntermediateSize = 9728,
    };

    public AceStepLmPlanner(TransformerConfig config, IReadOnlyDictionary<string, Tensor> weights)
    {
        _cfg = config;
        _lm = new GenericTransformer(config);
        _lm.LoadWeights(weights, "model");
    }

    public IEnumerable<Tensor> EnumerateWeights() => _lm.EnumerateWeights();

    /// <summary>Plans <c>duration·5</c> FSQ audio codes for the caption/lyrics; returns the codes and the CoT text (empty when thinking is off).</summary>
    public (int[] Codes, string Cot) Plan(IBackend backend, string caption, string lyrics, double durationSeconds,
        AceStepPlannerOptions opts, CancellationToken cancel = default)
    {
        ThrowIfDisposed();
        int targetCodes = Math.Max(5, (int)(Math.Clamp(durationSeconds, 1, 600) * 5));
        List<int> chatPrefix = BuildChatPrefix($"# Caption\n{caption}\n\n# Lyric\n{lyrics}\n");

        // ── Phase 1: CoT (single stream, no CFG; stop at </think>) ──
        List<int> cotIds = [];
        string cotText = "";
        if (opts.Thinking)
        {
            SamplerChain cotSampler = SamplerChain.FromOptions(new SamplingOptions
            {
                Temperature = opts.Temperature, TopK = opts.TopK, TopP = opts.TopP,
                Seed = (ulong)unchecked((uint)opts.Seed) + 1,
            });
            using IKvCache cache = KvCaches.ForDecode(_cfg.NumLayers, _cfg.NumKvHeads, _cfg.HeadDim, chatPrefix.Count + targetCodes + 700);
            int next = Prefill(backend, chatPrefix, cache, cotSampler, maskToCodes: false, history: chatPrefix);
            List<int> cotHistory = [.. chatPrefix];
            for (int i = 0; i < 600 && next != ImEnd; i++)
            {
                cancel.ThrowIfCancellationRequested();
                cotIds.Add(next);
                cotHistory.Add(next);
                if (next == ThinkClose) break;
                // The just-sampled token sits at absolute position prefixLen + (cotIds.Count − 1).
                next = Step(backend, next, chatPrefix.Count + cotIds.Count - 1, cache, cotSampler, maskToCodes: false, cotHistory);
            }
            cotText = DecodeCot(cotIds);
            Logs.Info($"ACE-Step LM planner CoT ({cotIds.Count} tokens): {cotText.Replace('\n', ' ')[..Math.Min(cotText.Length, 200)]}");
        }

        // ── Phase 2: codes (masked to audio ids; dual-stream CFG when cfgScale > 1) ──
        // Prompt = chat prefix + CoT ids + "\n\n" (open assistant turn; training layout `<think>…</think>\n\n{codes}`).
        // Empty reasoning renders as `<think>\n\n</think>` — the extra inner newline matters.
        List<int> condPrompt = [.. chatPrefix];
        if (cotIds.Count > 0)
        {
            condPrompt.AddRange(cotIds);
        }
        else
        {
            condPrompt.Add(ThinkOpen);
            condPrompt.AddRange(AudioTextFrontend.Qwen3Ids("\n\n"));
            condPrompt.Add(ThinkClose);
        }
        condPrompt.AddRange(AudioTextFrontend.Qwen3Ids("\n\n"));

        bool useCfg = opts.CfgScale > 1f;
        List<int>? uncondPrompt = null;
        if (useCfg)
        {
            // Training CFG-dropout format: user message = raw negative prompt (default "NO USER INPUT"), empty CoT.
            string neg = string.IsNullOrWhiteSpace(opts.NegativePrompt) ? "NO USER INPUT" : opts.NegativePrompt;
            uncondPrompt = BuildChatPrefix(neg);
            uncondPrompt.Add(ThinkOpen);
            uncondPrompt.AddRange(AudioTextFrontend.Qwen3Ids("\n\n"));
            uncondPrompt.Add(ThinkClose);
            uncondPrompt.AddRange(AudioTextFrontend.Qwen3Ids("\n\n"));
        }

        SamplerChain sampler = SamplerChain.FromOptions(new SamplingOptions
        {
            Temperature = opts.Temperature, TopK = opts.TopK, TopP = opts.TopP,
            Seed = (ulong)unchecked((uint)opts.Seed) + 2,
        });
        int maxSeq = Math.Max(condPrompt.Count, uncondPrompt?.Count ?? 0) + targetCodes + 16;
        using IKvCache condCache = KvCaches.ForDecode(_cfg.NumLayers, _cfg.NumKvHeads, _cfg.HeadDim, maxSeq);
        using IKvCache uncondCache = useCfg ? KvCaches.ForDecode(_cfg.NumLayers, _cfg.NumKvHeads, _cfg.HeadDim, maxSeq) : KvCaches.ForDecode(1, 1, 1, 1);

        List<int> codes = new(targetCodes);
        List<int> history = [.. condPrompt];
        float[]? condLogits = PrefillLogits(backend, condPrompt, condCache);
        float[]? uncondLogits = useCfg ? PrefillLogits(backend, uncondPrompt!, uncondCache) : null;
        int condPos = condPrompt.Count;
        int uncondPos = uncondPrompt?.Count ?? 0;

        while (codes.Count < targetCodes)
        {
            cancel.ThrowIfCancellationRequested();
            // CFG blend restricted to the audio-code ids (+EOS) — everything else −inf (upstream masks BEFORE
            // CFG to avoid (−inf)−(−inf) NaN). EOS stays masked until the target count forces the stop.
            float[] blended = new float[_cfg.VocabSize];
            Array.Fill(blended, float.NegativeInfinity);
            int hi = Math.Min(AudioCodeBase + NumAudioTokens, _cfg.VocabSize);
            for (int id = AudioCodeBase; id < hi; id++)
            {
                blended[id] = useCfg ? uncondLogits![id] + opts.CfgScale * (condLogits![id] - uncondLogits[id])
                    : condLogits![id];
            }
            int next = sampler.Next(blended.AsSpan(), history);
            codes.Add(next - AudioCodeBase);
            history.Add(next);
            if (codes.Count >= targetCodes) break;

            condLogits = StepLogits(backend, next, condPos++, condCache);
            if (useCfg) uncondLogits = StepLogits(backend, next, uncondPos++, uncondCache);
        }
        Logs.Info($"ACE-Step LM planner: {codes.Count} codes ({codes.Count / 5.0:0.0}s @ 5 Hz), cfg={(useCfg ? opts.CfgScale : 1f)}.");
        return ([.. codes], cotText);
    }

    // Rendered Qwen chat template (verified vs the repo's chat_template.jinja):
    // <|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n
    private static List<int> BuildChatPrefix(string userContent)
    {
        List<int> ids = [ImStart];
        ids.AddRange(AudioTextFrontend.Qwen3Ids($"system\n# Instruction\n{Instruction}\n\n"));
        ids.Add(ImEnd);
        ids.AddRange(AudioTextFrontend.Qwen3Ids("\n"));
        ids.Add(ImStart);
        ids.AddRange(AudioTextFrontend.Qwen3Ids($"user\n{userContent}"));
        ids.Add(ImEnd);
        ids.AddRange(AudioTextFrontend.Qwen3Ids("\n"));
        ids.Add(ImStart);
        ids.AddRange(AudioTextFrontend.Qwen3Ids("assistant\n"));
        return ids;
    }

    private int Prefill(IBackend backend, List<int> prompt, IKvCache cache, SamplerChain sampler, bool maskToCodes, List<int> history)
    {
        float[] logits = PrefillLogits(backend, prompt, cache);
        return SampleFrom(logits, sampler, maskToCodes, history);
    }

    private int Step(IBackend backend, int token, int pos, IKvCache cache, SamplerChain sampler, bool maskToCodes, List<int> history)
    {
        float[] logits = StepLogits(backend, token, pos, cache);
        return SampleFrom(logits, sampler, maskToCodes, history);
    }

    private int SampleFrom(float[] logits, SamplerChain sampler, bool maskToCodes, List<int> history)
    {
        if (maskToCodes)
        {
            for (int id = 0; id < AudioCodeBase; id++) logits[id] = float.NegativeInfinity;
        }
        return sampler.Next(logits.AsSpan(), history);
    }

    private float[] PrefillLogits(IBackend backend, List<int> prompt, IKvCache cache)
    {
        Tensor hidden = _lm.Forward(backend, prompt.ToArray(), 0, cache);
        float[] logits = LastRowLogits(backend, hidden, prompt.Count);
        hidden.Dispose();
        return logits;
    }

    // <paramref name="atPos"/> = the absolute sequence position of the fed token (cache already holds atPos entries).
    private float[] StepLogits(IBackend backend, int token, int atPos, IKvCache cache)
    {
        Tensor hidden = _lm.Forward(backend, [token], atPos, cache);
        float[] logits = LastRowLogits(backend, hidden, 1);
        hidden.Dispose();
        return logits;
    }

    private float[] LastRowLogits(IBackend backend, Tensor hidden, int t)
    {
        Tensor logitsT = _lm.ProjectLogits(backend, hidden, t);
        int vocab = _cfg.VocabSize;
        float[] row = new float[vocab];
        float* lp = (float*)logitsT.DataPointer + (long)(t - 1) * vocab;
        new ReadOnlySpan<float>(lp, vocab).CopyTo(row);
        logitsT.Dispose();
        return row;
    }

    // CoT decode for logging/prompt reuse: base-vocab BPE ids via the shared Qwen tokenizer; specials spelled out.
    private static string DecodeCot(List<int> ids)
    {
        System.Text.StringBuilder sb = new();
        List<int> run = [];
        void Flush()
        {
            if (run.Count > 0) { sb.Append(AudioTextFrontend.Qwen3Decode(run)); run.Clear(); }
        }
        foreach (int id in ids)
        {
            if (id == ThinkOpen) { Flush(); sb.Append("<think>"); }
            else if (id == ThinkClose) { Flush(); sb.Append("</think>"); }
            else if (id >= 151_643) { Flush(); }   // other specials/codes: skip in text form
            else run.Add(id);
        }
        Flush();
        return sb.ToString();
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _lm.Dispose();
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(AceStepLmPlanner));
    }
}
