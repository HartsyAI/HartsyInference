using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Sampling;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Multimodal;

/// <summary>End-to-end generation for Llama-3.2-Vision (mllama): unlike the splice-token VLMs, the text sequence carries a single <c>&lt;|image|&gt;</c> placeholder while <see cref="MllamaVisionEncoder"/>'s <c>cross_attention_states</c> are threaded through every <see cref="GenericTransformer.ForwardEmbeds"/> call (prefill + each decode step) for the interleaved gated cross-attention layers (<see cref="MllamaCrossAttentionLayer"/>) to read.</summary>
public sealed class MllamaGenerator
{
    private readonly GgufLanguageModel _text;
    private readonly MllamaVisionEncoder _vision;
    private readonly IBackend _backend;
    private readonly LlmPlacement? _placement;

    /// <summary>True when a genuine multi-stage layer-split placement is active — the cross-attention states then need a per-stage peer copy (see <see cref="GenericTransformer.ForwardEmbedsStaged"/>).</summary>
    private bool Staged => _placement is not null && !_placement.IsSingle;

    /// <summary><paramref name="placement"/> shards the decoder layers across devices (VRAM pooling), or null keeps the single-backend path byte-identical; when set, <paramref name="backend"/> must be the placement's LAST stage backend (final norm/lm_head/logits live there).</summary>
    public MllamaGenerator(GgufLanguageModel text, MllamaVisionEncoder vision, IBackend backend, LlmPlacement? placement = null)
    {
        _text = text; _vision = vision; _placement = placement;
        _backend = placement is not null ? placement.LastBackend : backend;
        if (Staged)
        {
            StagedWeightPreload.Preload(text.Transformer, _placement!);
            return;
        }
        // Same headroom-guarded weight preload TextGenerationPipeline does at construction: without it,
        // auto-promotion's size floor leaves every small weight (norms, gates) host-side FOREVER, and each
        // eager decode step re-uploads them (measured ~26 ms/generation of pure H2D churn on small models —
        // proportionally worse on mllama's 40-layer decode loop, which has no graph path to hide it).
        try
        {
            long headroom = 2L << 30;
            if (_backend.FreeMemoryBytes() is long free && free > headroom)
            {
                List<HartsyInference.Core.Tensors.Tensor> toPreload = [];
                long budget = free - headroom;
                foreach (HartsyInference.Core.Tensors.Tensor t in text.Transformer.EnumerateWeights())
                {
                    long bytes = HartsyInference.Core.Tensors.Tensor.ComputeByteSize(t.Shape, t.DType);
                    if (budget - bytes < 0) continue;
                    budget -= bytes;
                    toPreload.Add(t);
                }
                _backend.PreloadWeights(toPreload);
            }
        }
        catch (Exception ex) { HartsyInference.Core.Logging.Logs.Warning($"mllama weight preload failed (continuing lazy): {ex.Message}"); }
    }

    /// <summary>Generates a reply to <paramref name="question"/> about the image (<paramref name="pixelValues"/> = normalized <c>[1, 3, 560, 560]</c>) using a Llama-3 chat prompt with a single <c>&lt;|image|&gt;</c> token before the question.</summary>
    public unsafe string Generate(Tensor pixelValues, string question, int maxTokens = 64, SamplingOptions? sampling = null)
    {
        SamplerChain sampler = SamplerChain.FromOptions(sampling ?? new SamplingOptions
        {
            Temperature = 0.4f, TopP = 0.9f, RepetitionPenalty = 1.1f, Seed = 1,
        }, _text.Tokenizer, _text.Config.VocabSize);
        GenericTransformer model = _text.Transformer;
        int hidden = _text.Config.HiddenSize;

        // 1. Encode the image → cross_attention_states [1, visLen, hidden]. Staged: runs on stage 0's backend
        // (ForwardEmbedsStaged treats it as the master copy and peer-copies it onto every other cross-attention
        // stage — see StageHasCrossAttn there).
        IBackend visionBackend = Staged ? _placement!.Stages[0].Backend : _backend;
        using Tensor crossStates = _vision.Encode(visionBackend, pixelValues);
        int visLen = (int)crossStates.Shape[1];

        // 2. Llama-3 chat prompt with a single <|image|> placeholder before the question.
        // NOTE (2026-07-23): encoding this template with addSpecial:true (proper single-token template ids)
        // made the model STOP after ~2 tokens — the single-id <|image|> interacts badly with the splice-free
        // embed path (this generator feeds vision via cross-attention only; the literal "<|image|>" text being
        // ordinary-BPE'd is apparently what the verified-working configuration relies on). Kept at false;
        // template markers therefore BPE as text, which costs some answer discipline (occasional free-running /
        // template echoes — the echoes are now suppressed by the tokenizer's template-token decode skip). A
        // proper fix needs the image token handled explicitly alongside true special encoding — scoped follow-up.
        int[] ids = _text.Tokenizer.Encode(
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n<|image|>" + question +
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", addSpecial: false);
        int seqLen = ids.Length;

        HashSet<int> stops = new(_text.Tokenizer.StopIds);
        using FixedKvCache cache = new(model.Config.NumLayers, 1, model.Config.NumKvHeads, model.Config.HeadDim, seqLen + maxTokens + 1);
        System.Text.StringBuilder sb = new();

        // 3. Prefill (text embeddings + cross-attention to the vision states), then greedy/sampled decode.
        List<int> history = new();
        int token;
        using (Tensor embeds = new(new TensorShape(1, seqLen, hidden), DType.F32))
        {
            model.EmbedLookup(embeds, ids);
            using Tensor h = Forward(model, embeds, seqLen, 0, cache, crossStates, visLen);
            token = SampleLast(model, h, seqLen, sampler, history);
        }

        int pos = seqLen;
        System.Diagnostics.Stopwatch decodeSw = System.Diagnostics.Stopwatch.StartNew();
        int decoded = 0;
        for (int step = 0; step < maxTokens && !stops.Contains(token); step++)
        {
            decoded++;
            sb.Append(_text.Tokenizer.Decode([token]));
            history.Add(token);
            using Tensor stepEmb = new(new TensorShape(1, 1, hidden), DType.F32);
            model.EmbedLookup(stepEmb, new[] { token });
            using Tensor h = Forward(model, stepEmb, 1, pos, cache, crossStates, visLen);
            pos++;
            token = SampleLast(model, h, 1, sampler, history);
        }
        if (decoded > 1)
            HartsyInference.Core.Logging.Logs.Info(
                $"[mllama] decode: {decoded} tokens in {decodeSw.Elapsed.TotalSeconds:F2}s = {decoded / decodeSw.Elapsed.TotalSeconds:F1} tok/s");
        return sb.ToString();
    }

    /// <summary>The hidden-state forward for one step (prefill or one decode token): staged across the placement (cross-attention states peer-copied per stage) when sharded, else the plain single-backend cross-attention path.</summary>
    private Tensor Forward(GenericTransformer model, Tensor embeds, int t, int posStart, IKvCache cache,
        Tensor crossStates, int visLen) =>
        Staged
            ? model.ForwardEmbedsStaged(_placement!, embeds, t, posStart, cache, crossStates: crossStates, crossLen: visLen)
            : model.ForwardEmbeds(_backend, embeds, t, posStart, cache,
                applyFinalNorm: true, startLayer: 0, endLayer: null, crossStates: crossStates, crossLen: visLen);

    private int SampleLast(GenericTransformer model, Tensor hidden, int t, SamplerChain sampler, List<int> history)
        => VlmDecodeOps.SampleLast(_backend, model, hidden, t, sampler, history);
}
