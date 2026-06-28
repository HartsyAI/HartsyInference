using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Sampling;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Multimodal;

/// <summary>End-to-end generation for Llama-3.2-Vision (mllama). Unlike the splice-token VLMs, mllama does NOT
/// insert image embeddings into the token sequence: the <see cref="MllamaVisionEncoder"/> produces
/// <c>cross_attention_states</c> that the interleaved gated cross-attention decoder layers
/// (<see cref="MllamaCrossAttentionLayer"/>) read, while the text sequence carries a single <c>&lt;|image|&gt;</c>
/// placeholder token. The vision states are threaded through every <see cref="GenericTransformer.ForwardEmbeds"/>
/// call (prefill + each decode step) so the cross-attention layers always see the image.</summary>
public sealed class MllamaGenerator
{
    private readonly GgufLanguageModel _text;
    private readonly MllamaVisionEncoder _vision;
    private readonly IBackend _backend;

    public MllamaGenerator(GgufLanguageModel text, MllamaVisionEncoder vision, IBackend backend)
    {
        _text = text; _vision = vision; _backend = backend;
    }

    /// <summary>Generates a reply to <paramref name="question"/> about the image (<paramref name="pixelValues"/> =
    /// normalized <c>[1, 3, 560, 560]</c>). Llama-3 chat prompt with a single <c>&lt;|image|&gt;</c> token before
    /// the question.</summary>
    public unsafe string Generate(Tensor pixelValues, string question, int maxTokens = 64, SamplingOptions? sampling = null)
    {
        SamplerChain sampler = SamplerChain.FromOptions(sampling ?? new SamplingOptions
        {
            Temperature = 0.4f, TopP = 0.9f, RepetitionPenalty = 1.1f, Seed = 1,
        });
        GenericTransformer model = _text.Transformer;
        int hidden = _text.Config.HiddenSize;

        // 1. Encode the image → cross_attention_states [1, visLen, hidden].
        using Tensor crossStates = _vision.Encode(_backend, pixelValues);
        int visLen = (int)crossStates.Shape[1];

        // 2. Llama-3 chat prompt with a single <|image|> placeholder before the question.
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
            using Tensor h = model.ForwardEmbeds(_backend, embeds, seqLen, 0, cache,
                applyFinalNorm: true, startLayer: 0, endLayer: null, crossStates: crossStates, crossLen: visLen);
            token = SampleLast(model, h, seqLen, sampler, history);
        }

        int pos = seqLen;
        for (int step = 0; step < maxTokens && !stops.Contains(token); step++)
        {
            sb.Append(_text.Tokenizer.Decode([token]));
            history.Add(token);
            using Tensor stepEmb = new(new TensorShape(1, 1, hidden), DType.F32);
            model.EmbedLookup(stepEmb, new[] { token });
            using Tensor h = model.ForwardEmbeds(_backend, stepEmb, 1, pos, cache,
                applyFinalNorm: true, startLayer: 0, endLayer: null, crossStates: crossStates, crossLen: visLen);
            pos++;
            token = SampleLast(model, h, 1, sampler, history);
        }
        return sb.ToString();
    }

    private unsafe int SampleLast(GenericTransformer model, Tensor hidden, int t, SamplerChain sampler, List<int> history)
    {
        int h = model.Config.HiddenSize;
        using Tensor last = new(new TensorShape(1, 1, h), DType.F32);
        float* hp = (float*)hidden.DataPointer;
        Buffer.MemoryCopy(hp + (long)(t - 1) * h, (void*)last.DataPointer, (long)h * 4, (long)h * 4);
        using Tensor logits = model.ProjectLogits(_backend, last, 1);
        float* lp = (float*)logits.DataPointer;
        Span<float> span = new(lp, model.Config.VocabSize);
        return sampler.Next(span, history);
    }
}
