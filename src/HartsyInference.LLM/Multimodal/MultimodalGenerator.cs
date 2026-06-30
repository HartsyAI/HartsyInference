using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Sampling;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Multimodal;

/// <summary>End-to-end vision-language generation for Gemma-3 VLMs: encodes an image to a small set of embeddings
/// (<see cref="Gemma3VisionEncoder"/>), splices them into the text token-embedding sequence at the image
/// placeholder, and runs the (already-verified) Gemma-3 text decoder via its embedding-in path
/// (<see cref="GenericTransformer.ForwardEmbeds"/>). Greedy decode.
///
/// <para>Gemma-3 multiplies the combined (text + image) input embeddings by the √hidden normalizer. We apply that
/// scale to the text tokens via <see cref="GenericTransformer.EmbedLookup"/> and to the spliced image embeddings
/// directly, matching HF's <c>inputs_embeds.masked_scatter(image_mask, image_features)</c> followed by
/// <c>hidden_states = inputs_embeds * normalizer</c>.</para></summary>
public sealed class MultimodalGenerator
{
    private readonly GgufLanguageModel _text;
    private readonly IVlmImageEncoder _vision;
    private readonly IBackend _backend;

    public MultimodalGenerator(GgufLanguageModel text, IVlmImageEncoder vision, IBackend backend)
    {
        _text = text; _vision = vision; _backend = backend;
    }

    /// <summary>Builds the family-specific text segments that wrap the image embeddings: tokens before the image
    /// and tokens after. The image embeddings are spliced in between.</summary>
    private (int[] pre, int[] post) BuildPrompt(string question)
    {
        if (_vision.Family == "idefics3")
        {
            // SmolVLM / Idefics3: "<|im_start|>User:<fake_token_around_image><global-img> [image] <fake_token_around_image>{q}<end_of_utterance>\nAssistant:"
            int[] pre = _text.Tokenizer.Encode("<|im_start|>User:<fake_token_around_image><global-img>", addSpecial: true);
            int[] post = _text.Tokenizer.Encode($"<fake_token_around_image>{question}<end_of_utterance>\nAssistant:", addSpecial: true);
            return (pre, post);
        }
        if (_vision.Family == "qwen25vl")
        {
            // Qwen2.5-VL (ChatML): "<|im_start|>user\n<|vision_start|> [image] <|vision_end|>{q}<|im_end|>\n<|im_start|>assistant\n"
            int[] pre = _text.Tokenizer.Encode("<|im_start|>user\n<|vision_start|>", addSpecial: true);
            int[] post = _text.Tokenizer.Encode($"<|vision_end|>{question}<|im_end|>\n<|im_start|>assistant\n", addSpecial: true);
            return (pre, post);
        }
        if (_vision.Family == "internvl")
        {
            // InternVL (Qwen2 LLM, ChatML): "<|im_start|>system\n…<|im_end|>\n<|im_start|>user\n<img> [image] </img>\n{q}<|im_end|>\n<|im_start|>assistant\n"
            int[] pre = _text.Tokenizer.Encode(
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<img>", addSpecial: true);
            int[] post = _text.Tokenizer.Encode($"</img>\n{question}<|im_end|>\n<|im_start|>assistant\n", addSpecial: true);
            return (pre, post);
        }
        if (_vision.Family == "llava")
        {
            // LLaVA-1.5 (Vicuna v1): "<bos>{system} USER: [image]\n{q} ASSISTANT:"
            const string sys = "A chat between a curious human and an artificial intelligence assistant. " +
                "The assistant gives helpful, detailed, and polite answers to the human's questions.";
            int[] lpreText = _text.Tokenizer.Encode($"{sys} USER: ", addSpecial: true);
            int[] lpre = _text.Tokenizer.BosId is int lbos ? [lbos, .. lpreText] : lpreText;
            int[] lpost = _text.Tokenizer.Encode($"\n{question} ASSISTANT:", addSpecial: true);
            return (lpre, lpost);
        }
        // Gemma-3: <bos> + "<start_of_turn>user\n\n\n<start_of_image> [image] <end_of_image>\n\n{q}<end_of_turn>\n<start_of_turn>model\n"
        int[] preText = _text.Tokenizer.Encode("<start_of_turn>user\n\n\n<start_of_image>", addSpecial: true);
        int[] g = _text.Tokenizer.BosId is int bos ? [bos, .. preText] : preText;
        int[] gpost = _text.Tokenizer.Encode($"<end_of_image>\n\n{question}<end_of_turn>\n<start_of_turn>model\n", addSpecial: true);
        return (g, gpost);
    }

    /// <summary>Generates a reply to <paramref name="question"/> about the image given by normalized
    /// <paramref name="pixelValues"/> <c>[1, 3, imageSize, imageSize]</c>. Builds the Gemma-3 multimodal prompt
    /// (<c>&lt;start_of_turn&gt;user … &lt;start_of_image&gt; [image tokens] &lt;end_of_image&gt; question
    /// &lt;end_of_turn&gt;&lt;start_of_turn&gt;model</c>), splices the image embeddings, and greedy-decodes.</summary>
    public unsafe string Generate(Tensor pixelValues, string question, int maxTokens = 64, SamplingOptions? sampling = null)
    {
        // Greedy decode on small quantized VLMs is repetition-prone; default to a light sampler.
        SamplerChain sampler = SamplerChain.FromOptions(sampling ?? new SamplingOptions
        {
            Temperature = 0.4f, TopP = 0.9f, RepetitionPenalty = 1.1f, Seed = 1,
        });
        List<int> history = new();
        GenericTransformer model = _text.Transformer;
        int hidden = _text.Config.HiddenSize;

        // 1. Encode the image → [1, tokensPerImage, hidden].
        using Tensor imageEmbeds = _vision.Encode(_backend, pixelValues);
        int imgTokens = (int)imageEmbeds.Shape[1];
        bool dbg = Environment.GetEnvironmentVariable("HARTSY_VLM_DEBUG") == "1";
        if (dbg)
        {
            float* ie = (float*)imageEmbeds.DataPointer;
            long n = imageEmbeds.ElementCount; double sum = 0, max = 0; bool fin = true;
            for (long i = 0; i < n; i++) { float v = ie[i]; sum += v; max = Math.Max(max, Math.Abs(v)); if (!float.IsFinite(v)) fin = false; }
            Console.Error.WriteLine($"[vlm] imageEmbeds: tokens={imgTokens} dim={hidden} mean={sum / n:F4} maxabs={max:F4} finite={fin} first8=[{ie[0]:F3},{ie[1]:F3},{ie[2]:F3},{ie[3]:F3},{ie[4]:F3},{ie[5]:F3},{ie[6]:F3},{ie[7]:F3}]");
        }
        if (Environment.GetEnvironmentVariable("HARTSY_VLM_ENCODE_ONLY") == "1") return "";

        // 2. Tokenize the family-specific prompt segments that wrap the image.
        (int[] pre, int[] post) = BuildPrompt(question);

        // 3. Assemble the input embedding sequence: [pre tokens][image tokens][post tokens].
        int seqLen = pre.Length + imgTokens + post.Length;
        using Tensor embeds = new(new TensorShape(1, seqLen, hidden), DType.F32);
        float* dst = (float*)embeds.DataPointer;

        using (Tensor preEmb = new(new TensorShape(1, pre.Length, hidden), DType.F32))
        {
            model.EmbedLookup(preEmb, pre);   // text tokens carry the √hidden scale
            Buffer.MemoryCopy((void*)preEmb.DataPointer, dst, (long)pre.Length * hidden * 4, (long)pre.Length * hidden * 4);
        }
        // Gemma-3 applies the √hidden embedding normalizer to the *combined* (text+image) embeds. Our
        // EmbedLookup already scaled the text tokens, so scale the image embeds by the same factor here.
        long imgOff = (long)pre.Length * hidden;
        float scale = _text.Config.EmbeddingScale;
        float* imgSrc = (float*)imageEmbeds.DataPointer;   // D2H sync on CUDA
        for (long i = 0; i < (long)imgTokens * hidden; i++) dst[imgOff + i] = imgSrc[i] * scale;
        using (Tensor postEmb = new(new TensorShape(1, post.Length, hidden), DType.F32))
        {
            model.EmbedLookup(postEmb, post);
            long postOff = ((long)pre.Length + imgTokens) * hidden;
            Buffer.MemoryCopy((void*)postEmb.DataPointer, dst + postOff, (long)post.Length * hidden * 4, (long)post.Length * hidden * 4);
        }

        // 4. Prefill on the spliced embeddings, then greedy-decode token by token.
        HashSet<int> stops = new(_text.Tokenizer.StopIds);
        using FixedKvCache cache = new(model.Config.NumLayers, 1, model.Config.NumKvHeads, model.Config.HeadDim, seqLen + maxTokens + 1);
        System.Text.StringBuilder sb = new();

        int token;
        using (Tensor hiddenOut = model.ForwardEmbeds(_backend, embeds, seqLen, 0, cache))
            token = SampleLast(model, hiddenOut, seqLen, sampler, history);
        if (dbg) Console.Error.WriteLine($"[vlm] seqLen={seqLen} (pre={pre.Length} img={imgTokens} post={post.Length}) firstToken={token} decode=[{_text.Tokenizer.Decode([token])}]");

        int pos = seqLen;
        for (int step = 0; step < maxTokens && !stops.Contains(token); step++)
        {
            sb.Append(_text.Tokenizer.Decode([token]));
            history.Add(token);
            using Tensor stepEmb = new(new TensorShape(1, 1, model.Config.HiddenSize), DType.F32);
            model.EmbedLookup(stepEmb, new[] { token });
            using Tensor h = model.ForwardEmbeds(_backend, stepEmb, 1, pos, cache);
            pos++;
            token = SampleLast(model, h, 1, sampler, history);
        }
        return sb.ToString();
    }

    /// <summary>Projects the last position's hidden state to logits and selects the next token via the sampler
    /// (greedy when the options request it; otherwise temperature/top-p/repetition-penalty over the history).</summary>
    private unsafe int SampleLast(GenericTransformer model, Tensor hidden, int t, SamplerChain sampler, List<int> history)
    {
        int h = model.Config.HiddenSize;
        using Tensor last = new(new TensorShape(1, 1, h), DType.F32);
        float* hp = (float*)hidden.DataPointer;   // D2H sync
        Buffer.MemoryCopy(hp + (long)(t - 1) * h, (void*)last.DataPointer, (long)h * 4, (long)h * 4);
        using Tensor logits = model.ProjectLogits(_backend, last, 1);
        float* lp = (float*)logits.DataPointer;
        int vocab = model.Config.VocabSize;
        Span<float> span = new(lp, vocab);
        return sampler.Next(span, history);
    }
}
