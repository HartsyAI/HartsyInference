using HartsyInference.Core.Backends;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Sampling;
using HartsyInference.LLM.Ssm;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.Multimodal;

/// <summary>Vision-language generation on the Qwen3.5-VL SSM backbone (<see cref="Qwen35Model"/>): splices image embeddings into the ChatML prompt, prefills via <see cref="Qwen35Model.ForwardEmbedsLastLogits"/>, then decodes one token at a time; unlike <see cref="MultimodalGenerator"/>'s transformer + external KV cache, the SSM decoder carries its own Gated-DeltaNet state internally, so decode is just repeated single-token <see cref="Qwen35Model.ForwardLastLogits"/> calls with no cache threaded.</summary>
public sealed class Qwen35VlGenerator
{
    private readonly Qwen35Model _model;
    private readonly ILlmTokenizer _tok;
    private readonly IVlmImageEncoder _vision;
    private readonly IBackend _backend;

    public Qwen35VlGenerator(Qwen35Model model, ILlmTokenizer tok, IVlmImageEncoder vision, IBackend backend)
    {
        _model = model; _tok = tok; _vision = vision; _backend = backend;
    }

    public unsafe string Generate(Tensor pixelValues, string question, int maxTokens = 64, SamplingOptions? sampling = null)
    {
        // Greedy decode on small quantized VLMs is repetition-prone; default to a light sampler (matches MultimodalGenerator).
        SamplerChain sampler = SamplerChain.FromOptions(sampling ?? new SamplingOptions
        {
            Temperature = 0.4f, TopP = 0.9f, RepetitionPenalty = 1.1f, Seed = 1,
        }, _tok, _model.VocabSize);
        List<int> history = new();
        int hidden = _model.DModel;

        // 1. Encode the image → [1, imgTokens, hidden].
        using Tensor imageEmbeds = _vision.Encode(_backend, pixelValues);
        int imgTokens = (int)imageEmbeds.Shape[1];
        // 2. Qwen-VL ChatML wrapping (same token layout as qwen25vl):
        //    "<|im_start|>user\n<|vision_start|> [image] <|vision_end|>{q}<|im_end|>\n<|im_start|>assistant\n"
        int[] pre = _tok.Encode("<|im_start|>user\n<|vision_start|>", addSpecial: true);
        int[] post = _tok.Encode($"<|vision_end|>{question}<|im_end|>\n<|im_start|>assistant\n", addSpecial: true);
        int seqLen = pre.Length + imgTokens + post.Length;

        // 3. Assemble the spliced input-embedding sequence [pre][image][post] (host F32).
        using Tensor embeds = new(new TensorShape(1, seqLen, hidden), DType.F32);
        float* dst = (float*)embeds.DataPointer;
        using (Tensor preEmb = new(new TensorShape(1, pre.Length, hidden), DType.F32))
        {
            _model.EmbedLookup(preEmb, pre);
            Buffer.MemoryCopy((void*)preEmb.DataPointer, dst, (long)pre.Length * hidden * 4, (long)pre.Length * hidden * 4);
        }
        _backend.Sync();
        float* imgSrc = (float*)imageEmbeds.DataPointer;   // D2H sync on CUDA
        Buffer.MemoryCopy(imgSrc, dst + (long)pre.Length * hidden, (long)imgTokens * hidden * 4, (long)imgTokens * hidden * 4);
        using (Tensor postEmb = new(new TensorShape(1, post.Length, hidden), DType.F32))
        {
            _model.EmbedLookup(postEmb, post);
            long postOff = ((long)pre.Length + imgTokens) * hidden;
            Buffer.MemoryCopy((void*)postEmb.DataPointer, dst + postOff, (long)post.Length * hidden * 4, (long)post.Length * hidden * 4);
        }

        // 4. Prefill on the spliced embeddings, then single-token decode (state carries in the model).
        HashSet<int> stops = new(_tok.StopIds);
        System.Text.StringBuilder sb = new();

        _model.ResetState();
        float[] logits;
        {
            // Multimodal RoPE: the image's placeholder run takes (start + t, start + row, start + col) over its merged grid and the text after it
            // resumes at start + max(T, H, W), not at the sequence index. The decode steps below follow that offset.
            int side = (int)Math.Round(Math.Sqrt(imgTokens));
            if (side * side != imgTokens)
                throw new NotSupportedException($"The image encoder returned {imgTokens} tokens, which is not a square merged grid.");
            const int placeholder = -1;
            int[] layout = new int[seqLen];
            for (int i = 0; i < seqLen; i++) layout[i] = i >= pre.Length && i < pre.Length + imgTokens ? placeholder : 0;
            int[] positions = MRopePositions.Build(layout, placeholder, [new MRopeImageGrid(1, side, side)], out int nextPosition);
            (Tensor cos, Tensor sin) = _model.BuildMultimodalRope(positions, seqLen, 0, seqLen);
            using (cos)
            using (sin)
            {
                logits = _model.ForwardEmbedsLastLogits(_backend, embeds, seqLen, cos, sin, nextPosition);
            }
        }
        int token = sampler.Next(new Span<float>(logits), history);

        for (int step = 0; step < maxTokens && !stops.Contains(token); step++)
        {
            sb.Append(_tok.Decode(new[] { token }));
            history.Add(token);
            logits = _model.ForwardLastLogits(_backend, new[] { token });
            token = sampler.Next(new Span<float>(logits), history);
        }
        return sb.ToString();
    }
}
