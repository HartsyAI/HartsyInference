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
    private readonly Qwen3VlEncoder _vision;
    private readonly IBackend _backend;

    public Qwen35VlGenerator(Qwen35Model model, ILlmTokenizer tok, Qwen3VlEncoder vision, IBackend backend)
    {
        _model = model; _tok = tok; _vision = vision; _backend = backend;
    }

    /// <summary>Generates an answer about <paramref name="pixelValues"/>. The image's merged grid is read from those pixels via the Qwen3-VL tower
    /// (<see cref="Qwen3VlEncoder.MergedGrid"/>), never inferred from the token count: a 2×8 grid and a 4×4 grid both hold 16 tokens.</summary>
    public unsafe string Generate(Tensor pixelValues, string question, int maxTokens = 64, SamplingOptions? sampling = null)
    {
        MRopeImageGrid grid = Qwen3VlEncoder.MergedGrid(pixelValues, _vision.PatchSize, _vision.Merge);
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
            int[] positions = BuildPositions(seqLen, pre.Length, imgTokens, grid, out int nextPosition);
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

    /// <summary>Axis-major M-RoPE positions <c>[3 · seqLen]</c> for <c>[pre][image][post]</c>: the image's placeholder run at
    /// <paramref name="imageStart"/> takes <c>(start + t, start + row, start + col)</c> over <paramref name="grid"/>, and the text after it
    /// resumes at <c>start + max(T, H, W)</c>, not at the sequence index. Refuses a grid that does not hold exactly the encoder's tokens.</summary>
    internal static int[] BuildPositions(int seqLen, int imageStart, int imageTokens, MRopeImageGrid grid, out int nextPosition)
    {
        if (grid.Tokens != imageTokens)
            throw new ArgumentException($"The image grid {grid} holds {grid.Tokens} tokens but the encoder returned {imageTokens}.", nameof(grid));
        if (imageStart < 0 || imageStart + imageTokens > seqLen)
            throw new ArgumentOutOfRangeException(nameof(imageStart), "The image run does not fit in the sequence.");
        const int placeholder = -1;
        int[] layout = new int[seqLen];
        for (int i = 0; i < seqLen; i++) layout[i] = i >= imageStart && i < imageStart + imageTokens ? placeholder : 0;
        return MRopePositions.Build(layout, placeholder, [grid], out nextPosition);
    }
}
