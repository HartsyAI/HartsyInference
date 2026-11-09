using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Sampling;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Multimodal;

/// <summary>Decode-loop glue shared by the VLM generators (<see cref="MultimodalGenerator"/>, <see cref="MllamaGenerator"/>), whose prefill/step loops differ but whose token selection does not.</summary>
internal static unsafe class VlmDecodeOps
{
    /// <summary>Projects the last position's hidden state to logits and selects the next token via the sampler (greedy when the options request it; otherwise temperature/top-p/repetition-penalty over the history).</summary>
    public static int SampleLast(IBackend backend, GenericTransformer model, Tensor hidden, int t, SamplerChain sampler, List<int> history)
    {
        using Tensor logits = ProjectLast(backend, model, hidden, t);
        float* lp = (float*)logits.DataPointer;
        Span<float> span = new(lp, model.Config.VocabSize);
        return sampler.Next(span, history);
    }

    /// <summary>The vocabulary logits of the last position of <paramref name="hidden"/> (<c>[1, t, hidden]</c>), copied to a host array.
    /// Only callers that keep the logits past the step need this copy; <see cref="SampleLast"/> reads the device buffer in place.</summary>
    public static float[] LastLogits(IBackend backend, GenericTransformer model, Tensor hidden, int t)
    {
        using Tensor logits = ProjectLast(backend, model, hidden, t);
        return new ReadOnlySpan<float>((void*)logits.DataPointer, model.Config.VocabSize).ToArray();
    }

    /// <summary>The decode loop every VLM reply runs after its prefill: emit until a stop token or <paramref name="maxTokens"/>, feeding each
    /// token back through <paramref name="step"/>, streaming text through <paramref name="onDelta"/> once a character is complete, and
    /// checking <paramref name="cancel"/> before every step.</summary>
    /// <param name="first">The token the prefill selected.</param>
    /// <param name="step">Runs one token and returns the next one.</param>
    /// <returns>The reply, how many tokens it holds, and whether it stopped at the limit rather than at a stop token.</returns>
    public static (string Text, int Tokens, bool HitLimit) DecodeLoop(ModelAssets.Tokenizers.ILlmTokenizer tokenizer, int first,
        Func<int, int> step, IReadOnlySet<int> stops, int maxTokens, List<int> history, Action<string>? onDelta, CancellationToken cancel)
    {
        List<int> generated = [];
        int emitted = 0;
        int token = first;
        while (!stops.Contains(token) && generated.Count < maxTokens)
        {
            generated.Add(token);
            history.Add(token);
            if (onDelta is not null)
            {
                string text = tokenizer.Decode(generated);
                // A token can end inside a multi-byte character; hold the piece until the character completes.
                if (text.Length > emitted && text[^1] != '�')
                {
                    onDelta(text[emitted..]);
                    emitted = text.Length;
                }
            }
            if (generated.Count == maxTokens) break;
            cancel.ThrowIfCancellationRequested();
            token = step(token);
        }
        string reply = tokenizer.Decode(generated);
        if (onDelta is not null && reply.Length > emitted) onDelta(reply[emitted..]);
        return (reply, generated.Count, !stops.Contains(token));
    }

    /// <summary>Projects the last position to logits; the caller owns and disposes the returned tensor.</summary>
    private static Tensor ProjectLast(IBackend backend, GenericTransformer model, Tensor hidden, int t)
    {
        int h = model.Config.HiddenSize;
        using Tensor last = new(new TensorShape(1, 1, h), DType.F32);
        float* hp = (float*)hidden.DataPointer;   // D2H sync
        Buffer.MemoryCopy(hp + (long)(t - 1) * h, (void*)last.DataPointer, (long)h * 4, (long)h * 4);
        return model.ProjectLogits(backend, last, 1);
    }
}
