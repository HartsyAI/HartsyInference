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
