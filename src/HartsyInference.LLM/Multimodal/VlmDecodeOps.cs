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
        int h = model.Config.HiddenSize;
        using Tensor last = new(new TensorShape(1, 1, h), DType.F32);
        float* hp = (float*)hidden.DataPointer;   // D2H sync
        Buffer.MemoryCopy(hp + (long)(t - 1) * h, (void*)last.DataPointer, (long)h * 4, (long)h * 4);
        using Tensor logits = model.ProjectLogits(backend, last, 1);
        float* lp = (float*)logits.DataPointer;
        Span<float> span = new(lp, model.Config.VocabSize);
        return sampler.Next(span, history);
    }
}
