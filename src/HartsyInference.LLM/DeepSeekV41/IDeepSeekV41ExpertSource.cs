namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>Supplies a layer's routed experts as dequantized F32 weights, on demand.</summary>
/// <remarks>The executor asks only for experts a token actually routed to, so an implementation can dequantize lazily and cache.</remarks>
public interface IDeepSeekV41ExpertSource
{
    /// <summary>The weights of routed expert <paramref name="expert"/>; the caller does not modify or retain them.</summary>
    DeepSeekV41SwigluWeights GetExpert(int expert);
}
