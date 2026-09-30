namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>One of a routed expert's three projections, named as the checkpoint does.</summary>
public enum DeepSeekV41ExpertProjection
{
    /// <summary><c>w1</c>: the SwiGLU gate projection.</summary>
    W1,

    /// <summary><c>w2</c>: the down projection.</summary>
    W2,

    /// <summary><c>w3</c>: the SwiGLU up projection.</summary>
    W3,
}
