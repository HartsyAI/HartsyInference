using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Transformer.Mhc;

/// <summary>One sublayer wrapped in a manifold-constrained hyper-connection over a <c>[tokens, hc, hidden]</c> residual stream: the
/// stream's mixes collapse it to one row per token, the row is RMS-normalized and run through <paramref name="body"/>, and the output is
/// expanded back into the stream. Any family that uses <see cref="HyperConnection"/> runs its sublayers through this.</summary>
public static class HyperConnectionSublayer
{
    /// <summary>Runs one wrapped sublayer, updating <paramref name="stream"/> in place.</summary>
    /// <param name="norm">The RMS-norm weight applied to the collapsed row.</param>
    /// <param name="hidden">Hidden width.</param>
    /// <param name="eps">RMS-norm epsilon.</param>
    /// <param name="x">Scratch for the collapsed rows, <c>[tokens, hidden]</c>.</param>
    /// <param name="sub">Scratch for the sublayer output, <c>[tokens, hidden]</c>.</param>
    /// <param name="body">The sublayer: reads <c>input</c> rows and writes <c>output</c> rows, both <c>[tokens, hidden]</c>.</param>
    public static void Run(IBackend backend, HyperConnection hc, Tensor norm, int hidden, float eps, float[] stream, int tokens,
        float[] x, float[] sub, Action<float[], float[]> body)
    {
        int h = hc.Hc;
        float[] pre = new float[tokens * h], post = new float[tokens * h], comb = new float[tokens * h * h];
        hc.Mixes(stream, tokens, pre, post, comb);
        hc.Collapse(stream, pre, tokens, x);
        HostOps.RmsNorm(backend, x, hidden, norm, eps);
        body(x, sub);
        hc.Expand(sub, stream, post, comb, tokens, stream);
    }
}
