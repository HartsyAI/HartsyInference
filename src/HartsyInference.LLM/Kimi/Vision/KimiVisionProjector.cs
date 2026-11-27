using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Kimi.Vision;

/// <summary>The patch merger (<c>PatchMergerMLPV2</c>): groups each run of <c>merge²</c> consecutive tokens, then bias-free linear, exact GELU,
/// bias-free linear, and an RMS norm. Tokens must already be in merge-block order, so a group is a contiguous slice.</summary>
internal static class KimiVisionProjector
{
    /// <summary>Projects <paramref name="tokens"/> <c>[np, hidden]</c> to <c>[1, np/merge², textHidden]</c>.</summary>
    public static Tensor Apply(IBackend backend, Tensor tokens, Tensor up, Tensor down, Tensor postNorm, int merge, float eps)
    {
        int np = (int)tokens.Shape[0], hidden = (int)tokens.Shape[1], group = merge * merge;
        if (np % group != 0) throw new ArgumentException($"{np} tokens do not tile {merge}x{merge} merge blocks.", nameof(tokens));
        int nTok = np / group, mergedDim = hidden * group, outDim = (int)down.Shape[0];
        if (up.Shape[0] != mergedDim || up.Shape[1] != mergedDim || down.Shape[1] != mergedDim)
            throw new ArgumentException($"The projector expects a merged width of {mergedDim}.", nameof(up));

        Tensor grouped = tokens.Reshape(new TensorShape(1, nTok, mergedDim));
        Tensor mid = new(new TensorShape(1, nTok, mergedDim), DType.F32);
        backend.Linear(mid, grouped, up, null);
        Tensor act = new(new TensorShape(1, nTok, mergedDim), DType.F32);
        backend.GeluErf(act, mid);
        mid.Dispose();
        Tensor proj = new(new TensorShape(1, nTok, outDim), DType.F32);
        backend.Linear(proj, act, down, null);
        act.Dispose();
        Tensor output = new(new TensorShape(1, nTok, outDim), DType.F32);
        backend.RmsNorm(output, proj, postNorm, eps);
        proj.Dispose();
        return output;
    }
}
