using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.SanaWm;

/// <summary>Numerator-only camera delta-rule scan over frames, as upstream <c>_torch_cam_scan_single_chunk</c> and
/// <c>_torch_cam_scan_reference</c>. Per frame <c>f</c> with <c>Σ = Sᵀ</c> (<c>[Dk, Dv]</c>):
/// <c>Σ_f = d_f·(I − G_f)·Σ_{f−1} + U_fᵀ</c>, <c>G_f = Σ_s β k kᵀ</c>, <c>U_f = Σ_s β k vᵀ</c>, and the token output is
/// <c>q·Σ_f</c>. The forward pass is inclusive; the reverse pass reads the state before frame <c>f</c>'s update, so it
/// sums frames strictly after <c>f</c>. The output is the forward result plus the reverse result.</summary>
internal static class SanaWmCamScan
{
    /// <summary>Returns the bidirectional scan of one batch item's <paramref name="query"/>, <paramref name="key"/> and
    /// <paramref name="value"/>, F32 token-major <c>[N, H·D]</c>, as a new <c>[N, H·D]</c> tensor. <paramref name="beta"/> is
    /// <c>[H, T, S]</c> and <paramref name="decay"/> is <c>[H, T]</c>.</summary>
    internal static Tensor Run(IBackend backend, Tensor query, Tensor key, Tensor value, Tensor beta, Tensor decay, int frames,
        int spatial, int heads, int headDim)
    {
        long tokens = (long)frames * spatial;
        long channels = (long)heads * headDim;
        TensorShape stateShape = new(heads, headDim, headDim);
        using SanaWmScratch scratch = new();
        Tensor identity = scratch.Track(SanaWmTensors.Stage(backend, Identity(heads, headDim)));
        Tensor[] transitions = new Tensor[frames];
        Tensor[] injections = new Tensor[frames];
        for (int f = 0; f < frames; f++)
            (transitions[f], injections[f]) = BuildFrame(backend, scratch, key, value, beta, decay, identity, f, frames, spatial, heads, headDim);

        Tensor output = scratch.New(tokens, channels);
        Tensor current = scratch.Track(SanaWmTensors.Filled(stateShape, 0f));
        for (int f = 0; f < frames; f++)
        {
            current = Advance(backend, scratch, current, transitions[f], injections[f]);
            using SanaWmScratch step = new();
            Tensor framed = step.New(heads, spatial, headDim);
            backend.BatchedMatMul(framed, GatherFrame(backend, step, query, f, spatial, heads, headDim), current);
            Tensor rows = step.New(spatial, channels);
            backend.Permute0213(rows, framed, heads, spatial, headDim);
            SanaWmTensors.Write(backend, output, rows, f * spatial * channels);
        }

        scratch.Release(current);
        current = scratch.Track(SanaWmTensors.Filled(stateShape, 0f));
        for (int f = frames - 1; f >= 0; f--)
        {
            using (SanaWmScratch step = new())
            {
                Tensor framed = step.New(heads, spatial, headDim);
                backend.BatchedMatMul(framed, GatherFrame(backend, step, query, f, spatial, heads, headDim), current);
                backend.AccumulateStrided(output, framed, StridedCopySpec.Scatter(FrameView(f, spatial, heads, headDim)));
            }

            current = Advance(backend, scratch, current, transitions[f], injections[f]);
        }

        return scratch.Detach(output);
    }

    // transition_f = d_f·(I − Σ_s β k kᵀ) and injection_f = Σ_s β k vᵀ, both [H, D, D] and tracked by the scope.
    private static (Tensor Transition, Tensor Injection) BuildFrame(IBackend backend, SanaWmScratch scratch, Tensor key, Tensor value,
        Tensor beta, Tensor decay, Tensor identity, int frame, int frames, int spatial, int heads, int headDim)
    {
        using SanaWmScratch step = new();
        Tensor keys = GatherFrame(backend, step, key, frame, spatial, heads, headDim);
        Tensor values = GatherFrame(backend, step, value, frame, spatial, heads, headDim);
        Tensor betas = step.New(heads, spatial, headDim);
        ReadOnlySpan<long> betaDims = [heads, spatial, headDim];
        ReadOnlySpan<long> betaStrides = [(long)frames * spatial, 1, 0];
        backend.CopyStrided(betas, beta, StridedCopySpec.Gather(new StridedView(betaDims, betaStrides, (long)frame * spatial)));
        Tensor weighted = step.New(heads, spatial, headDim);
        backend.Mul(weighted, keys, betas);
        Tensor weightedT = step.New(heads, headDim, spatial);
        backend.CopyStrided(weightedT, weighted, StridedCopySpec.Gather(StridedView.Of(weighted.Shape).Permute([0, 2, 1])));

        Tensor gram = step.New(heads, headDim, headDim);
        backend.BatchedMatMul(gram, weightedT, keys);
        Tensor injection = scratch.New(heads, headDim, headDim);
        backend.BatchedMatMul(injection, weightedT, values);

        Tensor decays = step.New(heads, headDim, headDim);
        ReadOnlySpan<long> decayDims = [heads, headDim, headDim];
        ReadOnlySpan<long> decayStrides = [frames, 0, 0];
        backend.CopyStrided(decays, decay, StridedCopySpec.Gather(new StridedView(decayDims, decayStrides, frame)));
        Tensor negated = step.New(heads, headDim, headDim);
        backend.Scale(negated, gram, -1f);
        Tensor residual = step.New(heads, headDim, headDim);
        backend.Add(residual, negated, identity);
        Tensor transition = scratch.New(heads, headDim, headDim);
        backend.Mul(transition, residual, decays);
        return (transition, injection);
    }

    /// <summary>Returns <c>transition_f · current + injection_f</c> tracked by <paramref name="scratch"/>, releasing
    /// <paramref name="current"/>.</summary>
    private static Tensor Advance(IBackend backend, SanaWmScratch scratch, Tensor current, Tensor transition, Tensor injection)
    {
        Tensor product = scratch.New(current.Shape);
        backend.BatchedMatMul(product, transition, current);
        Tensor next = scratch.New(current.Shape);
        backend.Add(next, product, injection);
        scratch.Release(product);
        scratch.Release(current);
        return next;
    }

    /// <summary>Gathers one frame of a token-major <c>[N, H·D]</c> tensor into a tracked <c>[H, S, D]</c>.</summary>
    private static Tensor GatherFrame(IBackend backend, SanaWmScratch scratch, Tensor source, int frame, int spatial, int heads,
        int headDim)
    {
        Tensor packed = scratch.New(heads, spatial, headDim);
        backend.CopyStrided(packed, source, StridedCopySpec.Gather(FrameView(frame, spatial, heads, headDim)));
        return packed;
    }

    /// <summary>The frame's tokens as a <c>[H, S, D]</c> view of a token-major <c>[N, H·D]</c> tensor.</summary>
    private static StridedView FrameView(int frame, int spatial, int heads, int headDim)
    {
        long channels = (long)heads * headDim;
        ReadOnlySpan<long> dims = [heads, spatial, headDim];
        ReadOnlySpan<long> strides = [headDim, channels, 1];
        return new StridedView(dims, strides, (long)frame * spatial * channels);
    }

    private static Tensor Identity(int heads, int headDim)
    {
        Tensor eye = SanaWmTensors.Filled(new TensorShape(heads, headDim, headDim), 0f);
        Span<float> values = eye.AsSpan<float>();
        for (int h = 0; h < heads; h++)
            for (int d = 0; d < headDim; d++)
                values[(h * headDim + d) * headDim + d] = 1f;
        return eye;
    }
}
