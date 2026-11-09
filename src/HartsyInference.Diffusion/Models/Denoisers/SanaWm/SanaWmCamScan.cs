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
    /// <summary>Writes the bidirectional scan of <paramref name="query"/>, <paramref name="key"/> and <paramref name="value"/>
    /// into <paramref name="output"/>. Operands are F32 <c>[B, N, H·D]</c>; <paramref name="beta"/> is <c>[B, H, T, S]</c>
    /// and <paramref name="decay"/> is <c>[B, H, T]</c>.</summary>
    internal static void Run(IBackend backend, Tensor output, Tensor query, Tensor key, Tensor value, Tensor beta, Tensor decay,
        int batch, int frames, int spatial, int heads, int headDim)
    {
        int bh = batch * heads;
        int stateRows = frames * bh;
        TensorShape stateShape = new(bh, headDim, headDim);
        using Tensor identity = IdentityBatch(backend, bh, headDim);
        using Tensor transitions = new(new TensorShape(stateRows, headDim, headDim), DType.F32, backend.Device);
        using Tensor injections = new(new TensorShape(stateRows, headDim, headDim), DType.F32, backend.Device);
        for (int f = 0; f < frames; f++)
        {
            BuildFrame(backend, key, value, beta, decay, identity, transitions, injections, f,
                batch, frames, spatial, heads, headDim);
        }

        using Tensor sigmaA = new(stateShape, DType.F32, backend.Device);
        using Tensor sigmaB = new(stateShape, DType.F32, backend.Device);
        using Tensor scratch = new(stateShape, DType.F32, backend.Device);
        using Tensor framed = new(new TensorShape(bh, spatial, headDim), DType.F32, backend.Device);

        backend.Fill(sigmaA, 0f);
        Tensor current = sigmaA;
        Tensor next = sigmaB;
        for (int f = 0; f < frames; f++)
        {
            Advance(backend, current, next, scratch, transitions, injections, f, bh);
            (current, next) = (next, current);
            using Tensor queryFrame = GatherFrame(backend, query, f, batch, frames, spatial, heads, headDim);
            backend.BatchedMatMul(framed, queryFrame, current);
            backend.CopyStrided(output, framed, StridedCopySpec.Scatter(FrameView(f, batch, frames, spatial, heads, headDim)));
        }

        backend.Fill(sigmaA, 0f);
        current = sigmaA;
        next = sigmaB;
        for (int f = frames - 1; f >= 0; f--)
        {
            using Tensor queryFrame = GatherFrame(backend, query, f, batch, frames, spatial, heads, headDim);
            backend.BatchedMatMul(framed, queryFrame, current);
            backend.AccumulateStrided(output, framed, StridedCopySpec.Scatter(FrameView(f, batch, frames, spatial, heads, headDim)));
            Advance(backend, current, next, scratch, transitions, injections, f, bh);
            (current, next) = (next, current);
        }
    }

    private static void BuildFrame(IBackend backend, Tensor key, Tensor value, Tensor beta, Tensor decay, Tensor identity,
        Tensor transitions, Tensor injections, int frame, int batch, int frames, int spatial, int heads, int headDim)
    {
        int bh = batch * heads;
        TensorShape tokenShape = new(bh, spatial, headDim);
        using Tensor keys = GatherFrame(backend, key, frame, batch, frames, spatial, heads, headDim);
        using Tensor values = GatherFrame(backend, value, frame, batch, frames, spatial, heads, headDim);
        using Tensor betas = BroadcastFrame(backend, beta, frame, batch, frames, spatial, heads, headDim);
        using Tensor weighted = new(tokenShape, DType.F32, backend.Device);
        backend.Mul(weighted, keys, betas);
        using Tensor weightedT = new(new TensorShape(bh, headDim, spatial), DType.F32, backend.Device);
        backend.CopyStrided(weightedT, weighted, StridedCopySpec.Gather(StridedView.Of(weighted.Shape).Permute([0, 2, 1])));

        TensorShape squareShape = new(bh, headDim, headDim);
        using Tensor gram = new(squareShape, DType.F32, backend.Device);
        backend.BatchedMatMul(gram, weightedT, keys);
        using Tensor injection = new(squareShape, DType.F32, backend.Device);
        backend.BatchedMatMul(injection, weightedT, values);

        using Tensor decays = BroadcastDecay(backend, decay, frame, batch, frames, heads, headDim);
        using Tensor negated = new(squareShape, DType.F32, backend.Device);
        backend.Scale(negated, gram, -1f);
        using Tensor residual = new(squareShape, DType.F32, backend.Device);
        backend.Add(residual, negated, identity);
        using Tensor transition = new(squareShape, DType.F32, backend.Device);
        backend.Mul(transition, residual, decays);

        backend.CopyTo(transitions.SliceRows(frame * bh, bh), transition);
        backend.CopyTo(injections.SliceRows(frame * bh, bh), injection);
    }

    /// <summary><c>next = transition_f · current + injection_f</c>.</summary>
    private static void Advance(IBackend backend, Tensor current, Tensor next, Tensor scratch, Tensor transitions,
        Tensor injections, int frame, int bh)
    {
        backend.BatchedMatMul(scratch, transitions.SliceRows(frame * bh, bh), current);
        backend.Add(next, scratch, injections.SliceRows(frame * bh, bh));
    }

    /// <summary>Gathers one frame of a <c>[B, N, H·D]</c> tensor into <c>[B·H, S, D]</c>.</summary>
    private static Tensor GatherFrame(IBackend backend, Tensor source, int frame, int batch, int frames, int spatial,
        int heads, int headDim)
    {
        Tensor packed = new(new TensorShape(batch * heads, spatial, headDim), DType.F32, backend.Device);
        backend.CopyStrided(packed, source, StridedCopySpec.Gather(FrameView(frame, batch, frames, spatial, heads, headDim)));
        return packed;
    }

    /// <summary>The frame's tokens as a <c>[B, H, S, D]</c> view of a <c>[B, N, H·D]</c> tensor.</summary>
    private static StridedView FrameView(int frame, int batch, int frames, int spatial, int heads, int headDim)
    {
        long channels = (long)heads * headDim;
        long tokens = (long)frames * spatial;
        ReadOnlySpan<long> dims = [batch, heads, spatial, headDim];
        ReadOnlySpan<long> strides = [tokens * channels, headDim, channels, 1];
        return new StridedView(dims, strides, (long)frame * spatial * channels);
    }

    /// <summary>The frame's per-token betas, broadcast over the head dim, as <c>[B·H, S, D]</c>.</summary>
    private static Tensor BroadcastFrame(IBackend backend, Tensor beta, int frame, int batch, int frames, int spatial,
        int heads, int headDim)
    {
        ReadOnlySpan<long> dims = [batch, heads, spatial, headDim];
        ReadOnlySpan<long> strides = [(long)heads * frames * spatial, (long)frames * spatial, 1, 0];
        Tensor packed = new(new TensorShape(batch * heads, spatial, headDim), DType.F32, backend.Device);
        backend.CopyStrided(packed, beta, StridedCopySpec.Gather(new StridedView(dims, strides, (long)frame * spatial)));
        return packed;
    }

    /// <summary>The frame's decay, broadcast over the square state, as <c>[B·H, D, D]</c>.</summary>
    private static Tensor BroadcastDecay(IBackend backend, Tensor decay, int frame, int batch, int frames, int heads, int headDim)
    {
        ReadOnlySpan<long> dims = [batch, heads, headDim, headDim];
        ReadOnlySpan<long> strides = [(long)heads * frames, frames, 0, 0];
        Tensor packed = new(new TensorShape(batch * heads, headDim, headDim), DType.F32, backend.Device);
        backend.CopyStrided(packed, decay, StridedCopySpec.Gather(new StridedView(dims, strides, frame)));
        return packed;
    }

    private static Tensor IdentityBatch(IBackend backend, int bh, int headDim)
    {
        using Tensor hostEye = new(new TensorShape(bh, headDim, headDim), DType.F32, DeviceKind.Cpu);
        Span<float> eye = hostEye.AsSpan<float>();
        eye.Clear();
        for (int b = 0; b < bh; b++)
            for (int d = 0; d < headDim; d++)
                eye[(b * headDim + d) * headDim + d] = 1f;
        Tensor device = new(hostEye.Shape, DType.F32, backend.Device);
        backend.CopyTo(device, hostEye);
        return device;
    }
}
