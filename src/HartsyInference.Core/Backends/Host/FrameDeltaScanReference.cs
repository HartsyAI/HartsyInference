using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Host;

/// <summary>Host reference for <see cref="IBackend.FrameDeltaScan"/>: per head, the frame Gram <c>G_t</c> and injection
/// <c>U_t</c>, the forward and reverse state recurrences, and the token readout, in plain F32 loops. Heads fan out; within
/// a head every sum runs in index order.</summary>
public static class FrameDeltaScanReference
{
    /// <summary>Validates the operands of <see cref="IBackend.FrameDeltaScan"/>.</summary>
    public static void Validate(Tensor output, Tensor query, Tensor key, Tensor? value, Tensor beta, Tensor decay, int frames,
        int spatial, int heads, int keyDim, int valueDim, Tensor? initialState = null, Tensor? finalState = null)
    {
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(query);
        ArgumentNullException.ThrowIfNull(key);
        ArgumentNullException.ThrowIfNull(beta);
        ArgumentNullException.ThrowIfNull(decay);
        if (frames < 1 || spatial < 1 || heads < 1 || keyDim < 1 || valueDim < 1)
            throw new ArgumentException($"FrameDeltaScan needs positive dims, got frames={frames}, spatial={spatial}, heads={heads}, "
                + $"keyDim={keyDim}, valueDim={valueDim}.");
        if (value is null && valueDim != 1)
            throw new ArgumentException($"FrameDeltaScan without a value tensor scans a unit value; valueDim must be 1, got {valueDim}.",
                nameof(valueDim));
        foreach ((Tensor tensor, string name) in new[] { (output, nameof(output)), (query, nameof(query)), (key, nameof(key)),
            (beta, nameof(beta)), (decay, nameof(decay)) })
        {
            if (tensor.DType != DType.F32)
                throw new ArgumentException($"FrameDeltaScan {name} must be F32, got {tensor.DType.Name}.", name);
        }

        if (value is not null && value.DType != DType.F32)
            throw new ArgumentException($"FrameDeltaScan value must be F32, got {value.DType.Name}.", nameof(value));
        long tokens = (long)frames * spatial;
        Require(query, nameof(query), tokens * heads * keyDim);
        Require(key, nameof(key), tokens * heads * keyDim);
        Require(output, nameof(output), tokens * heads * valueDim);
        if (value is not null)
            Require(value, nameof(value), tokens * heads * valueDim);
        Require(beta, nameof(beta), tokens * heads);
        Require(decay, nameof(decay), (long)frames * heads);
        RequireState(initialState, nameof(initialState), heads, keyDim, valueDim);
        RequireState(finalState, nameof(finalState), heads, keyDim, valueDim);
        if (initialState is not null && ReferenceEquals(initialState, finalState))
            throw new ArgumentException("FrameDeltaScan finalState must not alias initialState.", nameof(finalState));
    }

    /// <summary>Host <see cref="IBackend.FrameDeltaScan"/>.</summary>
    public static unsafe void Run(Tensor output, Tensor query, Tensor key, Tensor? value, Tensor beta, Tensor decay, int frames,
        int spatial, int heads, int keyDim, int valueDim, Tensor? initialState = null, Tensor? finalState = null)
    {
        Validate(output, query, key, value, beta, decay, frames, spatial, heads, keyDim, valueDim, initialState, finalState);
        float* outPtr = (float*)output.DataPointer;
        float* qPtr = (float*)query.DataPointer;
        float* kPtr = (float*)key.DataPointer;
        float* vPtr = value is null ? null : (float*)value.DataPointer;
        float* betaPtr = (float*)beta.DataPointer;
        float* decayPtr = (float*)decay.DataPointer;
        float* initialPtr = initialState is null ? null : (float*)initialState.DataPointer;
        float* finalPtr = finalState is null ? null : (float*)finalState.DataPointer;
        Pointers p = new((nint)outPtr, (nint)qPtr, (nint)kPtr, (nint)vPtr, (nint)betaPtr, (nint)decayPtr, (nint)initialPtr, (nint)finalPtr);
        Shape shape = new(frames, spatial, heads, keyDim, valueDim);
        long work = (long)frames * spatial * keyDim * (keyDim + 2L * valueDim);
        CpuParallel.For(heads, work * heads, (p, shape), static (h, state) => Head(state.p, state.shape, h));
    }

    private static unsafe void Head(Pointers p, Shape s, int h)
    {
        int dk = s.KeyDim, dv = s.ValueDim, frames = s.Frames, spatial = s.Spatial;
        long keyRow = (long)s.Heads * dk, valueRow = (long)s.Heads * dv;
        float* q = (float*)p.Query, k = (float*)p.Key, v = (float*)p.Value, beta = (float*)p.Beta, decay = (float*)p.Decay;
        float* output = (float*)p.Output;
        float[] grams = new float[(long)frames * dk * dk];
        float[] injections = new float[(long)frames * dk * dv];
        for (int t = 0; t < frames; t++)
        {
            Span<float> gram = grams.AsSpan(t * dk * dk, dk * dk);
            Span<float> injection = injections.AsSpan(t * dk * dv, dk * dv);
            for (int sIndex = 0; sIndex < spatial; sIndex++)
            {
                long token = (long)t * spatial + sIndex;
                float b = beta[((long)h * frames + t) * spatial + sIndex];
                float* kr = k + token * keyRow + (long)h * dk;
                float* vr = v == null ? null : v + token * valueRow + (long)h * dv;
                for (int r = 0; r < dk; r++)
                {
                    float bk = b * kr[r];
                    for (int j = 0; j < dk; j++)
                        gram[r * dk + j] += bk * kr[j];
                    for (int c = 0; c < dv; c++)
                        injection[r * dv + c] += bk * (vr == null ? 1f : vr[c]);
                }
            }
        }

        float[] state = new float[dk * dv];
        float[] next = new float[dk * dv];
        if (p.Initial != 0)
            new ReadOnlySpan<float>((float*)p.Initial + (long)h * dk * dv, dk * dv).CopyTo(state);
        for (int t = 0; t < frames; t++)
        {
            Advance(state, next, grams.AsSpan(t * dk * dk, dk * dk), injections.AsSpan(t * dk * dv, dk * dv), decay[(long)h * frames + t], dk, dv);
            (state, next) = (next, state);
            Readout(output, q, state, t, h, s, accumulate: false);
        }

        if (p.Final != 0)
            state.AsSpan().CopyTo(new Span<float>((float*)p.Final + (long)h * dk * dv, dk * dv));

        Array.Clear(state);
        for (int t = frames - 1; t >= 0; t--)
        {
            Readout(output, q, state, t, h, s, accumulate: true);
            Advance(state, next, grams.AsSpan(t * dk * dk, dk * dk), injections.AsSpan(t * dk * dv, dk * dv), decay[(long)h * frames + t], dk, dv);
            (state, next) = (next, state);
        }
    }

    // next = d·(state − G·state) + U.
    private static void Advance(float[] state, float[] next, ReadOnlySpan<float> gram, ReadOnlySpan<float> injection, float d, int dk, int dv)
    {
        for (int r = 0; r < dk; r++)
        {
            for (int c = 0; c < dv; c++)
            {
                float acc = 0f;
                for (int j = 0; j < dk; j++)
                    acc += gram[r * dk + j] * state[j * dv + c];
                next[r * dv + c] = d * (state[r * dv + c] - acc) + injection[r * dv + c];
            }
        }
    }

    private static unsafe void Readout(float* output, float* q, float[] state, int t, int h, Shape s, bool accumulate)
    {
        int dk = s.KeyDim, dv = s.ValueDim;
        long keyRow = (long)s.Heads * dk, valueRow = (long)s.Heads * dv;
        for (int sIndex = 0; sIndex < s.Spatial; sIndex++)
        {
            long token = (long)t * s.Spatial + sIndex;
            float* qr = q + token * keyRow + (long)h * dk;
            float* orow = output + token * valueRow + (long)h * dv;
            for (int c = 0; c < dv; c++)
            {
                float acc = 0f;
                for (int j = 0; j < dk; j++)
                    acc += qr[j] * state[j * dv + c];
                orow[c] = accumulate ? orow[c] + acc : acc;
            }
        }
    }

    private static void Require(Tensor tensor, string name, long elements)
    {
        if (tensor.ElementCount != elements)
            throw new ArgumentException($"FrameDeltaScan {name} {tensor.Shape} must hold {elements} elements.", name);
    }

    private static void RequireState(Tensor? state, string name, int heads, int keyDim, int valueDim)
    {
        if (state is null)
            return;
        if (state.DType != DType.F32)
            throw new ArgumentException($"FrameDeltaScan {name} must be F32, got {state.DType.Name}.", name);
        Require(state, name, (long)heads * keyDim * valueDim);
    }

    private readonly record struct Pointers(nint Output, nint Query, nint Key, nint Value, nint Beta, nint Decay, nint Initial, nint Final);

    private readonly record struct Shape(int Frames, int Spatial, int Heads, int KeyDim, int ValueDim);
}
