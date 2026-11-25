using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Host;

/// <summary>Host reference for <see cref="IBackend.SelectiveScan"/>: the Mamba-1 selective state-space recurrence in F32, one
/// task per (row, channel) with that channel's state held in registers. Frames run in order, so a carried state continues a
/// stream exactly where the previous call stopped.</summary>
public static class SelectiveScanReference
{
    /// <summary>States per channel the reference keeps on the stack.</summary>
    public const int MaxStates = 256;

    /// <summary>Validates the operands of <see cref="IBackend.SelectiveScan"/>.</summary>
    public static void Validate(Tensor output, Tensor u, Tensor delta, Tensor a, Tensor b, Tensor c, Tensor d, Tensor? z,
        Tensor? state, int rows, int frames, int channels, int states)
    {
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(u);
        ArgumentNullException.ThrowIfNull(delta);
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);
        ArgumentNullException.ThrowIfNull(c);
        ArgumentNullException.ThrowIfNull(d);
        if (states > MaxStates)
        {
            throw new ArgumentOutOfRangeException(nameof(states), states, $"SelectiveScan holds at most {MaxStates} states per channel; got {states}.");
        }
        if (rows < 1 || frames < 1 || channels < 1 || states < 1)
            throw new ArgumentException($"SelectiveScan needs positive dims, got rows={rows}, frames={frames}, channels={channels}, states={states}.");
        long perChannel = (long)rows * frames * channels;
        long perState = (long)rows * frames * states;
        Check(output, perChannel, nameof(output));
        Check(u, perChannel, nameof(u));
        Check(delta, perChannel, nameof(delta));
        Check(a, (long)channels * states, nameof(a));
        Check(b, perState, nameof(b));
        Check(c, perState, nameof(c));
        Check(d, channels, nameof(d));
        if (z is not null) Check(z, perChannel, nameof(z));
        if (state is not null) Check(state, (long)rows * channels * states, nameof(state));
    }

    /// <summary>Runs the scan. <paramref name="output"/> may alias <paramref name="u"/>; <paramref name="state"/> is read as the
    /// initial state and rewritten with the state after the last frame.</summary>
    public static unsafe void Apply(Tensor output, Tensor u, Tensor delta, Tensor a, Tensor b, Tensor c, Tensor d, Tensor? z,
        Tensor? state, int rows, int frames, int channels, int states)
    {
        Validate(output, u, delta, a, b, c, d, z, state, rows, frames, channels, states);
        float* uP = (float*)u.DataPointer;
        float* deltaP = (float*)delta.DataPointer;
        float* aP = (float*)a.DataPointer;
        float* bP = (float*)b.DataPointer;
        float* cP = (float*)c.DataPointer;
        float* dP = (float*)d.DataPointer;
        float* zP = z is null ? null : (float*)z.DataPointer;
        float* stateP = state is null ? null : (float*)state.DataPointer;
        float* outP = (float*)output.DataPointer;
        long work = (long)rows * channels * frames * states;
        CpuParallel.For(rows * channels, work, task =>
        {
            int row = task / channels;
            int channel = task - row * channels;
            float* h = stackalloc float[states];
            float* rate = aP + (long)channel * states;
            float* carried = stateP is null ? null : stateP + ((long)row * channels + channel) * states;
            for (int n = 0; n < states; n++) h[n] = carried is null ? 0f : carried[n];
            float skip = dP[channel];
            for (int t = 0; t < frames; t++)
            {
                long at = ((long)row * frames + t) * channels + channel;
                float* bt = bP + ((long)row * frames + t) * states;
                float* ct = cP + ((long)row * frames + t) * states;
                float step = deltaP[at];
                float x = uP[at];
                float drive = step * x;
                float acc = 0f;
                for (int n = 0; n < states; n++)
                {
                    h[n] = MathF.Exp(step * rate[n]) * h[n] + drive * bt[n];
                    acc += h[n] * ct[n];
                }
                acc += x * skip;
                if (zP is not null)
                {
                    float g = zP[at];
                    acc *= g / (1f + MathF.Exp(-g));
                }
                outP[at] = acc;
            }
            if (carried is not null)
            {
                for (int n = 0; n < states; n++) carried[n] = h[n];
            }
        });
    }

    private static void Check(Tensor t, long elements, string name)
    {
        if (t.DType != DType.F32)
            throw new ArgumentException($"SelectiveScan {name} must be F32, got {t.DType.Name}.", name);
        if (t.ElementCount != elements)
            throw new ArgumentException($"SelectiveScan {name} holds {t.ElementCount} elements, expected {elements}.", name);
    }
}
