using HartsyInference.Core.Backends;

namespace HartsyInference.Audio.Models.VoiceIsolator;

/// <summary>Renders clips of any length through a <see cref="VoiceIsolatorNetwork"/> in overlapping windows with a linear crossfade,
/// as the reference's <c>enhance</c> does: every window is a full one, the last ends at the clip's end and fades in where the
/// window before it fades out.</summary>
public static class VoiceIsolatorRenderer
{
    /// <summary>Window and crossfade in samples, the crossfade capped at half the window.</summary>
    public static (int Chunk, int Overlap) WindowSamples(double chunkSeconds, double overlapSeconds, int sampleRate, int minSamples)
    {
        if (!double.IsFinite(chunkSeconds) || !double.IsFinite(overlapSeconds))
        {
            throw new ArgumentException("chunkSeconds and overlapSeconds must be finite numbers of seconds.");
        }
        int chunk = (int)(chunkSeconds * sampleRate);
        if (chunk < minSamples)
        {
            throw new ArgumentException($"chunkSeconds={chunkSeconds} is shorter than {1000.0 * minSamples / sampleRate:g} ms.");
        }
        if (overlapSeconds < 0)
        {
            throw new ArgumentException($"overlapSeconds must not be negative, not {overlapSeconds}.");
        }
        return (chunk, Math.Min((int)(overlapSeconds * sampleRate), chunk / 2));
    }

    /// <summary>Isolates the voice of <paramref name="audio"/> (mono, at the network's rate). A clip no longer than one window is a
    /// single pass.</summary>
    public static float[] Render(VoiceIsolatorNetwork network, IBackend backend, float[] audio, double? chunkSeconds = null,
        double? overlapSeconds = null, Action<int, int>? progress = null, CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(network);
        ArgumentNullException.ThrowIfNull(audio);
        if (network.Config.Causal)
        {
            throw new ArgumentException("The offline renderer takes the two-sided checkpoint; stream a causal one with VoiceIsolatorStream.", nameof(network));
        }
        VoiceIsolatorConfig config = network.Config;
        (int chunk, int overlap) = WindowSamples(chunkSeconds ?? config.ChunkSeconds, overlapSeconds ?? config.OverlapSeconds,
            config.SampleRate, config.HopLength);
        int n = audio.Length;
        if (n <= chunk)
        {
            progress?.Invoke(1, 1);
            return network.Isolate(backend, audio, 0);
        }
        List<int> starts = WindowStarts(n, chunk, overlap);
        float[] output = new float[n];
        float[] weight = new float[n];
        float[] window = new float[chunk];
        for (int index = 0; index < starts.Count; index++)
        {
            cancel.ThrowIfCancellationRequested();
            int start = starts[index];
            FillWindow(window, starts, index, chunk, overlap, n);
            float[] rendered = network.Isolate(backend, audio.AsSpan(start, chunk).ToArray(), 0);
            for (int i = 0; i < chunk; i++)
            {
                output[start + i] += rendered[i] * window[i];
                weight[start + i] += window[i];
            }
            progress?.Invoke(index + 1, starts.Count);
        }
        for (int i = 0; i < n; i++)
        {
            output[i] /= MathF.Max(weight[i], 1e-6f);
        }
        return output;
    }

    /// <summary>First sample of every window of a clip of <paramref name="n"/> samples longer than one window.</summary>
    internal static List<int> WindowStarts(int n, int chunk, int overlap)
    {
        List<int> starts = [];
        for (int s = 0; s < n - chunk; s += chunk - overlap)
        {
            starts.Add(s);
        }
        starts.Add(n - chunk);
        return starts;
    }

    /// <summary>Fills window <paramref name="index"/>'s weights: zero where the window before still owns the clip, a linear fade-in
    /// over the overlap, one, then a fade-out unless the window ends the clip.</summary>
    internal static void FillWindow(float[] window, List<int> starts, int index, int chunk, int overlap, int n)
    {
        Array.Fill(window, 1f);
        if (index > 0)
        {
            int fadeIn = starts[index - 1] + chunk - overlap - starts[index];
            Array.Clear(window, 0, fadeIn);
            for (int i = 0; i < overlap; i++)
            {
                window[fadeIn + i] = Linspace(0f, 1f, overlap, i);
            }
        }
        if (overlap > 0 && starts[index] + chunk < n)
        {
            for (int i = 0; i < overlap; i++)
            {
                window[chunk - overlap + i] = Linspace(1f, 0f, overlap, i);
            }
        }
    }

    /// <summary>Element <paramref name="i"/> of <c>torch.linspace(start, end, steps)</c> in float32 (symmetric from both ends).</summary>
    private static float Linspace(float start, float end, int steps, int i)
    {
        if (steps == 1)
        {
            return start;
        }
        float step = (end - start) / (steps - 1);
        return i < steps / 2 ? start + step * i : end - step * (steps - 1 - i);
    }
}
