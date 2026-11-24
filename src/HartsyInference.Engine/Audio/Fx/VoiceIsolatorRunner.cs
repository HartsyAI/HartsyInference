using HartsyInference.Audio.Models.VoiceIsolator;
using HartsyInference.Core.Backends;

namespace HartsyInference.Engine.Audio;

/// <summary>A loaded Inworld Voice Isolator checkpoint: the offline one renders in overlapped 8 s windows, the realtime one runs
/// as a causal stream over the clip. Both take mono audio at 48 kHz and return the isolated voice.</summary>
internal sealed class VoiceIsolatorRunner(VoiceIsolatorNetwork network) : IDisposable
{
    private const int StreamWindowFrames = 600;
    private const int StreamCrossfadeFrames = 200;
    private const double StreamPieceSeconds = 0.5;

    /// <summary>Sample rate the network runs at.</summary>
    internal int SampleRate => network.Config.SampleRate;

    /// <summary>True for the causal checkpoint.</summary>
    internal bool Realtime => network.Config.Causal;

    /// <summary>The look-ahead positions the checkpoint was trained at (frames of 10 ms); empty for the offline one.</summary>
    internal IReadOnlyList<int> LookaheadChoices => network.Config.LookaheadChoices;

    /// <summary>Isolates the voice of <paramref name="mono48k"/>; the background is the input minus the voice.</summary>
    internal (float[] Voice, float[] Background) Isolate(IBackend backend, float[] mono48k, int? lookaheadFrames = null,
        Action<int, int>? progress = null, CancellationToken cancel = default)
    {
        float[] voice = Realtime
            ? Stream(backend, mono48k, lookaheadFrames, cancel)
            : VoiceIsolatorRenderer.Render(network, backend, mono48k, progress: progress, cancel: cancel);
        float[] background = new float[mono48k.Length];
        for (int i = 0; i < background.Length; i++)
        {
            background[i] = mono48k[i] - voice[i];
        }
        return (voice, background);
    }

    private float[] Stream(IBackend backend, float[] audio, int? lookaheadFrames, CancellationToken cancel)
    {
        if (lookaheadFrames is { } frames)
        {
            network.SetLookahead(frames);
        }
        using VoiceIsolatorStream stream = new(network, backend, windowFrames: StreamWindowFrames, crossfadeFrames: StreamCrossfadeFrames);
        int piece = (int)(StreamPieceSeconds * network.Config.SampleRate);
        List<float> output = new(audio.Length);
        for (int i = 0; i < audio.Length; i += piece)
        {
            cancel.ThrowIfCancellationRequested();
            output.AddRange(stream.Push(audio.AsSpan(i, Math.Min(piece, audio.Length - i))));
        }
        output.AddRange(stream.Flush());
        return [.. output];
    }

    /// <inheritdoc/>
    public void Dispose() => network.Dispose();
}
