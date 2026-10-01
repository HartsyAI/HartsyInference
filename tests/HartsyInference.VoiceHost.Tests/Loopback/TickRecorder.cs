using HartsyInference.Audio.Io;
using HartsyInference.Core.Runtime;
using HartsyInference.PhoneGateway.Media;

namespace HartsyInference.VoiceHost.Tests.Loopback;

/// <summary>Records every frame the gateway's RTP tick thread hands to the media session (the packet it sends): when, and
/// the frame's peak level after G.711 decoding. Fixed arrays, written by the tick thread alone.</summary>
internal sealed class TickRecorder
{
    private const int Capacity = 1 << 16;

    private readonly long[] _ns = new long[Capacity];
    private readonly int[] _peak = new int[Capacity];
    private readonly short[] _pcm = new short[ClockedAudioSource.FrameSamples];
    private int _count;

    /// <summary>Starts recording <paramref name="source"/>'s frames; called before the tick thread starts.</summary>
    public void Attach(ClockedAudioSource source)
    {
        source.OnAudioSourceEncodedSample += (_, encoded) =>
        {
            int index = Volatile.Read(ref _count);
            if (index >= Capacity)
            {
                return;
            }
            G711.Decode(encoded, _pcm, source.Law);
            int peak = 0;
            foreach (short sample in _pcm)
            {
                peak = Math.Max(peak, Math.Abs((int)sample));
            }
            _ns[index] = MonotonicClock.NowNs();
            _peak[index] = peak;
            Volatile.Write(ref _count, index + 1);
        };
    }

    /// <summary>Frames recorded so far, oldest first.</summary>
    public (long Ns, int Peak)[] Frames()
    {
        int count = Volatile.Read(ref _count);
        (long, int)[] frames = new (long, int)[count];
        for (int i = 0; i < count; i++)
        {
            frames[i] = (_ns[i], _peak[i]);
        }
        return frames;
    }

    /// <summary>Frames at or above <paramref name="peak"/> in <c>[fromNs, toNs]</c>.</summary>
    public (long Ns, int Peak)[] Audible(long fromNs, long toNs, int peak) =>
        [.. Frames().Where(f => f.Ns >= fromNs && f.Ns <= toNs && f.Peak >= peak)];
}
