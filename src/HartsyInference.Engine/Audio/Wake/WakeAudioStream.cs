using HartsyInference.Core.Logging;

namespace HartsyInference.Engine.Audio.Wake;

/// <summary>One spoken reply on its way to one satellite.
///
/// <para>A reply is a single thing to the listener and to the device — the playback ring is reset when one
/// starts and drained when one ends — but on the server it is produced in pieces, a sentence at a time, so that
/// the first can be heard while the rest is still being synthesized. This holds the numbering and the pacing
/// across those pieces so they arrive as the one reply they are. Sending each piece as its own reply instead
/// makes every sentence after the first reset the device's ring and cut off the one before it.</para>
///
/// <para>Pacing is deliberately not "as fast as the socket accepts". The device has a few hundred milliseconds
/// of jitter buffer; filling it faster than it drains costs frames, and filling it at exactly the speed of
/// speech leaves it one late packet from silence. So this runs a fixed lead ahead of playback and then
/// waits.</para></summary>
public sealed class WakeAudioStream : IAsyncDisposable
{
    private readonly WakeFrameCodec _codec;
    private readonly string _deviceId;
    private readonly int _sampleRate;
    private readonly int _leadMs;
    private readonly int _frameBytes;
    private readonly long _started = Environment.TickCount64;
    private long _sent;
    private int _seq;
    private bool _closed;
    private byte[]? _carry;
    private int _carryLength;

    /// <summary>Prefer <see cref="WakeService.BeginAudio"/>, which finds the device's codec for you. This is
    /// public so a host holding a codec of its own — and a test — can drive one directly.</summary>
    /// <param name="leadMs">How far ahead of playback to run. Zero sends as fast as the socket accepts.</param>
    public WakeAudioStream(WakeFrameCodec codec, string deviceId, int sampleRate, int leadMs)
    {
        _codec = codec;
        _deviceId = deviceId;
        _sampleRate = sampleRate;
        _leadMs = leadMs;
        // 40 ms a frame: two of the device's own 20 ms audio frames, and comfortably under the 1280-byte
        // payloads its receive buffer is sized for.
        _frameBytes = Math.Max(2, sampleRate / 25 * 2);
    }

    /// <summary>Bytes accepted by the device so far.</summary>
    public int BytesSent => (int)_sent;

    /// <summary>Whether the final frame has gone out.</summary>
    public bool IsComplete => _closed;

    /// <summary>Sends one piece of the reply, paced to playback.
    ///
    /// <para>A piece that does not divide evenly into frames leaves its tail here rather than going out short:
    /// a short frame is legal on the wire but would put an odd number of bytes into the device's ring, and the
    /// sample after it would be assembled from one byte of each of two different samples.</para></summary>
    public async Task WriteAsync(ReadOnlyMemory<byte> pcm, CancellationToken cancel = default)
    {
        if (_closed)
        {
            throw new InvalidOperationException("This reply has already been completed.");
        }
        if (pcm.Length == 0)
        {
            return;
        }
        if (_carryLength > 0)
        {
            int need = Math.Min(_frameBytes - _carryLength, pcm.Length);
            pcm.Slice(0, need).CopyTo(_carry.AsMemory(_carryLength));
            _carryLength += need;
            pcm = pcm[need..];
            if (_carryLength < _frameBytes)
            {
                return;
            }
            if (!await SendFrameAsync(_carry.AsMemory(0, _frameBytes), final: false, cancel).ConfigureAwait(false))
            {
                return;
            }
            _carryLength = 0;
        }
        int offset = 0;
        for (; offset + _frameBytes <= pcm.Length; offset += _frameBytes)
        {
            if (!await SendFrameAsync(pcm.Slice(offset, _frameBytes), final: false, cancel).ConfigureAwait(false))
            {
                return;
            }
        }
        int left = pcm.Length - offset;
        if (left > 0)
        {
            _carry ??= new byte[_frameBytes];
            pcm.Slice(offset, left).CopyTo(_carry);
            _carryLength = left;
        }
    }

    /// <summary>Flushes anything held back and marks the reply finished, so the device can drain and stop.</summary>
    public async Task CompleteAsync(CancellationToken cancel = default)
    {
        if (_closed)
        {
            return;
        }
        _closed = true;
        // The final frame always goes out, even with nothing left to say: it is what tells the device the reply
        // is over. A device that never sees it holds the turn open until its own watchdog gives up.
        ReadOnlyMemory<byte> tail = _carryLength > 0 ? _carry.AsMemory(0, _carryLength) : ReadOnlyMemory<byte>.Empty;
        _carryLength = 0;
        await SendFrameAsync(tail, final: true, cancel).ConfigureAwait(false);
    }

    /// <returns>False when the device hung up and there is no point continuing.</returns>
    private async Task<bool> SendFrameAsync(ReadOnlyMemory<byte> payload, bool final, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        string data = $"{{\"seq\":{_seq},\"final\":{(final ? "true" : "false")}}}";
        try
        {
            await _codec.WriteAsync("audio", data, payload, cancel).ConfigureAwait(false);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            // The device hung up mid-reply. Everything before this point was already played.
            Logs.Verbose($"[Audio][Wake] Audio to '{_deviceId}' stopped after {_sent} bytes: {ex.Message}");
            _closed = true;
            return false;
        }
        _seq++;
        _sent += payload.Length;

        // Stay a little ahead of playback and no further.
        double playedMs = _sent / 2.0 / _sampleRate * 1000.0;
        int ahead = (int)(playedMs - (Environment.TickCount64 - _started)) - _leadMs;
        if (ahead > 0)
        {
            await Task.Delay(ahead, cancel).ConfigureAwait(false);
        }
        return true;
    }

    /// <summary>Completes the reply if the caller did not — including on the way out of a cancelled turn, where
    /// the device would otherwise be left holding a stream that never ends.</summary>
    public async ValueTask DisposeAsync()
    {
        if (_closed)
        {
            return;
        }
        try
        {
            await CompleteAsync(CancellationToken.None).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            Logs.Verbose($"[Audio][Wake] Could not close the reply to '{_deviceId}': {ex.Message}");
        }
    }
}
