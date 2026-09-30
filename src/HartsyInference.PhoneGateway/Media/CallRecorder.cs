using HartsyInference.Core.Logging;

namespace HartsyInference.PhoneGateway.Media;

/// <summary>Writes one call's raw PCM to two files: the caller at 16 kHz (from the pump thread) and what the gateway
/// sent at 8 kHz (from the outbound producer). Each side is written by exactly one thread; the file streams carry
/// their own buffers, so a write is a memcpy until a buffer fills.</summary>
public sealed class CallRecorder : IDisposable
{
    private const int BufferBytes = 64 * 1024;

    private readonly FileStream _inbound;
    private readonly FileStream _outbound;
    private int _disposed;

    private CallRecorder(FileStream inbound, FileStream outbound)
    {
        _inbound = inbound;
        _outbound = outbound;
    }

    /// <summary>Opens the pair of files for <paramref name="sipCallId"/> under <paramref name="directory"/>, or returns
    /// null (after logging) when the directory cannot be used, so a recording problem never fails a call.</summary>
    public static CallRecorder? TryOpen(string directory, string sipCallId)
    {
        try
        {
            Directory.CreateDirectory(directory);
            string stamp = DateTime.UtcNow.ToString("yyyyMMdd'T'HHmmss'Z'", System.Globalization.CultureInfo.InvariantCulture);
            string safeId = string.Concat(sipCallId.Select(c => char.IsLetterOrDigit(c) ? c : '_'));
            string prefix = Path.Combine(directory, $"{stamp}_{safeId}");
            FileStream inbound = new(prefix + "_in16k.pcm", FileMode.CreateNew, FileAccess.Write, FileShare.Read, BufferBytes);
            FileStream outbound = new(prefix + "_out8k.pcm", FileMode.CreateNew, FileAccess.Write, FileShare.Read, BufferBytes);
            return new CallRecorder(inbound, outbound);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            Logs.Error($"[PhoneGateway] Recording disabled for call {sipCallId}: cannot write under {directory}", ex);
            return null;
        }
    }

    public void WriteInbound(ReadOnlySpan<short> pcm16k) => Write(_inbound, pcm16k);

    public void WriteOutbound(ReadOnlySpan<short> pcm8k) => Write(_outbound, pcm8k);

    private void Write(FileStream stream, ReadOnlySpan<short> pcm)
    {
        if (Volatile.Read(ref _disposed) != 0)
        {
            return;
        }
        try
        {
            stream.Write(MemoryMarshal.AsBytes(pcm));
        }
        catch (Exception ex) when (ex is IOException or ObjectDisposedException)
        {
            Logs.Error("[PhoneGateway] Recording write failed; recording stops for this call", ex);
            Volatile.Write(ref _disposed, 1);
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }
        _inbound.Dispose();
        _outbound.Dispose();
    }
}
