using HartsyInference.Audio.Io;
using HartsyInference.PhoneGateway.Media;
using Xunit;

namespace HartsyInference.PhoneGateway.Tests;

/// <summary>The jitter buffer decides what the host hears when the network misbehaves, and every one of these
/// failures would be silent in a call: a reorder played out of order, a loss played as a click, a late packet
/// played after its slot, a duplicate played twice, or a sequence wrap that stalls playout.</summary>
public sealed class RtpJitterBufferTests
{
    private const byte Pcmu = 0;
    private const byte Pcma = 8;

    [Fact]
    public void PreBuffers_TargetDepthBeforeFirstPop()
    {
        RtpJitterBuffer buffer = new(targetDepth: 3);
        byte[] frame = new byte[RtpJitterBuffer.FrameBytes];
        Assert.Equal(JitterPopResult.Silence, buffer.Pop(frame, out _));
        Push(buffer, 100, 0);
        Push(buffer, 101, 0);
        Assert.Equal(JitterPopResult.Silence, buffer.Pop(frame, out _));
        Push(buffer, 102, 0);
        Assert.Equal(3, buffer.Depth);
        Assert.Equal(JitterPopResult.Frame, buffer.Pop(frame, out byte pt));
        Assert.Equal(Pcmu, pt);
        Assert.Equal(100, frame[0]);
        Assert.Equal(2, buffer.Depth);
    }

    [Fact]
    public void Reorder_PlaysInSequenceOrder()
    {
        RtpJitterBuffer buffer = new();
        Push(buffer, 10, 0);
        Push(buffer, 12, 0);
        Push(buffer, 11, 0);
        Assert.Equal(1, buffer.Reordered);
        byte[] frame = new byte[RtpJitterBuffer.FrameBytes];
        Assert.Equal(JitterPopResult.Frame, buffer.Pop(frame, out _));
        Assert.Equal(10, frame[0]);
        Assert.Equal(JitterPopResult.Frame, buffer.Pop(frame, out _));
        Assert.Equal(11, frame[0]);
        Assert.Equal(JitterPopResult.Frame, buffer.Pop(frame, out _));
        Assert.Equal(12, frame[0]);
        Assert.Equal(0, buffer.Lost);
    }

    [Fact]
    public void Loss_RepeatsOnceThenSilence()
    {
        RtpJitterBuffer buffer = new();
        Push(buffer, 1, 0, Pcmu, fill: 0x11);
        Push(buffer, 2, 0, Pcmu, fill: 0x22);
        Push(buffer, 3, 0, Pcmu, fill: 0x33);
        byte[] frame = new byte[RtpJitterBuffer.FrameBytes];
        Assert.Equal(JitterPopResult.Frame, buffer.Pop(frame, out _));
        Assert.Equal(JitterPopResult.Frame, buffer.Pop(frame, out _));
        Assert.Equal(JitterPopResult.Frame, buffer.Pop(frame, out _));
        Assert.Equal(0x33, frame[0]);
        // Frames 4 and 5 never arrive.
        Assert.Equal(JitterPopResult.Concealed, buffer.Pop(frame, out byte pt));
        Assert.Equal(0x33, frame[0]);
        Assert.Equal(Pcmu, pt);
        Assert.Equal(JitterPopResult.Concealed, buffer.Pop(frame, out _));
        Assert.Equal(G711.MuLawSilence, frame[0]);
        Assert.Equal(2, buffer.Lost);
        Assert.Equal(2, buffer.Concealed);
        // Frame 6 arrives on time and plays.
        Push(buffer, 6, 0, Pcmu, fill: 0x66);
        Assert.Equal(JitterPopResult.Frame, buffer.Pop(frame, out _));
        Assert.Equal(0x66, frame[0]);
    }

    [Fact]
    public void ALawLoss_ConcealsWithALawSilence()
    {
        RtpJitterBuffer buffer = new();
        for (ushort s = 1; s <= 3; s++)
        {
            Push(buffer, s, 0, Pcma, fill: 0x40);
        }
        byte[] frame = new byte[RtpJitterBuffer.FrameBytes];
        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(JitterPopResult.Frame, buffer.Pop(frame, out _));
        }
        Assert.Equal(JitterPopResult.Concealed, buffer.Pop(frame, out _));
        Assert.Equal(JitterPopResult.Concealed, buffer.Pop(frame, out byte pt));
        Assert.Equal(Pcma, pt);
        Assert.Equal(G711.ALawSilence, frame[0]);
    }

    [Fact]
    public void LatePacket_IsDroppedAndCounted()
    {
        RtpJitterBuffer buffer = new();
        Push(buffer, 1, 0);
        Push(buffer, 2, 0);
        Push(buffer, 3, 0);
        byte[] frame = new byte[RtpJitterBuffer.FrameBytes];
        buffer.Pop(frame, out _);
        buffer.Pop(frame, out _);
        Push(buffer, 1, 0);
        Assert.Equal(1, buffer.Late);
        Assert.Equal(1, buffer.Depth);
    }

    [Fact]
    public void Duplicate_IsCountedAndPlayedOnce()
    {
        RtpJitterBuffer buffer = new();
        Push(buffer, 1, 0);
        Push(buffer, 2, 0);
        Push(buffer, 2, 0);
        Push(buffer, 3, 0);
        Assert.Equal(1, buffer.Duplicate);
        Assert.Equal(3, buffer.Depth);
        byte[] frame = new byte[RtpJitterBuffer.FrameBytes];
        Assert.Equal(JitterPopResult.Frame, buffer.Pop(frame, out _));
        Assert.Equal(JitterPopResult.Frame, buffer.Pop(frame, out _));
        Assert.Equal(2, frame[0]);
        Assert.Equal(JitterPopResult.Frame, buffer.Pop(frame, out _));
        Assert.Equal(3, frame[0]);
    }

    [Fact]
    public void SequenceWrap_ContinuesAcross65535()
    {
        RtpJitterBuffer buffer = new();
        uint ts = 0;
        Push(buffer, 65534, ts);
        Push(buffer, 65535, ts += 160);
        Push(buffer, 0, ts += 160);
        Push(buffer, 1, ts += 160);
        byte[] frame = new byte[RtpJitterBuffer.FrameBytes];
        Assert.Equal(JitterPopResult.Frame, buffer.Pop(frame, out _));
        Assert.Equal(65534 & 0xFF, frame[0]);
        Assert.Equal(JitterPopResult.Frame, buffer.Pop(frame, out _));
        Assert.Equal(65535 & 0xFF, frame[0]);
        Assert.Equal(JitterPopResult.Frame, buffer.Pop(frame, out _));
        Assert.Equal(0, frame[0]);
        Assert.Equal(JitterPopResult.Frame, buffer.Pop(frame, out _));
        Assert.Equal(1, frame[0]);
        Assert.Equal(0, buffer.Resets);
        Assert.Equal(0, buffer.Lost);
    }

    [Fact]
    public void MarkerWithTimestampJump_ResetsToTheNewTalkspurt()
    {
        RtpJitterBuffer buffer = new();
        Push(buffer, 1, 0);
        Push(buffer, 2, 160);
        Push(buffer, 3, 320);
        byte[] frame = new byte[RtpJitterBuffer.FrameBytes];
        Assert.Equal(JitterPopResult.Frame, buffer.Pop(frame, out _));
        // A new talkspurt after silence suppression: marker set, timestamp far ahead.
        buffer.Push(4, 80000, marker: true, Pcmu, Fill(4));
        Assert.Equal(1, buffer.Resets);
        Assert.Equal(1, buffer.Depth);
        Assert.Equal(JitterPopResult.Silence, buffer.Pop(frame, out _));
        Push(buffer, 5, 80160);
        Push(buffer, 6, 80320);
        Assert.Equal(JitterPopResult.Frame, buffer.Pop(frame, out _));
        Assert.Equal(4, frame[0]);
    }

    [Fact]
    public void MarkerWithContinuousTimestamp_DoesNotReset()
    {
        RtpJitterBuffer buffer = new();
        Push(buffer, 1, 0);
        Push(buffer, 2, 160);
        buffer.Push(3, 320, marker: true, Pcmu, Fill(3));
        Assert.Equal(0, buffer.Resets);
        Assert.Equal(3, buffer.Depth);
    }

    [Fact]
    public void TimestampJumpWithoutMarker_Resets()
    {
        RtpJitterBuffer buffer = new();
        Push(buffer, 1, 0);
        Push(buffer, 2, 160);
        Push(buffer, 3, 9_000_000);
        Assert.Equal(1, buffer.Resets);
        Assert.Equal(1, buffer.Depth);
    }

    [Fact]
    public void FarAheadSequence_ResetsInsteadOfOverflowing()
    {
        RtpJitterBuffer buffer = new();
        Push(buffer, 1, 0);
        Push(buffer, 2, 160);
        Push(buffer, 3, 320);
        Push(buffer, 40, 320 + 37 * 160);
        Assert.Equal(1, buffer.Resets);
        Assert.Equal(1, buffer.Depth);
        Assert.False(buffer.Depth > RtpJitterBuffer.Slots);
    }

    [Fact]
    public void LongSilence_GoesIdleThenPrebuffersAgain()
    {
        RtpJitterBuffer buffer = new();
        Push(buffer, 1, 0);
        Push(buffer, 2, 160);
        Push(buffer, 3, 320);
        byte[] frame = new byte[RtpJitterBuffer.FrameBytes];
        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(JitterPopResult.Frame, buffer.Pop(frame, out _));
        }
        for (int i = 0; i < RtpJitterBuffer.MaxMisses; i++)
        {
            Assert.Equal(JitterPopResult.Concealed, buffer.Pop(frame, out _));
        }
        Assert.Equal(JitterPopResult.Silence, buffer.Pop(frame, out _));
        Assert.False(buffer.IsActive);
        Assert.Equal(JitterPopResult.Silence, buffer.Pop(frame, out _));
        Push(buffer, 200, 100_000);
        Assert.True(buffer.IsActive);
        Assert.Equal(JitterPopResult.Silence, buffer.Pop(frame, out _));
        Push(buffer, 201, 100_160);
        Push(buffer, 202, 100_320);
        Assert.Equal(JitterPopResult.Frame, buffer.Pop(frame, out _));
        Assert.Equal(200, frame[0]);
    }

    [Fact]
    public void OverDeepQueue_TrimsTowardTarget()
    {
        RtpJitterBuffer buffer = new(targetDepth: 3);
        for (ushort s = 1; s <= 7; s++)
        {
            Push(buffer, s, (uint)(s * 160));
        }
        Assert.Equal(7, buffer.Depth);
        byte[] frame = new byte[RtpJitterBuffer.FrameBytes];
        Assert.Equal(JitterPopResult.Frame, buffer.Pop(frame, out _));
        Assert.Equal(1, frame[0]);
        Assert.True(buffer.Trimmed >= 1);
        Assert.True(buffer.Depth <= 5);
    }

    [Fact]
    public void WrongPayloadSize_IsRefused()
    {
        RtpJitterBuffer buffer = new();
        buffer.Push(1, 0, false, Pcmu, new byte[240]);
        Assert.Equal(1, buffer.BadPayload);
        Assert.Equal(0, buffer.Depth);
        Assert.False(buffer.IsActive);
    }

    private static void Push(RtpJitterBuffer buffer, ushort seq, uint ts, byte pt = Pcmu, byte? fill = null) =>
        buffer.Push(seq, ts, marker: false, pt, Fill(fill ?? (byte)seq));

    private static byte[] Fill(byte value)
    {
        byte[] frame = new byte[RtpJitterBuffer.FrameBytes];
        Array.Fill(frame, value);
        return frame;
    }
}
