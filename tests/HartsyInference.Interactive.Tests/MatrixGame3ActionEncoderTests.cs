using Xunit;
using HartsyInference.Conditioning;
using HartsyInference.Interactive.ActionEncoders;

namespace HartsyInference.Interactive.Tests;

/// <summary>Payload packing/encoding round-trip for the Matrix-Game 3.0 action encoder.</summary>
public class MatrixGame3ActionEncoderTests
{
    [Fact]
    public void PackAndEncode_RoundTripsKeyboardAndMouse()
    {
        byte[] payload = new byte[MatrixGame3ActionEncoder.PayloadBytes];
        MatrixGame3ActionEncoder.PackPayload([1, 0, 0, 1, 0, 0], mouseDx: 0.05f, mouseDy: -0.08f, payload);
        ActionInput action = new(payload, FrameIndex: 3, TimestampNanos: 0);

        using MatrixGame3ActionEncoder encoder = new(maxMouseMagnitude: 0.1f);
        Assert.Equal(2, encoder.Streams.Count);
        Assert.Equal(ActionStreamRole.PerBlockCrossAttn, encoder.Streams[0].Role);
        Assert.Equal(ActionStreamRole.PerBlockSelfAttn, encoder.Streams[1].Role);

        Span<float> kbd = stackalloc float[6];
        encoder.Encode(action, "keyboard", kbd);
        Assert.Equal(new float[] { 1, 0, 0, 1, 0, 0 }, kbd.ToArray());

        Span<float> mouse = stackalloc float[2];
        encoder.Encode(action, "mouse", mouse);
        Assert.Equal(0.05f, mouse[0], 5);
        Assert.Equal(-0.08f, mouse[1], 5);
    }

    [Fact]
    public void Encode_ClampsMouseAndRejectsUnknownStream()
    {
        byte[] payload = new byte[MatrixGame3ActionEncoder.PayloadBytes];
        MatrixGame3ActionEncoder.PackPayload([0, 0, 0, 0, 0, 0], mouseDx: 5f, mouseDy: -5f, payload);
        ActionInput action = new(payload, 0, 0);

        using MatrixGame3ActionEncoder encoder = new(maxMouseMagnitude: 0.1f);
        float[] mouse = new float[2];
        encoder.Encode(action, "mouse", mouse);
        Assert.Equal(0.1f, mouse[0], 5);
        Assert.Equal(-0.1f, mouse[1], 5);

        Assert.Throws<ArgumentException>(() => encoder.Encode(action, "gamepad", new float[2]));
    }
}
