using Xunit;
using SharpInference.Conditioning;
using SharpInference.Interactive.ActionEncoders;

namespace SharpInference.Interactive.Tests;

/// <summary>Round-trip and layout tests for the 25-dim VPT action encoder (camera floats at slots 15/16).</summary>
public class OasisActionEncoderTests
{
    [Fact]
    public void BuildRow_PlacesCameraFloatsAtVptSlots()
    {
        byte[] keys = new byte[OasisActionEncoder.KeyCount];
        keys[0] = 1;      // inventory
        keys[11] = 1;     // forward
        keys[22] = 1;     // drop (last binary key)
        float[] row = new float[OasisActionEncoder.ActionDim];
        OasisActionEncoder.BuildRow(keys, cameraX: 0.5f, cameraY: -0.25f, row);

        Assert.Equal(1f, row[0]);
        Assert.Equal(1f, row[11]);
        Assert.Equal(0.5f, row[OasisActionEncoder.CameraXIndex]);
        Assert.Equal(-0.25f, row[OasisActionEncoder.CameraYIndex]);
        Assert.Equal(1f, row[24]);    // last binary key lands after the camera slots
        Assert.Equal(3.25f, row.Sum(), 3);   // 3 keys + 0.5 − 0.25 — nothing else set
    }

    [Fact]
    public void EncodeFromPayload_RoundTrips()
    {
        byte[] keys = new byte[OasisActionEncoder.KeyCount];
        keys[14] = 1;   // right
        byte[] payload = new byte[OasisActionEncoder.PayloadBytes];
        OasisActionEncoder.PackPayload(keys, cameraX: 2f, cameraY: -2f, payload);   // clamped to ±1

        using OasisActionEncoder encoder = new();
        Assert.Single(encoder.Streams);
        Assert.Equal(ActionStreamRole.TimestepAddon, encoder.Streams[0].Role);

        float[] row = new float[OasisActionEncoder.ActionDim];
        encoder.Encode(new ActionInput(payload, 0, 0), "vpt", row);
        Assert.Equal(1f, row[14]);
        Assert.Equal(1f, row[OasisActionEncoder.CameraXIndex]);
        Assert.Equal(-1f, row[OasisActionEncoder.CameraYIndex]);
    }
}
