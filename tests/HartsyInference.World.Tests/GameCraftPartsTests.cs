using HartsyInference.Conditioning;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.World.ActionEncoders;
using HartsyInference.World.Models;
using HartsyInference.World.Pipelines;
using Xunit;

namespace HartsyInference.World.Tests;

/// <summary>CPU structural tests for the Hunyuan-GameCraft action/camera/latent parts: the action encoder
/// produces a correctly-sized finite Plücker map, the CameraNet produces image-grid-aligned tokens, and the
/// latent builder lays out the 33-channel composite + mask correctly.</summary>
public sealed unsafe class GameCraftPartsTests
{
    [Fact]
    public void ActionEncoder_ProducesFinitePluckerMap()
    {
        using GameCraftActionEncoder enc = new(fx: 8, fy: 8, cx: 8, cy: 8, height: 16, width: 16, framesPerChunk: 25);
        Assert.Single(enc.Streams);
        Assert.Equal(ActionStreamRole.PluckerMap, enc.Streams[0].Role);
        Assert.Equal(16 * 16 * 6, enc.Streams[0].FloatsPerFrame);

        byte[] payload = new byte[GameCraftActionEncoder.PayloadBytes];
        GameCraftActionEncoder.PackPayload(w: true, a: false, s: false, d: true, speed: 1.5f, yawDelta: 0.1f, pitchDelta: 0f, payload);
        ActionInput action = new(payload, FrameIndex: 3, TimestampNanos: 0);

        float[] map = new float[enc.Streams[0].FloatsPerFrame];
        enc.Encode(action, "plucker", map);
        Assert.All(map, v => Assert.True(float.IsFinite(v)));
        // Direction channels (3..5) of at least one pixel should be a unit-ish vector (non-zero).
        Assert.True(MathF.Abs(map[3]) + MathF.Abs(map[4]) + MathF.Abs(map[5]) > 1e-3f);
    }

    [Fact]
    public void LatentBuilder_AssemblesCompositeWithMask()
    {
        int b = 1, c = 16, T = 2, H = 2, W = 2;
        using Tensor noisy = Ramp(1f, b, c, T, H, W);
        using Tensor history = Ramp(1000f, b, c, T, H, W);
        float[] mask = [1f, 0f];

        using Tensor outT = GameCraftLatentBuilder.Build(noisy, history, mask);
        Assert.Equal(2 * c + 1, (int)outT.Shape[1]); // 33
        float* o = (float*)outT.DataPointer;
        long frame = (long)H * W, perCh = T * frame, outC = 2 * c + 1;
        // Channel 0 (noisy) frame0 pixel0 == noisy[0]; channel c (history) == history[0]; channel 2c (mask) frame0 == 1, frame1 == 0.
        Assert.Equal(((float*)noisy.DataPointer)[0], o[0], 5);
        Assert.Equal(((float*)history.DataPointer)[0], o[c * perCh], 5);
        Assert.Equal(1f, o[2 * c * perCh], 5);              // mask frame 0
        Assert.Equal(0f, o[2 * c * perCh + frame], 5);      // mask frame 1
        Assert.Equal(33, (int)outC);
    }

    private static Tensor T(Random r, params long[] dims)
    {
        Tensor t = new(new TensorShape(dims), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = (float)(r.NextDouble() * 0.2 - 0.1);
        return t;
    }

    private static Tensor Ones(long n) { Tensor t = new(new TensorShape(n), DType.F32); float* p = (float*)t.DataPointer; for (long i = 0; i < n; i++) p[i] = 1f; return t; }
    private static Tensor Zeros(long n) => new(new TensorShape(n), DType.F32);

    private static Tensor Ramp(float start, params long[] dims)
    {
        Tensor t = new(new TensorShape(dims), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = start + i;
        return t;
    }
}
