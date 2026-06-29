using System.Buffers.Binary;
using HartsyInference.Conditioning;
using HartsyInference.Interactive.Camera;

namespace HartsyInference.Interactive.ActionEncoders;

/// <summary>Hunyuan-GameCraft action encoder. Per-chunk action (one symbol per <see cref="FramesPerChunk"/>-frame
/// chunk): WASD keys + a translation magnitude + optional mouse yaw/pitch. Produces the
/// <see cref="ActionStreamRole.PluckerMap"/> stream: per-frame 6-channel Plücker ray maps (moment + direction, the
/// GameCraft <c>ray_condition</c> layout) from the cumulative camera pose. Translation follows upstream
/// <c>ActionToPoseFromID</c>: the total move <c>value·dir</c> is spread linearly over the chunk, so frame <c>f</c> sits
/// at <c>value·dir·(f/duration)</c> along the camera's local forward/right axes; rotation accumulates the same way.
/// Reuses <see cref="Se3Math"/> for the extrinsic and <see cref="PluckerEmbedding"/> for the ray map — the maps feed
/// <c>GameCraftCameraNet</c>.
/// <para>Payload (16 bytes): <c>[w][a][s][d][float speed][float yawDelta][float pitchDelta]</c> where <c>speed</c> is
/// the upstream <c>value</c> (reference default 0.2). Cumulative pose uses the action's
/// <see cref="ActionInput.FrameIndex"/> as the chunk-relative frame. <b>Validation-gated:</b> the exact axis
/// convention is reconciled against the reference on first real-weight run.</para></summary>
public sealed class GameCraftActionEncoder : IActionEncoder
{
    /// <summary>Payload size: 4 key bytes + speed + yawDelta + pitchDelta.</summary>
    public const int PayloadBytes = 4 + 3 * sizeof(float);

    /// <summary>Upstream default per-chunk frame count (<c>duration</c> in <c>ActionToPoseFromID</c>).</summary>
    public const int FramesPerChunk = 33;

    private readonly float _fx, _fy, _cx, _cy;
    private readonly int _height, _width, _duration;
    private readonly float _rotScaleDeg;
    private readonly ActionStreamSpec[] _streams;

    /// <param name="framesPerChunk">Frames per action chunk (upstream <c>duration</c>, default 33). Translation and
    /// rotation are spread linearly across this many frames.</param>
    public GameCraftActionEncoder(float fx, float fy, float cx, float cy, int height, int width,
        int framesPerChunk = FramesPerChunk, float rotScaleDeg = 1.0f)
    {
        _fx = fx; _fy = fy; _cx = cx; _cy = cy; _height = height; _width = width;
        _duration = Math.Max(1, framesPerChunk);
        _rotScaleDeg = rotScaleDeg;
        _streams = [new("plucker", ActionStreamRole.PluckerMap, height * width * PluckerEmbedding.Channels)];
    }

    /// <summary>Builds an encoder from the upstream normalized intrinsics <c>[fx, fy, cx, cy] =
    /// [0.50505, 0.8979, 0.5, 0.5]</c> (fractions of width/height), scaled to pixel units for the given resolution.
    /// These are the values <c>ActionToPoseFromID</c> uses by default.</summary>
    public static GameCraftActionEncoder CreateDefault(int height, int width,
        int framesPerChunk = FramesPerChunk, float rotScaleDeg = 1.0f) =>
        new(0.50505f * width, 0.8979f * height, 0.5f * width, 0.5f * height, height, width, framesPerChunk, rotScaleDeg);

    public IReadOnlyList<ActionStreamSpec> Streams => _streams;

    /// <summary>Packs raw inputs into the 16-byte payload (keys 0/1).</summary>
    public static void PackPayload(bool w, bool a, bool s, bool d, float speed, float yawDelta, float pitchDelta, Span<byte> payload)
    {
        if (payload.Length < PayloadBytes) throw new ArgumentException($"payload must hold {PayloadBytes} bytes.", nameof(payload));
        payload[0] = (byte)(w ? 1 : 0); payload[1] = (byte)(a ? 1 : 0); payload[2] = (byte)(s ? 1 : 0); payload[3] = (byte)(d ? 1 : 0);
        BinaryPrimitives.WriteSingleLittleEndian(payload[4..], speed);
        BinaryPrimitives.WriteSingleLittleEndian(payload[8..], yawDelta);
        BinaryPrimitives.WriteSingleLittleEndian(payload[12..], pitchDelta);
    }

    public void Encode(in ActionInput action, string streamName, Span<float> destination)
    {
        if (streamName != "plucker") throw new ArgumentException($"unknown stream '{streamName}'.", nameof(streamName));
        ReadOnlySpan<byte> p = action.Payload.Span;
        if (p.Length < PayloadBytes) throw new ArgumentException($"GameCraft payload must be {PayloadBytes} bytes; got {p.Length}.", nameof(action));
        if (destination.Length < _streams[0].FloatsPerFrame) throw new ArgumentException("destination too small for Plücker map.", nameof(destination));

        float forward = (p[0] != 0 ? 1f : 0f) - (p[2] != 0 ? 1f : 0f);   // w (forward) − s (backward)
        float strafe = (p[3] != 0 ? 1f : 0f) - (p[1] != 0 ? 1f : 0f);    // d (right) − a (left)
        float speed = BinaryPrimitives.ReadSingleLittleEndian(p[4..]);    // upstream "value"
        float yawDelta = BinaryPrimitives.ReadSingleLittleEndian(p[8..]);
        float pitchDelta = BinaryPrimitives.ReadSingleLittleEndian(p[12..]);

        float f = action.FrameIndex;          // chunk-relative frame
        float tNorm = f / _duration;          // upstream spreads the move over `duration` frames, not fps
        float yaw = yawDelta * _rotScaleDeg * f;
        float pitch = Math.Clamp(pitchDelta * _rotScaleDeg * f, -89f, 89f);

        // Camera-local forward/right axes (upstream ActionToPoseFromID convention), evaluated at the current heading.
        float yawRad = yaw * MathF.PI / 180f, pitchRad = pitch * MathF.PI / 180f;
        float fwdX = -MathF.Sin(yawRad) * MathF.Cos(pitchRad), fwdY = MathF.Sin(pitchRad), fwdZ = -MathF.Cos(yawRad) * MathF.Cos(pitchRad);
        float rightX = MathF.Cos(yawRad), rightY = 0f, rightZ = -MathF.Sin(yawRad);
        float mag = speed * tNorm;
        float x = (fwdX * forward + rightX * strafe) * mag;
        float y = (fwdY * forward + rightY * strafe) * mag;
        float z = (fwdZ * forward + rightZ * strafe) * mag;

        Span<float> pose = stackalloc float[16];
        Se3Math.GetExtrinsics(0f, pitch, yaw, x, y, z, pose);
        PluckerEmbedding.Compute(pose, _fx, _fy, _cx, _cy, _height, _width, destination, moment: true);
    }

    public void Dispose() { }
}
