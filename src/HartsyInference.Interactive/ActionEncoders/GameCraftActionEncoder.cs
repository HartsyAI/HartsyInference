using System.Buffers.Binary;
using HartsyInference.Conditioning;
using HartsyInference.Interactive.Camera;

namespace HartsyInference.Interactive.ActionEncoders;

/// <summary>Hunyuan-GameCraft action encoder. Per-chunk action (one symbol per 33-frame chunk): WASD keys +
/// a translation speed ∈ [0,3] + optional mouse yaw/pitch. Produces the <see cref="ActionStreamRole.PluckerMap"/>
/// stream: per-frame 6-channel Plücker ray maps from the cumulative camera pose (translation
/// <c>= speed·dir·(f/fps)</c>, rotation accumulated from mouse). Reuses <see cref="Se3Math"/> for the pose and
/// <see cref="PluckerEmbedding"/> for the ray map — the maps feed <c>GameCraftCameraNet</c>.
/// <para>Payload (16 bytes): <c>[w][a][s][d][float speed][float yawDelta][float pitchDelta]</c>. Cumulative pose
/// uses the action's <see cref="ActionInput.FrameIndex"/> as the chunk-relative frame. <b>Validation-gated:</b>
/// FOV/intrinsics, axis convention, and speed scaling are reconciled against the reference.</para></summary>
public sealed class GameCraftActionEncoder : IActionEncoder
{
    /// <summary>Payload size: 4 key bytes + speed + yawDelta + pitchDelta.</summary>
    public const int PayloadBytes = 4 + 3 * sizeof(float);

    private readonly float _fx, _fy, _cx, _cy;
    private readonly int _height, _width, _fps;
    private readonly float _rotScaleDeg;
    private readonly ActionStreamSpec[] _streams;

    public GameCraftActionEncoder(float fx, float fy, float cx, float cy, int height, int width, int fps, float rotScaleDeg = 1.0f)
    {
        _fx = fx; _fy = fy; _cx = cx; _cy = cy; _height = height; _width = width; _fps = Math.Max(1, fps);
        _rotScaleDeg = rotScaleDeg;
        _streams = [new("plucker", ActionStreamRole.PluckerMap, height * width * PluckerEmbedding.Channels)];
    }

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

        float dirX = (p[3] != 0 ? 1f : 0f) - (p[1] != 0 ? 1f : 0f);   // d (+x) − a (−x)
        float dirZ = (p[0] != 0 ? 1f : 0f) - (p[2] != 0 ? 1f : 0f);   // w (+z) − s (−z)
        float speed = BinaryPrimitives.ReadSingleLittleEndian(p[4..]);
        float yawDelta = BinaryPrimitives.ReadSingleLittleEndian(p[8..]);
        float pitchDelta = BinaryPrimitives.ReadSingleLittleEndian(p[12..]);

        float f = action.FrameIndex; // chunk-relative frame
        float tNorm = f / _fps;
        float x = dirX * speed * tNorm, z = dirZ * speed * tNorm, y = 0f;
        float yaw = yawDelta * _rotScaleDeg * f;
        float pitch = Math.Clamp(pitchDelta * _rotScaleDeg * f, -89f, 89f);

        Span<float> pose = stackalloc float[16];
        Se3Math.GetExtrinsics(0f, pitch, yaw, x, y, z, pose);
        PluckerEmbedding.Compute(pose, _fx, _fy, _cx, _cy, _height, _width, destination);
    }

    public void Dispose() { }
}
