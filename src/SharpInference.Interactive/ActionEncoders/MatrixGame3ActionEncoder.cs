using System.Buffers.Binary;
using SharpInference.Conditioning;

namespace SharpInference.Interactive.ActionEncoders;

/// <summary>Matrix-Game 3.0 action encoder. Payload layout (14 bytes per frame):
/// <c>[key W][key S][key A][key D][key 5][key 6][float Δx][float Δy]</c> — 6 key-state bytes (0/1) followed by two
/// little-endian floats with the mouse yaw/pitch deltas. Produces two streams at the RGB frame rate: <c>"keyboard"</c>
/// (6 floats, one-hot, cross-attended by the ActionModule) and <c>"mouse"</c> (2 floats, self-attended). The pipeline
/// groups frames into latent windows (4 frames per latent, window 3) — the encoder stays per-frame.
/// See <c>docs/Research/MATRIX_GAME_3_ARCHITECTURE.md</c> § 5.</summary>
public sealed class MatrixGame3ActionEncoder : IActionEncoder
{
    /// <summary>Keyboard one-hot width (W/S/A/D + 2 extra action slots).</summary>
    public const int KeyboardDim = 6;

    /// <summary>Mouse stream width (Δx, Δy).</summary>
    public const int MouseDim = 2;

    /// <summary>Expected payload size in bytes.</summary>
    public const int PayloadBytes = KeyboardDim + 2 * sizeof(float);

    private readonly float _maxMouseMagnitude;
    private static readonly ActionStreamSpec[] _streamSpecs =
    [
        new("keyboard", ActionStreamRole.PerBlockCrossAttn, KeyboardDim),
        new("mouse", ActionStreamRole.PerBlockSelfAttn, MouseDim),
    ];

    public MatrixGame3ActionEncoder(float maxMouseMagnitude = 1.0f) => _maxMouseMagnitude = maxMouseMagnitude;

    public IReadOnlyList<ActionStreamSpec> Streams => _streamSpecs;

    /// <summary>Packs raw inputs into the 14-byte payload this encoder consumes.</summary>
    public static void PackPayload(ReadOnlySpan<byte> keyStates, float mouseDx, float mouseDy, Span<byte> payload)
    {
        if (keyStates.Length != KeyboardDim)
            throw new ArgumentException($"Expected {KeyboardDim} key states, got {keyStates.Length}.", nameof(keyStates));
        if (payload.Length < PayloadBytes)
            throw new ArgumentException($"payload must hold {PayloadBytes} bytes.", nameof(payload));
        keyStates.CopyTo(payload);
        BinaryPrimitives.WriteSingleLittleEndian(payload[KeyboardDim..], mouseDx);
        BinaryPrimitives.WriteSingleLittleEndian(payload[(KeyboardDim + sizeof(float))..], mouseDy);
    }

    public void Encode(in ActionInput action, string streamName, Span<float> destination)
    {
        ReadOnlySpan<byte> payload = action.Payload.Span;
        if (payload.Length < PayloadBytes)
            throw new ArgumentException($"Matrix-Game 3 payload must be {PayloadBytes} bytes; got {payload.Length}.", nameof(action));
        switch (streamName)
        {
            case "keyboard":
                KeyboardOneHotEncoder.Encode(payload[..KeyboardDim], destination);
                break;
            case "mouse":
                float dx = BinaryPrimitives.ReadSingleLittleEndian(payload[KeyboardDim..]);
                float dy = BinaryPrimitives.ReadSingleLittleEndian(payload[(KeyboardDim + sizeof(float))..]);
                MouseDeltaEncoder.Encode(dx, dy, _maxMouseMagnitude, destination);
                break;
            default:
                throw new ArgumentException($"Unknown stream '{streamName}' (expected 'keyboard' or 'mouse').", nameof(streamName));
        }
    }

    public void Dispose()
    {
    }
}
