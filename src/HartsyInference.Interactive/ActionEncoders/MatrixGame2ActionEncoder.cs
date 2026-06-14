using System.Buffers.Binary;
using HartsyInference.Conditioning;

namespace HartsyInference.Interactive.ActionEncoders;

/// <summary>Matrix-Game 2.0 action encoder — per-domain variants share one layout: <c>[K key bytes][float pitchΔ]
/// [float yawΔ]</c> (mouse floats only when the variant has a mouse). Keyboard index maps per
/// <c>utils/conditions.py</c>: universal = {forward, back, left, right}; GTA = {forward, back}; TempleRun =
/// {nomove, jump, slide, turnleft, turnright, leftside, rightside} (keyboard-only). Mouse deltas are continuous,
/// trained around ±0.1 (<c>CAM_VALUE</c>). See <c>docs/Research/MATRIX_GAME_2_ARCHITECTURE.md</c> § 3/Data Layouts.</summary>
public sealed class MatrixGame2ActionEncoder : IActionEncoder
{
    /// <summary>The trained mouse-delta magnitude per discrete camera keystroke.</summary>
    public const float CamValue = 0.1f;

    private readonly int _keyboardDim;
    private readonly bool _hasMouse;
    private readonly float _maxMouseMagnitude;
    private readonly ActionStreamSpec[] _streams;

    public MatrixGame2ActionEncoder(int keyboardDim, bool hasMouse, float maxMouseMagnitude = 1.0f)
    {
        if (keyboardDim <= 0) throw new ArgumentOutOfRangeException(nameof(keyboardDim));
        _keyboardDim = keyboardDim;
        _hasMouse = hasMouse;
        _maxMouseMagnitude = maxMouseMagnitude;
        _streams = hasMouse
            ?
            [
                new ActionStreamSpec("keyboard", ActionStreamRole.PerBlockCrossAttn, keyboardDim),
                new ActionStreamSpec("mouse", ActionStreamRole.PerBlockSelfAttn, 2),
            ]
            : [new ActionStreamSpec("keyboard", ActionStreamRole.PerBlockCrossAttn, keyboardDim)];
    }

    /// <summary>Universal variant: 4-key WASD + mouse.</summary>
    public static MatrixGame2ActionEncoder CreateUniversal() => new(keyboardDim: 4, hasMouse: true);

    /// <summary>GTA-driving variant: forward/back + mouse steering.</summary>
    public static MatrixGame2ActionEncoder CreateGtaDrive() => new(keyboardDim: 2, hasMouse: true);

    /// <summary>TempleRun variant: 7 discrete moves, no mouse.</summary>
    public static MatrixGame2ActionEncoder CreateTempleRun() => new(keyboardDim: 7, hasMouse: false);

    /// <summary>Keyboard one-hot width for this variant.</summary>
    public int KeyboardDim => _keyboardDim;

    /// <summary>Expected payload size in bytes.</summary>
    public int PayloadBytes => _keyboardDim + (_hasMouse ? 2 * sizeof(float) : 0);

    public IReadOnlyList<ActionStreamSpec> Streams => _streams;

    /// <summary>Packs raw inputs into this variant's payload.</summary>
    public void PackPayload(ReadOnlySpan<byte> keyStates, float pitchDelta, float yawDelta, Span<byte> payload)
    {
        if (keyStates.Length != _keyboardDim)
            throw new ArgumentException($"Expected {_keyboardDim} key states, got {keyStates.Length}.", nameof(keyStates));
        if (payload.Length < PayloadBytes)
            throw new ArgumentException($"payload must hold {PayloadBytes} bytes.", nameof(payload));
        keyStates.CopyTo(payload);
        if (_hasMouse)
        {
            BinaryPrimitives.WriteSingleLittleEndian(payload[_keyboardDim..], pitchDelta);
            BinaryPrimitives.WriteSingleLittleEndian(payload[(_keyboardDim + sizeof(float))..], yawDelta);
        }
    }

    public void Encode(in ActionInput action, string streamName, Span<float> destination)
    {
        ReadOnlySpan<byte> payload = action.Payload.Span;
        if (payload.Length < PayloadBytes)
            throw new ArgumentException($"Matrix-Game 2 payload must be {PayloadBytes} bytes; got {payload.Length}.", nameof(action));
        switch (streamName)
        {
            case "keyboard":
                KeyboardOneHotEncoder.Encode(payload[.._keyboardDim], destination);
                break;
            case "mouse" when _hasMouse:
                float pitch = BinaryPrimitives.ReadSingleLittleEndian(payload[_keyboardDim..]);
                float yaw = BinaryPrimitives.ReadSingleLittleEndian(payload[(_keyboardDim + sizeof(float))..]);
                MouseDeltaEncoder.Encode(pitch, yaw, _maxMouseMagnitude, destination);
                break;
            default:
                throw new ArgumentException($"Unknown stream '{streamName}' for this variant.", nameof(streamName));
        }
    }

    public void Dispose()
    {
    }
}
