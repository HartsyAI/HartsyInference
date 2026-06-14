using System.Buffers.Binary;
using HartsyInference.Conditioning;

namespace HartsyInference.Interactive.ActionEncoders;

/// <summary>Oasis-500m action encoder — the 25-dim Minecraft VPT action vector (<c>open-oasis/utils.py
/// ACTION_KEYS</c>): 23 binary keys (inventory, ESC, hotbar 1-9, WASD, jump/sneak/sprint/swapHands,
/// attack/use/pickItem/drop) and 2 normalized camera floats in [−1, 1] at slots 15/16. One stream, added to the
/// per-frame timestep embedding (<see cref="ActionStreamRole.TimestepAddon"/>). Payload layout (31 bytes):
/// <c>[23 key bytes][float cameraX][float cameraY]</c>. The VPT bucket math (<c>(raw − 40)/40</c>) only matters when
/// replaying VPT logs — the primary API takes normalized floats directly.</summary>
public sealed class OasisActionEncoder : IActionEncoder
{
    /// <summary>Action vector width.</summary>
    public const int ActionDim = 25;

    /// <summary>Binary key count (camera floats occupy vector slots 15/16, after the 4 movement keys).</summary>
    public const int KeyCount = 23;

    /// <summary>Index of cameraX in the 25-dim vector.</summary>
    public const int CameraXIndex = 15;

    /// <summary>Index of cameraY in the 25-dim vector.</summary>
    public const int CameraYIndex = 16;

    /// <summary>Expected payload size in bytes.</summary>
    public const int PayloadBytes = KeyCount + 2 * sizeof(float);

    private static readonly ActionStreamSpec[] _streamSpecs =
    [
        new("vpt", ActionStreamRole.TimestepAddon, ActionDim),
    ];

    public IReadOnlyList<ActionStreamSpec> Streams => _streamSpecs;

    /// <summary>Builds one 25-float action row directly (the convenient non-payload path for canned plans).
    /// <paramref name="keyStates"/> is the 23 binary keys in ACTION_KEYS order (camera slots excluded).</summary>
    public static void BuildRow(ReadOnlySpan<byte> keyStates, float cameraX, float cameraY, Span<float> row)
    {
        if (keyStates.Length != KeyCount)
            throw new ArgumentException($"Expected {KeyCount} key states, got {keyStates.Length}.", nameof(keyStates));
        if (row.Length < ActionDim)
            throw new ArgumentException($"row must hold {ActionDim} floats.", nameof(row));
        int k = 0;
        for (int i = 0; i < ActionDim; i++)
        {
            if (i == CameraXIndex) row[i] = Math.Clamp(cameraX, -1f, 1f);
            else if (i == CameraYIndex) row[i] = Math.Clamp(cameraY, -1f, 1f);
            else row[i] = keyStates[k++] != 0 ? 1f : 0f;
        }
    }

    /// <summary>Packs raw inputs into the 31-byte payload this encoder consumes.</summary>
    public static void PackPayload(ReadOnlySpan<byte> keyStates, float cameraX, float cameraY, Span<byte> payload)
    {
        if (keyStates.Length != KeyCount)
            throw new ArgumentException($"Expected {KeyCount} key states, got {keyStates.Length}.", nameof(keyStates));
        if (payload.Length < PayloadBytes)
            throw new ArgumentException($"payload must hold {PayloadBytes} bytes.", nameof(payload));
        keyStates.CopyTo(payload);
        BinaryPrimitives.WriteSingleLittleEndian(payload[KeyCount..], cameraX);
        BinaryPrimitives.WriteSingleLittleEndian(payload[(KeyCount + sizeof(float))..], cameraY);
    }

    public void Encode(in ActionInput action, string streamName, Span<float> destination)
    {
        if (streamName != "vpt")
            throw new ArgumentException($"Unknown stream '{streamName}' (expected 'vpt').", nameof(streamName));
        ReadOnlySpan<byte> payload = action.Payload.Span;
        if (payload.Length < PayloadBytes)
            throw new ArgumentException($"Oasis payload must be {PayloadBytes} bytes; got {payload.Length}.", nameof(action));
        float camX = BinaryPrimitives.ReadSingleLittleEndian(payload[KeyCount..]);
        float camY = BinaryPrimitives.ReadSingleLittleEndian(payload[(KeyCount + sizeof(float))..]);
        BuildRow(payload[..KeyCount], camX, camY, destination);
    }

    public void Dispose()
    {
    }
}
