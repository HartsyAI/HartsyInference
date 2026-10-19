using HartsyInference.Core.Backends;

namespace HartsyInference.Core.Rope;

/// <summary>Host pairs for one position signature in one slot, and the device tables built from them. Guarded by the owning
/// cache's gate.</summary>
internal sealed class RopeHostTable
{
    public RopeHostTable(float[] positions, int tokens, float[] cos, float[] sin)
    {
        Positions = positions;
        Tokens = tokens;
        Cos = cos;
        Sin = sin;
        Devices = new Dictionary<IBackend, RopeDeviceTable>(ReferenceEqualityComparer.Instance);
    }

    public float[] Positions { get; }

    public int Tokens { get; }

    public float[] Cos { get; }

    public float[] Sin { get; }

    public Dictionary<IBackend, RopeDeviceTable> Devices { get; }
}
