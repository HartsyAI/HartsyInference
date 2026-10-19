using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Rope;

/// <summary>Device rows for one host table on one backend. Lease counts are per device, so releasing one backend never waits
/// on another backend's leases. Guarded by the owning cache's gate.</summary>
internal sealed class RopeDeviceTable
{
    public RopeDeviceTable(IBackend backend, Tensor cos, Tensor sin)
    {
        Backend = backend;
        Cos = cos;
        Sin = sin;
    }

    public IBackend Backend { get; }

    public Tensor Cos { get; }

    public Tensor Sin { get; }

    public int Outstanding { get; set; }

    /// <summary>Set when the table has been unmapped while leases were outstanding; freed by the last release.</summary>
    public bool Retired { get; set; }
}
