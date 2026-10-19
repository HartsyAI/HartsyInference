using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Rope;

/// <summary>Device rows for one host table on one backend. Freed through that backend when retired.</summary>
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
}
