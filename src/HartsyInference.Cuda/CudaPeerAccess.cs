using System.Collections.Concurrent;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Logging;

namespace HartsyInference.Cuda;

/// <summary>Probe/enable memo for CUDA peer (P2P/NVLink) access between device pairs. Enablement is per DIRECTED context pair and sticky for the contexts' lifetime, so each pair is probed once and remembered. <c>HARTSY_P2P_DISABLE=1</c> forces every query to false — the deterministic consumer-path test switch (and escape hatch for the flaky-P2P boards the design doc warns about).</summary>
internal static class CudaPeerAccess
{
    private static readonly bool _disabled = EngineKnobs.P2pDisable.Value;

    /// <summary>CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED — a repeat enable, i.e. success for our purposes.</summary>
    private const int AlreadyEnabled = 704;

    /// <summary>Memo keyed by (accessing ordinal, peer ordinal); value = enabled both probe and grant.</summary>
    private static readonly ConcurrentDictionary<(int, int), bool> _pairs = new();

    /// <summary>True when <paramref name="accessor"/>'s context can directly address <paramref name="peer"/>'s memory (probing and enabling on first ask). Binds <paramref name="accessor"/> current on this thread — call from the accessing backend's op context only.</summary>
    internal static bool TryEnable(CudaContext accessor, CudaContext peer)
    {
        if (_disabled || accessor.DeviceOrdinal == peer.DeviceOrdinal)
        {
            return false;
        }
        return _pairs.GetOrAdd((accessor.DeviceOrdinal, peer.DeviceOrdinal), _ =>
        {
            try
            {
                CudaDriverApi.cuDeviceGet(out int dev, accessor.DeviceOrdinal).ThrowOnError();
                CudaDriverApi.cuDeviceGet(out int peerDev, peer.DeviceOrdinal).ThrowOnError();
                CudaDriverApi.cuDeviceCanAccessPeer(out int can, dev, peerDev).ThrowOnError();
                if (can == 0)
                {
                    Logs.Info($"[Cuda] No P2P path from device {accessor.DeviceOrdinal} to {peer.DeviceOrdinal} — cross-GPU copies stage through host.");
                    return false;
                }
                accessor.EnsureCurrent();
                int rc = CudaDriverApi.cuCtxEnablePeerAccess(peer.Handle, 0);
                if (rc != 0 && rc != AlreadyEnabled)
                {
                    rc.ThrowOnError();
                }
                Logs.Info($"[Cuda] P2P enabled: device {accessor.DeviceOrdinal} → {peer.DeviceOrdinal} (direct cross-GPU copies).");
                return true;
            }
            catch (Exception ex)
            {
                Logs.Warning($"[Cuda] P2P probe/enable {accessor.DeviceOrdinal}→{peer.DeviceOrdinal} failed ({ex.Message}) — staging through host.");
                return false;
            }
        });
    }
}
