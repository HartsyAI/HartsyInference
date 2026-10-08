namespace HartsyInference.ModelAssets.MoePack;

/// <summary>
/// The pack's index, written last. It names the topology it was built for, the expert shape and quant dtype, and every record.
/// A reader refuses a directory without <c>COMPLETE</c>, and refuses a fingerprint it was not built for.
/// </summary>
/// <param name="Format">Pack format version; readers reject other values.</param>
/// <param name="TopologyFingerprint">The <c>SparseModelTopology.Fingerprint</c> the experts were packed under.</param>
/// <param name="Hidden">Model width H.</param>
/// <param name="Intermediate">Expert inner width I.</param>
/// <param name="DType">Quant dtype name of every projection (for example Q8_0 or Q4_K).</param>
/// <param name="Records">One record per expert.</param>
public sealed record ExpertPackManifest(int Format, string TopologyFingerprint, int Hidden, int Intermediate, string DType,
        List<ExpertPackRecord> Records);
