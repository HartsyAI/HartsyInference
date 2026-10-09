using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.MoePack;

/// <summary>Packs a GGUF checkpoint's experts and verifies a pack against the checkpoint it came from.</summary>
public static class GgufExpertPack
{
    /// <summary>Quantizes every expert of <paramref name="ggufPath"/> into a new pack in <paramref name="packDirectory"/>.</summary>
    /// <param name="ggufPath">GGUF checkpoint to read.</param>
    /// <param name="packDirectory">Pack directory to create.</param>
    /// <param name="dtype">Quantized dtype of every projection.</param>
    /// <param name="topologyFingerprint">
    /// The runtime topology fingerprint (SparseModelTopology.Fingerprint) to bind the pack to, or null for the provisional
    /// GGUF-geometry fingerprint. A pack built with one fingerprint is refused by any reader expecting another.
    /// </param>
    /// <exception cref="InvalidOperationException">The directory already holds a completed pack, or another writer holds it.</exception>
    /// <exception cref="ArgumentException">The supplied fingerprint is empty.</exception>
    public static void Write(string ggufPath, string packDirectory, DType dtype, string? topologyFingerprint = null)
    {
        using GgufExpertSource source = GgufExpertSource.Open(ggufPath, topologyFingerprint);
        using ExpertPackWriter writer = new(packDirectory, source.TopologyFingerprint, source.Hidden, source.Intermediate, dtype, source.Keys());
        foreach (ExpertKey key in source.Keys())
        {
            (float[] gate, float[] up, float[] down) = source.Read(key.Layer, key.Expert);
            writer.AddExpert(key.Layer, key.Expert, gate, up, down);
        }
        writer.Finish();
    }

    /// <summary>Checks a pack's checksums and its values against the checkpoint.</summary>
    /// <param name="ggufPath">GGUF checkpoint the pack came from.</param>
    /// <param name="packDirectory">Pack directory to check.</param>
    /// <param name="expectedFingerprint">
    /// The topology fingerprint the pack must have been built for, or null to expect the provisional GGUF-geometry fingerprint.
    /// </param>
    /// <exception cref="ArgumentException">The supplied fingerprint is empty.</exception>
    /// <exception cref="InvalidDataException">The pack is incomplete, or built for another fingerprint or checkpoint geometry.</exception>
    public static ExpertPackVerification Verify(string ggufPath, string packDirectory, string? expectedFingerprint = null)
    {
        if (expectedFingerprint is not null) ArgumentException.ThrowIfNullOrWhiteSpace(expectedFingerprint);
        using GgufExpertSource source = GgufExpertSource.Open(ggufPath);
        string fingerprint = expectedFingerprint ?? source.TopologyFingerprint;
        using ExpertPackReader reader = ExpertPackReader.Open(packDirectory, fingerprint, verifyChecksums: true);
        return ExpertPackVerifier.Verify(reader, key => source.Read(key.Layer, key.Expert), reader.Keys);
    }
}
