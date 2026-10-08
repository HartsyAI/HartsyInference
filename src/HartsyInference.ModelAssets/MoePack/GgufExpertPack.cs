using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.MoePack;

/// <summary>Packs a GGUF checkpoint's experts and verifies a pack against the checkpoint it came from.</summary>
public static class GgufExpertPack
{
    /// <summary>Quantizes every expert of <paramref name="ggufPath"/> into a new pack in <paramref name="packDirectory"/>.</summary>
    /// <exception cref="InvalidOperationException">The directory already holds a completed pack, or another writer holds it.</exception>
    public static void Write(string ggufPath, string packDirectory, DType dtype)
    {
        using GgufExpertSource source = GgufExpertSource.Open(ggufPath);
        using ExpertPackWriter writer = new(packDirectory, source.TopologyFingerprint, source.Hidden, source.Intermediate, dtype, source.Keys());
        foreach (ExpertKey key in source.Keys())
        {
            (float[] gate, float[] up, float[] down) = source.Read(key.Layer, key.Expert);
            writer.AddExpert(key.Layer, key.Expert, gate, up, down);
        }
        writer.Finish();
    }

    /// <summary>Checks a pack's checksums and its values against the checkpoint.</summary>
    /// <exception cref="InvalidDataException">The pack is incomplete, or built for a different checkpoint geometry.</exception>
    public static ExpertPackVerification Verify(string ggufPath, string packDirectory)
    {
        using GgufExpertSource source = GgufExpertSource.Open(ggufPath);
        using ExpertPackReader reader = ExpertPackReader.Open(packDirectory, source.TopologyFingerprint, verifyChecksums: true);
        return ExpertPackVerifier.Verify(reader, key => source.Read(key.Layer, key.Expert), reader.Keys);
    }
}
