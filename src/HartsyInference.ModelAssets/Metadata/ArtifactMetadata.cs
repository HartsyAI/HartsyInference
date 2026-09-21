using System.Globalization;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.Metadata;

/// <summary>Builds the <c>__metadata__</c> map that makes a converted checkpoint self-describing.
///
/// <para>SwarmUI classifies a scanned model from <c>modelspec.architecture</c> before it looks at a single tensor,
/// and reads title/author/license from the same block. Everything the engine wrote before this carried no
/// <c>modelspec.*</c> key at all, so a repack we produced landed with a null class — which for an audio model means
/// its parameters silently vanish from the UI.</para>
///
/// <para>The same keys work in a GGUF: SwarmUI parses every GGUF metadata KV into the same
/// <c>__metadata__</c> object it builds for safetensors, so a quantized artifact needs no sidecar either.</para>
///
/// <para>Only the primary weights get an architecture. A codec, vocoder or encoder that happens to live in its own
/// file is part of a model, not a model — stamping one as classifiable would offer it in the model list as
/// something a user could select and generate nothing with.</para></summary>
public static class ArtifactMetadata
{
    /// <summary>SAI ModelSpec revision these keys conform to.</summary>
    public const string SpecVersion = "1.0.1";

    /// <summary>Recorded as <c>modelspec.implementation</c>, naming what can run the file.</summary>
    public const string Implementation = "https://github.com/HartsyAI/HartsyInference";

    /// <summary>Key whose value SwarmUI treats as the tensor-payload digest, <c>0x</c>-prefixed.</summary>
    public const string HashKey = "modelspec.hash_sha256";

    /// <summary>Metadata for a file written through <see cref="PyTorch.PickleCheckpointRepacker"/>, which fills the
    /// hash slot from the payload it is already streaming. The slot is left empty here on purpose — that is the
    /// signal the repacker acts on, and it costs nothing versus a second pass over a multi-gigabyte file.</summary>
    public static Dictionary<string, string> ForRepack(ArtifactIdentity identity, ArtifactProvenance provenance)
    {
        Dictionary<string, string> metadata = Build(identity, provenance);
        metadata[HashKey] = "";
        return metadata;
    }

    /// <summary>Metadata for a file written straight through <see cref="SafeTensorsWriter"/>, with the payload
    /// digest computed from <paramref name="tensors"/> up front so the header can carry it.</summary>
    public static Dictionary<string, string> ForWriter(ArtifactIdentity identity, ArtifactProvenance provenance,
        IReadOnlyDictionary<string, Tensor> tensors)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        Dictionary<string, string> metadata = Build(identity, provenance);
        metadata[HashKey] = "0x" + SafeTensorsWriter.ComputeTensorDataSha256(tensors);
        return metadata;
    }

    /// <summary>Metadata for a container the caller hashes itself, such as a GGUF whose bytes are not laid out by
    /// <see cref="SafeTensorsWriter"/>. No hash key is emitted; SwarmUI treats it as absent rather than wrong.</summary>
    public static Dictionary<string, string> WithoutHash(ArtifactIdentity identity, ArtifactProvenance provenance) =>
        Build(identity, provenance);

    private static Dictionary<string, string> Build(ArtifactIdentity identity, ArtifactProvenance provenance)
    {
        ArgumentNullException.ThrowIfNull(identity);
        ArgumentNullException.ThrowIfNull(provenance);
        if (string.IsNullOrWhiteSpace(provenance.Component))
        {
            throw new ArgumentException(
                $"Provenance for '{identity.EngineId}' names no component. Use "
                + $"'{ArtifactProvenance.MainComponent}' for the primary weights, or the part's role.",
                nameof(provenance));
        }
        bool isPrimary = string.Equals(provenance.Component, ArtifactProvenance.MainComponent, StringComparison.Ordinal);
        Dictionary<string, string> metadata = new(StringComparer.Ordinal)
        {
            ["modelspec.sai_model_spec"] = SpecVersion,
            ["modelspec.implementation"] = Implementation,
            ["modelspec.title"] = Title(identity, provenance, isPrimary),
            ["modelspec.date"] = DateTime.UtcNow.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture),
            ["hartsy.engine_id"] = identity.EngineId,
            ["hartsy.component"] = provenance.Component,
            ["hartsy.converter"] = provenance.Converter,
        };
        if (isPrimary)
        {
            if (string.IsNullOrWhiteSpace(identity.SwarmClassId))
            {
                throw new ArgumentException(
                    $"Identity '{identity.EngineId}' has no SwarmClassId. A published artifact without an "
                    + "architecture classifies as null, which hides every parameter the model would offer.",
                    nameof(identity));
            }
            metadata["modelspec.architecture"] = identity.SwarmClassId;
            metadata["modelspec.author"] = identity.Author;
            metadata["modelspec.license"] = identity.License;
            // Emitted only when the class declares one: a resolution that disagrees makes SwarmUI clone the class
            // with its matcher disabled, and audio classes declare none at all.
            if (!string.IsNullOrWhiteSpace(identity.StandardResolution))
            {
                metadata["modelspec.resolution"] = identity.StandardResolution;
            }
            if (identity.Tags.Count > 0)
            {
                metadata["modelspec.tags"] = string.Join(",", identity.Tags);
            }
        }
        Put(metadata, "modelspec.description", provenance.Description);
        Put(metadata, "hartsy.source_repo", provenance.SourceRepo ?? identity.UpstreamRepo);
        Put(metadata, "hartsy.source_file", provenance.SourceFile);
        Put(metadata, "hartsy.source_sha256", provenance.SourceSha256);
        Put(metadata, "hartsy.precision", provenance.Precision);
        return metadata;
    }

    private static string Title(ArtifactIdentity identity, ArtifactProvenance provenance, bool isPrimary)
    {
        string qualifier = isPrimary ? provenance.Precision ?? "" : provenance.Component;
        return string.IsNullOrWhiteSpace(qualifier) ? identity.DisplayName : $"{identity.DisplayName} ({qualifier})";
    }

    private static void Put(Dictionary<string, string> metadata, string key, string? value)
    {
        if (!string.IsNullOrWhiteSpace(value))
        {
            metadata[key] = value;
        }
    }
}
