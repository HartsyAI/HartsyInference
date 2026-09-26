using System.Security.Cryptography;

namespace HartsyInference.ModelAssets.Metadata;

/// <summary>Where a converted artifact came from and what was done to it, recorded in the output's own header so the
/// file answers the question without its manifest.</summary>
public sealed record ArtifactProvenance
{
    /// <summary>Value of <see cref="Component"/> that marks a file as the model's primary weights.</summary>
    public const string MainComponent = "main";

    /// <summary>Component that produced the output, e.g. <c>"HartsyInference.PickleCheckpointRepacker"</c>.</summary>
    public required string Converter { get; init; }

    /// <summary>Which part of the model this file is: <see cref="MainComponent"/> for the primary weights, else a
    /// role such as <c>"codec"</c> or <c>"vocoder"</c>.
    /// <para>Required, because the default is the dangerous one. A model's index admits only
    /// <c>hartsy.component=main</c>, and a component stamped as primary becomes a separately selectable model that
    /// cannot generate anything — so every conversion site has to say which it is writing.</para></summary>
    public required string Component { get; init; }

    /// <summary>Repo the source came from, when it was fetched rather than supplied.</summary>
    public string? SourceRepo { get; init; }

    /// <summary>File name of the source, e.g. <c>"kokoro-v1_0.pth"</c>.</summary>
    public string? SourceFile { get; init; }

    /// <summary>Whole-file SHA-256 of the source, so a repack can be traced back to exact bytes.</summary>
    public string? SourceSha256 { get; init; }

    /// <summary>Precision token of the output, e.g. <c>"bf16"</c>, <c>"fp8-scaled"</c>, <c>"Q4_K_M"</c>. Null when
    /// the conversion changed the container only.</summary>
    public string? Precision { get; init; }

    /// <summary>The provider's model/variant id (<c>hartsy.model_id</c>), e.g. <c>"large-v3"</c> or <c>"1.7B-Base"</c>:
    /// which catalog row a primary file backs.</summary>
    public string? ModelId { get; init; }

    /// <summary>Optional one-line description for the model card.</summary>
    public string? Description { get; init; }

    /// <summary>Provenance for a file converted from <paramref name="sourcePath"/>, hashing the source so the output
    /// can be traced back to exact bytes rather than to a file name that may since have been replaced.</summary>
    public static ArtifactProvenance FromSourceFile(string converter, string component, string sourcePath,
        string? sourceRepo = null, string? precision = null) =>
        new()
        {
            Converter = converter,
            Component = component,
            SourceRepo = sourceRepo,
            SourceFile = Path.GetFileName(sourcePath),
            SourceSha256 = HashFile(sourcePath),
            Precision = precision,
        };

    /// <summary>Lowercase hex SHA-256 of a whole file, or null when it cannot be read — provenance is worth
    /// recording but never worth failing a conversion over.</summary>
    public static string? HashFile(string path)
    {
        try
        {
            using FileStream stream = File.OpenRead(path);
            return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
        }
        catch (IOException)
        {
            return null;
        }
        catch (UnauthorizedAccessException)
        {
            return null;
        }
    }
}
