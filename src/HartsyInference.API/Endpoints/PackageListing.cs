using HartsyInference.Core.IO;
using HartsyInference.Engine;
using HartsyInference.ModelAssets.Checkpoints;

namespace HartsyInference.API.Endpoints;

/// <summary>The <c>/admin/packages?root=</c> route: the text model packages under a directory, discovered from disk. Nothing is loaded.</summary>
internal static class PackageListing
{
    /// <summary>Lists the packages under <paramref name="root"/>, or under the default text folder (the <c>LLM</c> folder in the models root) when none is given. The root must be
    /// the models root or a directory under it, and a relative root is taken under it; any other root is a 400. A root that is given but does not exist is a 404, and a
    /// default folder that does not exist lists nothing. Error messages repeat the root as given, never the resolved server path.</summary>
    public static IResult Respond(string? root)
    {
        bool explicitRoot = !string.IsNullOrWhiteSpace(root);
        string modelsRoot = Path.GetFullPath(RepoPaths.ModelsRoot());
        string scanRoot;
        try
        {
            scanRoot = Path.GetFullPath(explicitRoot ? root! : DefaultRoot(modelsRoot), modelsRoot);
        }
        catch (Exception ex) when (ex is ArgumentException or NotSupportedException)
        {
            return HartsyInferenceServiceExtensions.Problem(StatusCodes.Status400BadRequest, $"'{root}' is not a valid path: {ex.Message}", "invalid_request_error");
        }
        if (!IsUnder(scanRoot, modelsRoot))
        {
            return HartsyInferenceServiceExtensions.Problem(StatusCodes.Status400BadRequest,
                $"'{root}' is outside the models root; only the models root and the directories under it can be listed.", "invalid_request_error");
        }
        if (explicitRoot && !Directory.Exists(scanRoot))
            return HartsyInferenceServiceExtensions.Problem(StatusCodes.Status404NotFound, $"'{root}' is not a directory.", "invalid_request_error");

        TextPackageScan scan = TextPackageDiscovery.Discover(scanRoot);
        return Results.Ok(new PackageListResponse
        {
            Root = Path.GetRelativePath(modelsRoot, scanRoot),
            Packages = [.. scan.Packages.Select(p => ToDto(p, modelsRoot))],
            Problems = scan.Problems,
        });
    }

    /// <summary>The default scan folder: the text models folder under the models root. The engine names it <c>LLM</c>, so the folder is matched in any case; a
    /// lowercase guess finds nothing on a case-sensitive filesystem.</summary>
    internal static string DefaultRoot(string modelsRoot) => CaseInsensitivePath.ResolveDirectory(modelsRoot, "LLM");

    /// <summary>Whether <paramref name="path"/> is <paramref name="root"/> or lies under it, by name with the platform's case rule. Links are not resolved, so a link
    /// placed under the models root is followed by the scan.</summary>
    internal static bool IsUnder(string path, string root)
    {
        string relative = Path.GetRelativePath(root, path);
        return relative == "." || !(Path.IsPathRooted(relative) || relative == ".."
            || relative.StartsWith(".." + Path.DirectorySeparatorChar, StringComparison.Ordinal));
    }

    /// <summary>The wire shape of one discovered package. Its paths, and its sidecars' paths, are relative to the models root, so no server path leaves the route.</summary>
    internal static PackageDto ToDto(TextModelPackage package, string modelsRoot) => new()
    {
        Id = package.Id,
        Format = package.Format switch
        {
            TextPackageFormat.SafetensorsShards => "safetensors_shards",
            TextPackageFormat.SplitGguf => "split_gguf",
            _ => "gguf",
        },
        Path = Path.GetRelativePath(modelsRoot, package.EntryPath),
        Files = package.Files,
        TotalBytes = package.TotalBytes,
        Family = package.Family,
        Quant = package.Quant,
        Sidecars = [.. package.Sidecars.Select(s => new SidecarDto { Role = s.Role, Path = Path.GetRelativePath(modelsRoot, s.Path) })],
        Capabilities = new CapabilitiesDto { Speculation = package.Speculation.Supported, SpeculationReason = package.Speculation.Reason },
        Problems = package.Problems,
    };
}
