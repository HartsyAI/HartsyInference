using HartsyInference.Core.Configuration;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Registry;
using HartsyInference.ThreeD.Models.TrellisV2;

namespace HartsyInference.Engine.Services;

/// <summary>Answers <see cref="IMeshAssetService"/> from the catalog and the places the loaders look; nothing is loaded or read beyond file metadata.</summary>
public sealed class MeshAssetService : IMeshAssetService
{
    /// <summary>Files no catalog entry can fetch, by model id: the file's catalog-relative path, its role and what the user does.</summary>
    private static readonly Dictionary<string, (string Role, string RelativePath, string Instructions)[]> ManualFiles = new(StringComparer.OrdinalIgnoreCase)
    {
        [MeshModelIds.Pixal3D] =
        [
            ("NAF upsampler (manual)", "3D/pixal3d/naf_release.pth",
                "Download the 'model' asset of the valeoai/NAF GitHub release (naf_release.pth, 2,664,431 bytes) and place it in 3D/pixal3d."),
        ],
    };

    private const string NoFilesMessage = "This model has no downloadable files in the catalog. Place its checkpoint folder on disk and pass it as the model path.";

    /// <inheritdoc/>
    public MeshAssetStatus GetStatus(ModelSpec spec)
    {
        ArgumentNullException.ThrowIfNull(spec);
        string id = MeshService.IdOf(spec);
        if (MeshModelProfiles.Find(id) is null)
            throw new NotSupportedException($"3D model '{id}' has no wired pipeline; supported: {string.Join(", ", MeshModelIds.Described)}.");
        CatalogEntry? entry = spec.Catalog ?? ModelCatalog.Find(id);
        if (entry is null || entry.Assets.Count == 0)
        {
            return new MeshAssetStatus
            {
                ModelId = id,
                Assets = [],
                ModelBlocker = new MeshAssetBlocker("checkpoint", MeshAssetBlockerKind.NoCatalogEntry, NoFilesMessage),
            };
        }

        bool token = HasToken();
        string root = RepoPaths.ModelsRoot();
        List<MeshAssetInfo> assets = [];
        foreach (ModelAsset asset in entry.Assets)
        {
            string? path = Locate(asset, id, entry, spec.LocalPath, root);
            bool gated = asset.Role.Contains("gated", StringComparison.OrdinalIgnoreCase);
            bool blockedByToken = gated && !token;
            assets.Add(new MeshAssetInfo
            {
                Role = asset.Role,
                RelativePath = Relative(asset),
                Required = !asset.Role.Contains("optional", StringComparison.OrdinalIgnoreCase),
                Present = path is not null,
                SizeBytes = path is null ? null : new FileInfo(path).Length,
                Gated = gated,
                Blocker = path is null && blockedByToken ? MeshAssetBlockerKind.Gated : null,
                Instructions = path is null && blockedByToken
                    ? $"{asset.Repo} is gated: accept its terms on Hugging Face and set HF_TOKEN to a token with read access, then fetch again."
                    : null,
            });
        }
        if (ManualFiles.TryGetValue(id, out (string Role, string RelativePath, string Instructions)[]? manual))
        {
            foreach ((string role, string relative, string instructions) in manual)
            {
                string? path = FindManual(relative, root);
                assets.Add(new MeshAssetInfo
                {
                    Role = role,
                    RelativePath = relative,
                    Required = true,
                    Present = path is not null,
                    SizeBytes = path is null ? null : new FileInfo(path).Length,
                    Blocker = path is null ? MeshAssetBlockerKind.ManualPlacement : null,
                    Instructions = path is null ? instructions : null,
                });
            }
        }
        return new MeshAssetStatus { ModelId = id, Assets = assets };
    }

    /// <summary>Whether a Hugging Face token is configured, the way the downloader reads it.</summary>
    internal static bool HasToken() => !string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("HF_TOKEN"));

    /// <summary>The path the loader will read <paramref name="asset"/> from, or null: the catalog path, a legacy name, the model folder the caller named, or a folder the loader also searches.</summary>
    internal static string? Locate(ModelAsset asset, string id, CatalogEntry entry, string? localPath, string root)
    {
        string? found = ModelDownloader.FindExisting(asset, ignoreCase: false) ?? ModelDownloader.FindExisting(asset, ignoreCase: true);
        if (found is not null)
            return found;
        string? folder = localPath is null ? null : File.Exists(localPath) ? Path.GetDirectoryName(localPath) : localPath;
        if (entry.LoadsFromFolder && folder is not null && Directory.Exists(folder))
        {
            string inFolder = Path.Combine(folder, asset.FileName);
            if (File.Exists(inFolder))
                return inFolder;
            // SAM 3D accepts Meta's pickled .ckpt in place of the safetensors conversion.
            string ckpt = Path.Combine(folder, Path.ChangeExtension(asset.FileName, ".ckpt"));
            if (id == MeshModelIds.Sam3dObjects && File.Exists(ckpt))
                return ckpt;
        }
        return Alternative(asset, id, root);
    }

    /// <summary>Where a loader also looks for a shared file: the shape VAE folder beside the TRELLIS.2 flows, a DINOv3 folder at the root, the DINOv2 copy inside a Hunyuan3D shape checkpoint.</summary>
    private static string? Alternative(ModelAsset asset, string id, string root)
    {
        if (id is MeshModelIds.Trellis2 or MeshModelIds.Pixal3D)
        {
            if (asset.FileName is Trellis2ModelFiles.ShapeDecoderFile or Trellis2ModelFiles.TextureDecoderFile)
                return FirstExisting(Path.Combine(root, "TRELLIS2-ShapeVAE", asset.FileName));
            if (asset.FileName == Trellis2ModelFiles.DinoFile && asset.TargetSubdir.EndsWith("Dinov3", StringComparison.OrdinalIgnoreCase))
                return FirstExisting(Path.Combine(root, "DINOv3", asset.FileName));
        }
        if (id == "hunyuan3d-paint21" && asset.TargetName?.StartsWith("dino", StringComparison.OrdinalIgnoreCase) == true)
            return MeshService.FindPaintDino(Path.Combine(root, ModelFolders.MapSubdir(asset.TargetSubdir)));
        return null;
    }

    private static string? FindManual(string relative, string root)
    {
        string path = Path.Combine(root, ModelFolders.MapSubdir(Path.GetDirectoryName(relative) ?? ""), Path.GetFileName(relative));
        return FirstExisting(path);
    }

    private static string? FirstExisting(string path) => File.Exists(path) ? path : null;

    private static string Relative(ModelAsset asset) => Path.Combine(asset.TargetSubdir, asset.FileName).Replace('\\', '/');
}
