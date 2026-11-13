using HartsyInference.Core.Configuration;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Registry;
using HartsyInference.ThreeD.Models.TrellisV2;

namespace HartsyInference.Engine.Services;

/// <summary>Answers <see cref="IMeshAssetService"/> from the catalog and the places the loaders look; nothing is loaded or read beyond file metadata.</summary>
public sealed class MeshAssetService : IMeshAssetService
{
    private const string NoFilesMessage = "This model has no downloadable files in the catalog. Place its checkpoint folder on disk and pass it as the model path.";

    /// <summary>Files no catalog entry can fetch, by model id: the file's catalog-relative path, its role and what the user does.</summary>
    private static readonly Dictionary<string, (string Role, string RelativePath, string Instructions)[]> ManualFiles = new(StringComparer.OrdinalIgnoreCase)
    {
        [MeshModelIds.Pixal3D] =
        [
            ("NAF upsampler (manual)", "3D/pixal3d/naf_release.pth",
                "Download the 'model' asset of the valeoai/NAF GitHub release (naf_release.pth, 2,664,431 bytes) and place it beside the model's other files: in 3D/pixal3d for the catalog install, or in the folder you pass as the model path."),
        ],
    };

    /// <inheritdoc/>
    public MeshAssetStatus GetStatus(ModelSpec spec) => GetInventory(spec).Status;

    /// <summary>The instructions of a gated file the user has to unlock on Hugging Face.</summary>
    internal static string GatedInstructions(string repo) =>
        $"{repo} is gated: accept its terms on Hugging Face and set HF_TOKEN to a token with read access, then fetch again.";

    /// <summary>The status of <paramref name="spec"/>'s model with the catalog asset and found path behind each row.</summary>
    internal MeshInventory GetInventory(ModelSpec spec)
    {
        ArgumentNullException.ThrowIfNull(spec);
        string id = MeshService.IdOf(spec);
        if (MeshModelProfiles.Find(id) is null)
            throw new NotSupportedException($"3D model '{id}' has no wired pipeline; supported: {string.Join(", ", MeshModelIds.Described)}.");
        CatalogEntry? entry = spec.Catalog ?? ModelCatalog.Find(id);
        if (entry is null || entry.Assets.Count == 0)
        {
            MeshAssetStatus none = new()
            {
                ModelId = id,
                Assets = [],
                ModelBlocker = new MeshAssetBlocker("checkpoint", MeshAssetBlockerKind.NoCatalogEntry, NoFilesMessage),
            };
            return new MeshInventory(none, [], []);
        }

        bool token = HasToken();
        string root = RepoPaths.ModelsRoot();
        List<MeshAssetInfo> assets = [];
        List<ModelAsset?> catalog = [];
        List<string?> paths = [];
        foreach (ModelAsset asset in entry.Assets)
        {
            string? path = Locate(asset, id, entry, spec.LocalPath, root);
            bool gated = IsGated(asset);
            bool blockedByToken = gated && !token;
            assets.Add(new MeshAssetInfo
            {
                Role = asset.Role,
                RelativePath = Relative(asset),
                Required = !IsOptional(asset),
                Present = path is not null,
                SizeBytes = path is null ? null : new FileInfo(path).Length,
                Gated = gated,
                Blocker = path is null && blockedByToken ? MeshAssetBlockerKind.Gated : null,
                Instructions = path is null && blockedByToken ? GatedInstructions(asset.Repo) : null,
            });
            catalog.Add(asset);
            paths.Add(path);
        }
        if (ManualFiles.TryGetValue(id, out (string Role, string RelativePath, string Instructions)[]? manual))
        {
            foreach ((string role, string relative, string instructions) in manual)
            {
                string? path = FindManual(relative, spec.LocalPath, root);
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
                catalog.Add(null);
                paths.Add(path);
            }
        }
        return new MeshInventory(new MeshAssetStatus { ModelId = id, Assets = assets }, catalog, paths);
    }

    /// <summary>Whether a Hugging Face token is configured, the way the downloader reads it.</summary>
    internal static bool HasToken() => !string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("HF_TOKEN"));

    /// <summary>The catalog marks a gated repository in the role text, as <c>encoder (gated)</c>; the catalog test pins every role that carries the word.</summary>
    internal static bool IsGated(ModelAsset asset) => asset.Role.Contains("gated", StringComparison.OrdinalIgnoreCase);

    /// <summary>The catalog marks an asset only an optional feature needs in the role text, as <c>matting (optional)</c>.</summary>
    internal static bool IsOptional(ModelAsset asset) => asset.Role.Contains("optional", StringComparison.OrdinalIgnoreCase);

    /// <summary>The path the loader will read <paramref name="asset"/> from, or null: the catalog path, a legacy name, the model folder the caller named, or a folder the loader also searches.</summary>
    internal static string? Locate(ModelAsset asset, string id, CatalogEntry entry, string? localPath, string root)
    {
        // For the two part models a path the caller names decides, found or not: their loaders read it and never the catalog folder.
        if (localPath is not null && id is MeshModelIds.P3Sam or MeshModelIds.XPart)
            return LoaderPath(asset, id, localPath);
        string? found = ModelDownloader.FindExisting(asset, ignoreCase: false) ?? ModelDownloader.FindExisting(asset, ignoreCase: true);
        if (found is not null)
            return found;
        string? folder = ModelFolderOf(localPath);
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

    /// <summary>Where the P3-SAM and X-Part loaders read <paramref name="asset"/> from when the caller names a path: P3-SAM takes the checkpoint file or the folder holding it (itself or in <c>p3sam/</c>), X-Part the folder with
    /// the four checkpoint subfolders or a file or subfolder inside it. The status then agrees with what a run would load, file by file.</summary>
    private static string? LoaderPath(ModelAsset asset, string id, string? localPath)
    {
        if (localPath is null)
            return null;
        if (id == MeshModelIds.P3Sam)
        {
            try
            {
                return MeshService.PartCheckpoint(localPath);
            }
            catch (FileNotFoundException)
            {
                return null;
            }
        }
        if (id != MeshModelIds.XPart)
            return null;
        // The folder the loader would use, then each file in it: a file found in another folder is not one the loader reads.
        string file = Path.Combine(MeshService.XPartInstall(localPath).Folder, asset.FileName);
        return File.Exists(file) ? file : null;
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

    /// <summary>A file placed by hand is looked up where the loader reads it: beside the checkpoints of the model folder the caller named, else in the catalog folder.</summary>
    private static string? FindManual(string relative, string? localPath, string root)
    {
        string name = Path.GetFileName(relative);
        string? folder = ModelFolderOf(localPath);
        if (folder is not null)
            return FirstExisting(Path.Combine(folder, name));
        return FirstExisting(Path.Combine(root, ModelFolders.MapSubdir(Path.GetDirectoryName(relative) ?? ""), name));
    }

    private static string? ModelFolderOf(string? localPath) => localPath is null ? null : File.Exists(localPath) ? Path.GetDirectoryName(localPath) : localPath;

    private static string? FirstExisting(string path) => File.Exists(path) ? path : null;

    private static string Relative(ModelAsset asset) => Path.Combine(asset.TargetSubdir, asset.FileName).Replace('\\', '/');
}
