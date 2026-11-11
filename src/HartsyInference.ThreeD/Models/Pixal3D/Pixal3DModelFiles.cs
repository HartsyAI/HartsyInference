using HartsyInference.ThreeD.Models.TrellisV2;

namespace HartsyInference.ThreeD.Models.Pixal3D;

/// <summary>The files of a Pixal3D shape-only install: the projection-mode flows, <c>pipeline.json</c> and NAF in <c>3D/pixal3d</c>; the structure
/// decoder, shape VAE decoder and DINOv3 encoder are the TRELLIS.2 files (byte-identical): the decoders are found in the same folder or beside it
/// (<c>3D/trellis2</c>, <c>TRELLIS2-ShapeVAE</c>), DINOv3 under <c>Vision/Dinov3</c> or <c>DINOv3</c>, never inside this folder.</summary>
public sealed record Pixal3DModelFiles
{
    /// <summary>The released NAF upsampler weights (<c>valeoai/NAF</c>, release <c>model</c>).</summary>
    public const string NafFile = "naf_release.pth";

    /// <summary>Folder of a TRELLIS.2 install, where the shared decoders may already sit.</summary>
    public const string Trellis2Folder = "trellis2";

    public required string Pipeline { get; init; }

    public required string StructureFlow { get; init; }

    public required string StructureDecoder { get; init; }

    public required string ShapeFlow512 { get; init; }

    public required string ShapeFlow1024 { get; init; }

    public required string ShapeDecoder { get; init; }

    public required string Dino { get; init; }

    public required string Naf { get; init; }

    /// <summary>Finds every file from the Pixal3D folder (or any file inside it).</summary>
    /// <exception cref="DirectoryNotFoundException">The folder does not exist.</exception>
    /// <exception cref="FileNotFoundException">One or more files are missing; the message names each with the places searched.</exception>
    public static Pixal3DModelFiles Resolve(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        string dir = File.Exists(path) ? Path.GetDirectoryName(Path.GetFullPath(path))! : Path.GetFullPath(path);
        if (!Directory.Exists(dir))
            throw new DirectoryNotFoundException($"Pixal3D model directory not found: '{dir}'.");
        string trellis2 = Path.Combine(Path.GetFullPath(Path.Combine(dir, "..")), Trellis2Folder);
        List<string> missing = [];
        string Find(string name, params string[] candidates)
        {
            foreach (string candidate in candidates)
            {
                if (File.Exists(candidate))
                    return candidate;
            }

            missing.Add($"{name} (looked in: {string.Join("; ", candidates.Select(Path.GetDirectoryName))})");
            return string.Empty;
        }

        Pixal3DModelFiles files = new()
        {
            Pipeline = Find(Trellis2ModelFiles.PipelineFile, Path.Combine(dir, Trellis2ModelFiles.PipelineFile)),
            StructureFlow = Find(Trellis2ModelFiles.StructureFlowFile, Path.Combine(dir, Trellis2ModelFiles.StructureFlowFile)),
            ShapeFlow512 = Find(Trellis2ModelFiles.ShapeFlow512File, Path.Combine(dir, Trellis2ModelFiles.ShapeFlow512File)),
            ShapeFlow1024 = Find(Trellis2ModelFiles.ShapeFlow1024File, Path.Combine(dir, Trellis2ModelFiles.ShapeFlow1024File)),
            Naf = Find(NafFile, Path.Combine(dir, NafFile)),
            StructureDecoder = Find(
                Trellis2ModelFiles.StructureDecoderFile,
                Path.Combine(dir, Trellis2ModelFiles.StructureDecoderFile),
                Path.Combine(trellis2, Trellis2ModelFiles.StructureDecoderFile)),
            ShapeDecoder = Find(
                Trellis2ModelFiles.ShapeDecoderFile,
                [.. Trellis2ModelFiles.ShapeDecoderCandidates(dir), Path.Combine(trellis2, Trellis2ModelFiles.ShapeDecoderFile)]),
            Dino = Find("DINOv3 " + Trellis2ModelFiles.DinoFile, Trellis2ModelFiles.DinoCandidates(dir)),
        };
        if (missing.Count > 0)
            throw new FileNotFoundException("Pixal3D files not found: " + string.Join(" | ", missing));
        return files;
    }
}
