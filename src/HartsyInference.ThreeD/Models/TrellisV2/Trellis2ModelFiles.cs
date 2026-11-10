namespace HartsyInference.ThreeD.Models.TrellisV2;

/// <summary>The files of a TRELLIS.2 shape-only install: the flow set under <c>3D/trellis2</c>, plus the shape VAE decoder and the
/// DINOv3 encoder, which may sit in their own folders beside it (<c>TRELLIS2-ShapeVAE</c>, <c>DINOv3</c> or <c>Vision/Dinov3</c>).</summary>
public sealed record Trellis2ModelFiles
{
    public const string PipelineFile = "pipeline.json";
    public const string StructureFlowFile = "ss_flow_img_dit_1_3B_64_bf16.safetensors";
    public const string StructureDecoderFile = "ss_dec_conv3d_16l8_fp16.safetensors";
    public const string ShapeFlow512File = "slat_flow_img2shape_dit_1_3B_512_bf16.safetensors";
    public const string ShapeFlow1024File = "slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors";
    public const string ShapeDecoderFile = "shape_dec_next_dc_f16c32_fp16.safetensors";
    public const string DinoFile = "model.safetensors";

    public required string Pipeline { get; init; }

    public required string StructureFlow { get; init; }

    public required string StructureDecoder { get; init; }

    public required string ShapeFlow512 { get; init; }

    public required string ShapeFlow1024 { get; init; }

    public required string ShapeDecoder { get; init; }

    public required string Dino { get; init; }

    /// <summary>Finds every file from the flow folder (or any file inside it), looking in the sibling folders for the shared ones.</summary>
    /// <exception cref="DirectoryNotFoundException">The folder does not exist.</exception>
    /// <exception cref="FileNotFoundException">One or more files are missing; the message names each with the places searched.</exception>
    public static Trellis2ModelFiles Resolve(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        string dir = File.Exists(path) ? Path.GetDirectoryName(Path.GetFullPath(path))! : Path.GetFullPath(path);
        if (!Directory.Exists(dir))
            throw new DirectoryNotFoundException($"TRELLIS.2 model directory not found: '{dir}'.");
        string parent = Path.GetFullPath(Path.Combine(dir, ".."));
        string root = Path.GetFullPath(Path.Combine(dir, "..", ".."));
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

        Trellis2ModelFiles files = new()
        {
            Pipeline = Find(PipelineFile, Path.Combine(dir, PipelineFile)),
            StructureFlow = Find(StructureFlowFile, Path.Combine(dir, StructureFlowFile)),
            StructureDecoder = Find(StructureDecoderFile, Path.Combine(dir, StructureDecoderFile)),
            ShapeFlow512 = Find(ShapeFlow512File, Path.Combine(dir, ShapeFlow512File)),
            ShapeFlow1024 = Find(ShapeFlow1024File, Path.Combine(dir, ShapeFlow1024File)),
            ShapeDecoder = Find(
                ShapeDecoderFile,
                Path.Combine(dir, ShapeDecoderFile),
                Path.Combine(parent, "TRELLIS2-ShapeVAE", ShapeDecoderFile),
                Path.Combine(root, "TRELLIS2-ShapeVAE", ShapeDecoderFile)),
            Dino = Find(
                "DINOv3 " + DinoFile,
                Path.Combine(root, "Vision", "Dinov3", DinoFile),
                Path.Combine(root, "DINOv3", DinoFile),
                Path.Combine(parent, "DINOv3", DinoFile),
                Path.Combine(parent, "Vision", "Dinov3", DinoFile)),
        };
        if (missing.Count > 0)
            throw new FileNotFoundException("TRELLIS.2 files not found: " + string.Join(" | ", missing));
        return files;
    }
}
