using System.Globalization;
using HartsyInference.Cli.Infra;
using HartsyInference.Core.Backends;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Io;
using HartsyInference.ThreeD.Pipelines;
using HartsyInference.ThreeD.Pipelines.Requests;
using HartsyInference.Vision.Codec;

namespace HartsyInference.Cli.Dispatch.Handlers;

/// <summary>Image-to-3D mesh generation. Loads TripoSR (feed-forward) or Hunyuan3D (flow-match) from a model directory
/// and turns an input PNG into a GLB mesh. The generation "prompt" is the path to the input image.</summary>
public sealed class ThreeDHandler : IModalityHandler
{
    /// <inheritdoc/>
    public Modality Modality => Modality.ThreeD;

    /// <inheritdoc/>
    public IModalityRunner Load(ModelSpec spec, IBackend backend, IProgressSink progress)
    {
        if (spec.LocalPath is null)
        {
            throw new FileNotFoundException(
                "No 3D model found. Pass the model directory (or checkpoint) via --model-path.");
        }

        string id = (spec.Catalog?.Id ?? "triposr").ToLowerInvariant();
        progress.Stage($"Loading {id} from {Path.GetFileName(spec.LocalPath.TrimEnd(Path.DirectorySeparatorChar))} …");

        if (id == "hunyuan3d")
        {
            Hunyuan3DShapePipeline hunyuan = Hunyuan3DShapePipeline.LoadFromPath(backend, spec.LocalPath);
            return new ThreeDRunner("hunyuan3d", hunyuan, hunyuan.Generate);
        }

        TripoSrPipeline triposr = TripoSrPipeline.LoadFromPath(backend, spec.LocalPath);
        return new ThreeDRunner("triposr", triposr, triposr.Generate);
    }

    /// <inheritdoc/>
    public GeneratedArtifact Run(IModalityRunner runner, string prompt, ParamState parameters, IProgressSink progress, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        ThreeDRunner threeD = (ThreeDRunner)runner;

        string imagePath = prompt.Trim();
        if (!File.Exists(imagePath))
            throw new FileNotFoundException($"Input image not found: {imagePath}");

        (byte[] rgb, int width, int height) = PngDecoder.DecodeFromFile(imagePath);
        int seedParam = parameters.GetInt("seed", -1);

        ImageTo3DRequest request = new ImageTo3DRequest
        {
            ImageRgb = rgb,
            Width = width,
            Height = height,
            GridResolution = parameters.GetInt("grid", 0),
            Steps = parameters.GetInt("steps", 0),
            Seed = seedParam < 0 ? null : seedParam,
        };

        progress.BeginSteps("mesh", 0);
        ThreeDResult result = threeD.Generate(request, p => progress.Step(p.Step, $"{p.Step}/{p.TotalSteps}"));
        progress.EndSteps();

        if (result.Mesh is not { TriangleCount: > 0 } mesh)
            throw new InvalidOperationException("The pipeline produced an empty mesh (try a foreground-on-gray input).");

        byte[] glb = GlbWriter.Write(mesh);
        GeneratedArtifact artifact = new GeneratedArtifact
        {
            Kind = ArtifactKind.Mesh,
            FileBytes = glb,
            Extension = "glb",
            Text = $"{mesh.VertexCount} vertices / {mesh.TriangleCount} triangles (seed {result.Seed})",
        };
        artifact.Meta["model"] = threeD.ModelId;
        artifact.Meta["vertices"] = mesh.VertexCount.ToString(CultureInfo.InvariantCulture);
        artifact.Meta["triangles"] = mesh.TriangleCount.ToString(CultureInfo.InvariantCulture);
        artifact.Meta["seed"] = result.Seed.ToString(CultureInfo.InvariantCulture);
        return artifact;
    }
}
