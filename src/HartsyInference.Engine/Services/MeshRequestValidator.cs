using HartsyInference.Engine.Requests;
using HartsyInference.ThreeD.Models.CoLa3D;
using HartsyInference.ThreeD.Pipelines;
using HartsyInference.ThreeD.Pipelines.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>The checks a mesh request passes before any checkpoint loads. Generation and planning both call <see cref="Validate"/>, so a plan and a run cannot disagree.</summary>
internal static class MeshRequestValidator
{
    /// <summary>Throws the exception generation would throw for a request this model cannot run; returns when the request is acceptable up to the checkpoint.</summary>
    /// <param name="id">The lower-case catalog id.</param>
    /// <param name="matterAvailable">Whether the matting checkpoint TRELLIS.2 and Pixal3D use for images without alpha is installed.</param>
    /// <exception cref="NotSupportedException">The model has no wired pipeline, or the request sets an option or input it does not take.</exception>
    /// <exception cref="ArgumentException">A value is out of the model's range.</exception>
    public static void Validate(string id, MeshRequest request, bool matterAvailable)
    {
        ArgumentNullException.ThrowIfNull(request);
        if (id != MeshModelIds.Trellis2 && id != MeshModelIds.Pixal3D)
            Trellis2RequestMapper.RejectForOtherModels(request, id);
        Pixal3DRequestMapper.RejectForOtherModels(request, id);
        if (MeshModelIds.SceneMesh.Contains(id))
        {
            ValidateScene(id, request);
            return;
        }
        if (MeshModelIds.RigMesh.Contains(id))
        {
            ValidateRig(id, request);
            return;
        }
        if (MeshModelIds.PaintMesh.Contains(id))
        {
            ValidatePaint(id, request);
            return;
        }
        if (MeshModelIds.PartMesh.Contains(id))
        {
            ValidatePart(id, request);
            return;
        }
        if (id != MeshModelIds.HairDeck && !MeshModelIds.ImageToMesh.Contains(id))
            throw new NotSupportedException($"3D model '{id}' has no wired pipeline; supported: {string.Join(", ", MeshModelIds.Described)}.");
        if (request.ScenePrompt is not null)
            throw new NotSupportedException($"3D model '{id}' takes no scene prompt; ScenePrompt applies to scene models (cola3d).");
        if (request.Image is null)
            throw new NotSupportedException("Text-to-3D has no wired pipeline; supply MeshRequest.Image for image-to-3D (TripoSR / Hunyuan3D).");
        if (id == MeshModelIds.HairDeck)
        {
            HairDeckRequestMapper.ToOptions(request);
            return;
        }
        HairDeckRequestMapper.RejectForOtherModels(request, id);
        if (id == MeshModelIds.Trellis2)
        {
            Trellis2Options options = Trellis2RequestMapper.ToOptions(request);
            RequireTextureForSize(request, "TRELLIS.2");
            Trellis2ImageTo3DPipeline.ValidateRequest(options, ToInput(request.Image), matterAvailable);
        }
        else if (id == MeshModelIds.Pixal3D)
        {
            Pixal3DOptions options = Pixal3DRequestMapper.ToOptions(request);
            RequireTextureForSize(request, "Pixal3D");
            Pixal3DImageTo3DPipeline.ValidateRequest(options, ToInput(request.Image), matterAvailable);
        }
        else if (id == MeshModelIds.Sam3dObjects)
        {
            Sam3dObjectsRequestMapper.Validate(request);
        }
    }

    private static Trellis2InputImage ToInput(ImageData source) =>
        new() { Rgb = source.Rgb, Alpha = source.Alpha, Width = source.Width, Height = source.Height };

    private static void RequireTextureForSize(MeshRequest request, string model)
    {
        if (!request.Texture && request.TextureSize != 0)
            throw new NotSupportedException($"TextureSize needs Texture; leave it at 0 for a shape-only {model} mesh.");
    }

    private static void ValidateScene(string id, MeshRequest request)
    {
        HairDeckRequestMapper.RejectForOtherModels(request, id);
        if (request.SceneMesh is null)
            throw new NotSupportedException($"3D model '{id}' decomposes a triangle scene; supply MeshRequest.SceneMesh as obj or glb.");
        if (request.Image is not null)
            throw new NotSupportedException($"3D model '{id}' takes a scene mesh, not an image; leave MeshRequest.Image empty.");
        CoLa3DPrompt? prompt = request.ScenePrompt is null ? null : MeshService.ToCoLa3DPrompt(request.ScenePrompt);
        CoLa3DConfig.QuerySourceKind source = prompt?.Kind ?? CoLa3DConfig.QuerySourceKind.Fixed;
        CoLa3DPipeline.ValidateRequest(CoLa3DConfig.ForQuerySource(source), request.GridResolution, prompt);
    }

    private static void ValidateRig(string id, MeshRequest request)
    {
        HairDeckRequestMapper.RejectForOtherModels(request, id);
        if (request.SceneMesh is null)
            throw new NotSupportedException($"3D model '{id}' rigs a triangle mesh; supply MeshRequest.SceneMesh as obj or glb.");
        if (request.Image is not null)
            throw new NotSupportedException($"3D model '{id}' takes a mesh, not an image; leave MeshRequest.Image empty.");
        if (request.ScenePrompt is not null)
            throw new NotSupportedException($"3D model '{id}' takes no scene prompt; ScenePrompt applies to cola3d.");
        if (request.Steps != 0 || request.GridResolution != 0)
            throw new NotSupportedException($"3D model '{id}' has no steps or grid setting; leave Steps and GridResolution at 0.");
    }

    private static void ValidatePart(string id, MeshRequest request)
    {
        ValidateRig(id, request);
        if (request.Seed > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(request), request.Seed, $"The P3-SAM seed must be at most {int.MaxValue}.");
    }

    private static void ValidatePaint(string id, MeshRequest request)
    {
        HairDeckRequestMapper.RejectForOtherModels(request, id);
        if (request.SceneMesh is null || request.Image is null)
        {
            throw new NotSupportedException(
                $"3D model '{id}' textures a triangle mesh from a reference image; supply MeshRequest.SceneMesh (obj or glb) and MeshRequest.Image.");
        }
        if (request.ScenePrompt is not null)
            throw new NotSupportedException($"3D model '{id}' takes no scene prompt; ScenePrompt applies to cola3d.");
        if (request.Seed > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(request), request.Seed, $"The Paint seed must be at most {int.MaxValue}.");
        if (request.Steps < 0 || request.GridResolution < 0)
            throw new ArgumentOutOfRangeException(nameof(request), "Steps and GridResolution must not be negative.");
    }
}
