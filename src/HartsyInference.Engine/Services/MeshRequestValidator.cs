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
        if (request.PreviewTriangles < 0)
            throw new ArgumentOutOfRangeException(nameof(request), request.PreviewTriangles, "PreviewTriangles must not be negative; 0 returns no preview.").ForField(nameof(MeshRequest.PreviewTriangles));
        // An id the engine does not know fails as an unknown model below, not as a range error.
        if (MeshModelProfiles.Find(id) is not null)
            MeshSeed.Require(request);
        if (request.Mesh && id != MeshModelIds.Sam3dObjects)
            throw new NotSupportedException($"3D model '{id}' takes no Mesh option; it applies to {MeshModelIds.Sam3dObjects}, which returns splats unless asked for a mesh.").ForField(nameof(MeshRequest.Mesh));
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
            RequireMeshOnly(id, request, "rigs a triangle mesh");
            return;
        }
        if (MeshModelIds.PaintMesh.Contains(id))
        {
            ValidatePaint(id, request, matterAvailable);
            return;
        }
        if (MeshModelIds.PartMesh.Contains(id))
        {
            RequireMeshOnly(id, request, "segments a triangle mesh into parts");
            return;
        }
        if (MeshModelIds.XPartMesh.Contains(id))
        {
            ValidateXPart(id, request);
            return;
        }
        if (id != MeshModelIds.HairDeck && !MeshModelIds.ImageToMesh.Contains(id))
            throw new NotSupportedException($"3D model '{id}' has no wired pipeline; supported: {string.Join(", ", MeshModelIds.Described)}.");
        if (request.ScenePrompt is not null)
            throw new NotSupportedException($"3D model '{id}' takes no scene prompt; ScenePrompt applies to scene models (cola3d).").ForField(nameof(MeshRequest.ScenePrompt));
        if (request.Image is null)
            throw new NotSupportedException("Text-to-3D has no wired pipeline; supply MeshRequest.Image for image-to-3D (TripoSR / Hunyuan3D).").ForField(nameof(MeshRequest.Image));
        if (id == MeshModelIds.HairDeck)
        {
            HairDeckRequestMapper.ToOptions(request);
            return;
        }
        HairDeckRequestMapper.RejectForOtherModels(request, id);
        RequireMatteForAuto(id, request, matterAvailable);
        if (id == MeshModelIds.Trellis2)
        {
            Trellis2Options options = Trellis2RequestMapper.ToOptions(request);
            RequireTextureForSize(request, "TRELLIS.2");
            TagPipelineErrors(() => Trellis2ImageTo3DPipeline.ValidateRequest(options, ToInput(request.Image), matterAvailable));
        }
        else if (id == MeshModelIds.Pixal3D)
        {
            Pixal3DOptions options = Pixal3DRequestMapper.ToOptions(request);
            RequireTextureForSize(request, "Pixal3D");
            TagPipelineErrors(() => Pixal3DImageTo3DPipeline.ValidateRequest(options, ToInput(request.Image), matterAvailable));
        }
        else if (id == MeshModelIds.Sam3dObjects)
        {
            Sam3dObjectsRequestMapper.Validate(request);
        }
        else if (MeshModelIds.Hunyuan3D.Contains(id))
        {
            TagPipelineErrors(() => Hunyuan3DShapePipeline.ValidateGrid(request.GridResolution, UsesFlashVdm(id)));
        }
    }

    /// <summary>Whether the catalog build decodes with FlashVDM: it carries the turbo VAE asset.</summary>
    private static bool UsesFlashVdm(string id) =>
        Registry.ModelCatalog.Find(id)?.Assets?.Any(asset => asset.Role == Registry.ModelCatalog.Hunyuan3DFlashVdmVaeRole) == true;

    /// <summary>Runs a pipeline's own request check and tags its exception with the request field it is about, which its messages name in the pipeline's words (<c>grid</c>, <c>FovRadians</c>).</summary>
    private static void TagPipelineErrors(Action check)
    {
        try
        {
            check();
        }
        catch (Exception error) when (error is ArgumentException or HartsyInference.Core.Exceptions.HartsyInferenceException)
        {
            if (PipelineField(error) is string field)
                error.ForField(field);
            throw;
        }
    }

    /// <summary>The Steps, Seed and TextureSize branches read the pipelines' own message text (they share ParamName "options"); rewording those messages needs this list updated.</summary>
    private static string? PipelineField(Exception error)
    {
        if (error is not ArgumentException argument)
            return nameof(MeshRequest.Image);
        string message = argument.Message;
        return argument.ParamName == "image" ? nameof(MeshRequest.Image)
            : argument.InnerException is ArgumentOutOfRangeException { ParamName: "FovRadians" } ? nameof(MeshRequest.FieldOfViewRadians)
            : argument.ParamName == "grid" ? nameof(MeshRequest.GridResolution)
            : message.StartsWith("Image", StringComparison.Ordinal) || message.StartsWith("RGB", StringComparison.Ordinal) || message.StartsWith("Alpha", StringComparison.Ordinal) ? nameof(MeshRequest.Image)
            : message.StartsWith("Steps", StringComparison.Ordinal) ? nameof(MeshRequest.Steps)
            : message.StartsWith("Seed", StringComparison.Ordinal) ? nameof(MeshRequest.Seed)
            : message.StartsWith("TextureSize", StringComparison.Ordinal) ? nameof(MeshRequest.TextureSize)
            : null;
    }

    private static void RequireMatteForAuto(string id, MeshRequest request, bool matterAvailable)
    {
        if (request.Preprocess == MeshPreprocess.Auto && MeshPreparationService.UsesPreparation(id) && request.Image is { } image && !MeshPreparationService.HasUsableAlpha(image) && !matterAvailable)
        {
            throw new NotSupportedException(
                $"Preprocess Auto needs the foreground cut out for {id}: send an image with alpha, or install the BiRefNet matting checkpoint (Vision/BiRefNet-matting/model.safetensors).")
                .ForField(nameof(MeshRequest.Preprocess));
        }
    }

    private static Trellis2InputImage ToInput(ImageData source) =>
        new() { Rgb = source.Rgb, Alpha = source.Alpha, Width = source.Width, Height = source.Height };

    private static void RequireTextureForSize(MeshRequest request, string model)
    {
        if (!request.Texture && request.TextureSize != 0)
            throw new NotSupportedException($"TextureSize needs Texture; leave it at 0 for a shape-only {model} mesh.").ForField(nameof(MeshRequest.TextureSize));
    }

    private static void ValidateScene(string id, MeshRequest request)
    {
        HairDeckRequestMapper.RejectForOtherModels(request, id);
        if (request.SceneMesh is null)
            throw new NotSupportedException($"3D model '{id}' decomposes a triangle scene; supply MeshRequest.SceneMesh as obj or glb.").ForField(nameof(MeshRequest.SceneMesh));
        if (request.Image is not null)
            throw new NotSupportedException($"3D model '{id}' takes a scene mesh, not an image; leave MeshRequest.Image empty.").ForField(nameof(MeshRequest.Image));
        CoLa3DPrompt? prompt = request.ScenePrompt is null ? null : MeshService.ToCoLa3DPrompt(request.ScenePrompt);
        CoLa3DConfig.QuerySourceKind source = prompt?.Kind ?? CoLa3DConfig.QuerySourceKind.Fixed;
        try
        {
            CoLa3DPipeline.ValidateRequest(CoLa3DConfig.ForQuerySource(source), request.GridResolution, prompt);
        }
        catch (ArgumentException error) when (error.ParamName == "resolution")
        {
            error.ForField(nameof(MeshRequest.GridResolution));
            throw;
        }
    }

    /// <summary>The checks of a model that takes a mesh and nothing else (UniRig, P3-SAM); <paramref name="verb"/> says what the model does to it.</summary>
    private static void RequireMeshOnly(string id, MeshRequest request, string verb)
    {
        HairDeckRequestMapper.RejectForOtherModels(request, id);
        if (request.SceneMesh is null)
            throw new NotSupportedException($"3D model '{id}' {verb}; supply MeshRequest.SceneMesh as obj or glb.").ForField(nameof(MeshRequest.SceneMesh));
        if (request.Image is not null)
            throw new NotSupportedException($"3D model '{id}' takes a mesh, not an image; leave MeshRequest.Image empty.").ForField(nameof(MeshRequest.Image));
        if (request.ScenePrompt is not null)
            throw new NotSupportedException($"3D model '{id}' takes no scene prompt; ScenePrompt applies to cola3d.").ForField(nameof(MeshRequest.ScenePrompt));
        if (request.Steps != 0 || request.GridResolution != 0)
        {
            throw new NotSupportedException($"3D model '{id}' has no steps or grid setting; leave Steps and GridResolution at 0.")
                .ForField(request.Steps != 0 ? nameof(MeshRequest.Steps) : nameof(MeshRequest.GridResolution));
        }
    }

    /// <summary>X-Part takes a mesh like P3-SAM does, and its Steps (the flow steps, 0 takes 50) and GridResolution (the marching-cubes grid of each part, 0 takes 256) are real settings.</summary>
    private static void ValidateXPart(string id, MeshRequest request)
    {
        HairDeckRequestMapper.RejectForOtherModels(request, id);
        if (request.SceneMesh is null)
            throw new NotSupportedException($"3D model '{id}' generates the parts of a triangle mesh; supply MeshRequest.SceneMesh as obj or glb.").ForField(nameof(MeshRequest.SceneMesh));
        if (request.Image is not null)
            throw new NotSupportedException($"3D model '{id}' takes a mesh, not an image; leave MeshRequest.Image empty.").ForField(nameof(MeshRequest.Image));
        if (request.ScenePrompt is not null)
            throw new NotSupportedException($"3D model '{id}' takes no scene prompt; ScenePrompt applies to cola3d.").ForField(nameof(MeshRequest.ScenePrompt));
        if (request.Steps is < 0 or > 1000)
            throw new ArgumentOutOfRangeException(nameof(request), request.Steps, "The X-Part flow steps must be between 1 and 1000 (0 takes the default of 50).").ForField(nameof(MeshRequest.Steps));
        if (request.GridResolution != 0 && request.GridResolution is < 32 or > 512)
            throw new ArgumentOutOfRangeException(nameof(request), request.GridResolution, "The X-Part grid must be between 32 and 512 (0 takes the default of 256; upstream's is 512).").ForField(nameof(MeshRequest.GridResolution));
    }

    private static void ValidatePaint(string id, MeshRequest request, bool matterAvailable)
    {
        HairDeckRequestMapper.RejectForOtherModels(request, id);
        if (request.SceneMesh is null || request.Image is null)
        {
            throw new NotSupportedException(
                $"3D model '{id}' textures a triangle mesh from a reference image; supply MeshRequest.SceneMesh (obj or glb) and MeshRequest.Image.")
                .ForField(request.SceneMesh is null ? nameof(MeshRequest.SceneMesh) : nameof(MeshRequest.Image));
        }
        if (request.ScenePrompt is not null)
            throw new NotSupportedException($"3D model '{id}' takes no scene prompt; ScenePrompt applies to cola3d.").ForField(nameof(MeshRequest.ScenePrompt));
        RequireMatteForAuto(id, request, matterAvailable);
        if (request.Steps < 0 || request.GridResolution < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(request), "Steps and GridResolution must not be negative.")
                .ForField(request.Steps < 0 ? nameof(MeshRequest.Steps) : nameof(MeshRequest.GridResolution));
        }
    }
}
