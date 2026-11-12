using HartsyInference.Engine.Requests;
using HartsyInference.ThreeD.Pipelines;

namespace HartsyInference.Engine.Services;

/// <summary>Maps a <see cref="MeshRequest"/> to a SAM 3D Objects request and rejects the fields SAM 3D does not read, so a request is never silently changed.</summary>
public static class Sam3dObjectsRequestMapper
{
    /// <summary>The structure and the SLAT flow each run this many Euler steps unless <see cref="MeshRequest.Steps"/> says otherwise (upstream's 25).</summary>
    public const int DefaultSteps = 25;

    /// <summary>The pipeline request of a SAM 3D request: the image's alpha plane is the object mask (the upstream input is an RGBA image), MoGe estimates the pointmap.</summary>
    /// <exception cref="NotSupportedException">There is no image or mask, or a field SAM 3D does not read is set.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The seed does not fit 32 bits or the step count is negative.</exception>
    public static Sam3dObjectsRequest ToRequest(MeshRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        Validate(request);
        ImageData image = request.Image!;
        int steps = request.Steps == 0 ? DefaultSteps : request.Steps;
        return new Sam3dObjectsRequest(
            image.Rgb, image.Alpha!, image.Width, image.Height, Pointmap: null,
            Seed: request.Seed < 0 ? null : (int)request.Seed, StructureSteps: steps, SlatSteps: steps);
    }

    /// <summary>Checks a SAM 3D request before any checkpoint loads.</summary>
    /// <exception cref="NotSupportedException">There is no image or mask, or a field SAM 3D does not read is set.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The seed does not fit 32 bits or the step count is negative.</exception>
    public static void Validate(MeshRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        string id = MeshModelIds.Sam3dObjects;
        ImageData image = request.Image ?? throw new NotSupportedException($"3D model '{id}' lifts an object from an image; supply MeshRequest.Image with the object mask as its alpha plane.");
        if (image.Width <= 0 || image.Height <= 0 || image.Rgb.Length != (long)image.Width * image.Height * 3)
            throw new NotSupportedException($"The image must hold {image.Width} x {image.Height} x 3 bytes of RGB.");
        if (!image.HasAlpha)
            throw new NotSupportedException($"3D model '{id}' needs the mask of the object to lift: give an RGBA image whose alpha plane is the mask (the CLI takes --mask for a separate mask file).");
        if (!image.Alpha!.Any(a => a != 0))
            throw new NotSupportedException($"The mask of the object is empty; '{id}' needs at least one non-zero alpha value.");
        if (request.Seed > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(request), request.Seed, $"The {id} seed must be at most {int.MaxValue}.");
        if (request.Steps < 0)
            throw new ArgumentOutOfRangeException(nameof(request), request.Steps, "The step count cannot be negative; 0 uses upstream's 25 for each flow.");
        if (request.GridResolution != 0)
            throw new NotSupportedException($"3D model '{id}' has a fixed 64-cube structure grid; GridResolution must be 0.");
        if (request.Texture || request.TextureSize != 0)
            throw new NotSupportedException($"3D model '{id}' returns Gaussian splats and takes no texture options.");
    }
}
