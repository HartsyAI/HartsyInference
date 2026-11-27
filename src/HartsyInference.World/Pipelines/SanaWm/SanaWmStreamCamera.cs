using HartsyInference.Core.Tensors;
using HartsyInference.World.Camera;

namespace HartsyInference.World.Pipelines.SanaWm;

/// <summary>The camera of a SANA-WM stream, packed into DiT inputs a chunk at a time. A fixed path is relativized once and
/// read as the chunks reach it; a live path starts at the identity (already relative to frame 0) and is extended by
/// <see cref="SanaWmCameraIntegrator"/> under the controls held or queued when a chunk begins.</summary>
internal sealed class SanaWmStreamCamera
{
    private readonly IReadOnlyList<float[]>? _fixedPoses;
    private readonly LiveCameraPath<SanaWmControls>? _live;
    private readonly PinholeIntrinsics[] _latentIntrinsics;
    private readonly int _latentHeight;
    private readonly int _latentWidth;

    private SanaWmStreamCamera(IReadOnlyList<float[]>? fixedPoses, LiveCameraPath<SanaWmControls>? live,
        PinholeIntrinsics[] latentIntrinsics, int latentHeight, int latentWidth)
    {
        _fixedPoses = fixedPoses;
        _live = live;
        _latentIntrinsics = latentIntrinsics;
        _latentHeight = latentHeight;
        _latentWidth = latentWidth;
    }

    /// <summary>Whether the camera is steered by actions rather than fixed.</summary>
    internal bool IsLive => _live is not null;

    /// <summary>Builds the camera of <paramref name="request"/> for a <paramref name="width"/> by <paramref name="height"/>
    /// cover-cropped image.</summary>
    /// <exception cref="FormatException">The initial action script does not parse.</exception>
    internal static SanaWmStreamCamera Create(SanaWmStreamRequest request, int width, int height)
    {
        CoverCropMapping crop = CoverCropMapping.Create(request.ImageWidth, request.ImageHeight, width, height);
        int latentWidth = width / SanaWmCameraConditioning.SpatialStride;
        int latentHeight = height / SanaWmCameraConditioning.SpatialStride;
        IReadOnlyList<PinholeIntrinsics> source = request.CameraToWorld is not null && request.Frames is int frames
            ? SanaWmCameraConditioning.FitOverTime(request.Intrinsics, frames)
            : request.Intrinsics;
        PinholeIntrinsics[] latent = new PinholeIntrinsics[source.Count];
        for (int i = 0; i < latent.Length; i++)
        {
            (float fx, float fy, float cx, float cy) = crop.MapIntrinsics(source[i].Fx, source[i].Fy, source[i].Cx, source[i].Cy);
            latent[i] = SanaWmCameraConditioning.ToLatent(new PinholeIntrinsics(fx, fy, cx, cy), width, height, latentWidth, latentHeight);
        }

        if (request.CameraToWorld is not null)
            return new SanaWmStreamCamera(SanaWmCameraConditioning.Relativize(request.CameraToWorld), null, latent, latentHeight, latentWidth);

        SanaWmStreamCamera camera = new(null, new LiveCameraPath<SanaWmControls>(new SanaWmCameraIntegrator(), SanaWmControls.None),
            latent, latentHeight, latentWidth);
        if (!string.IsNullOrWhiteSpace(request.InitialActions))
            camera.Send(request.InitialActions);
        return camera;
    }

    /// <summary>Applies a live action: a key set such as <c>wd</c> or <c>none</c> is held from the next unscripted frame on; a
    /// script of <c>&lt;keys&gt;-&lt;frames&gt;</c> segments is queued and played before the held keys resume.</summary>
    /// <exception cref="FormatException">The action is neither a key set nor a script.</exception>
    /// <exception cref="InvalidOperationException">The stream's camera path is fixed.</exception>
    internal void Send(string action)
    {
        ArgumentNullException.ThrowIfNull(action);
        if (_live is null)
            throw new InvalidOperationException("This SANA-WM stream follows the camera path it was opened with; it takes no live actions.");
        if (!action.Contains('-'))
        {
            _live.Hold(SanaWmAction.ParseKeys(action));
            return;
        }

        SanaWmControls[] perFrame = SanaWmAction.Parse(action);
        int start = 0;
        for (int i = 1; i <= perFrame.Length; i++)
        {
            if (i < perFrame.Length && perFrame[i] == perFrame[start])
                continue;
            _live.Enqueue(perFrame[start], i - start);
            start = i;
        }
    }

    /// <summary>The DiT camera inputs of latent frames <c>[firstLatentFrame, firstLatentFrame + latentFrames)</c>, rounded to
    /// BF16 as the DiT reads them: the conditions (20 floats per latent frame) and the host F32 chunk Plücker
    /// <c>[1, 48, latentFrames, H, W]</c>, which the caller owns. A live path is extended to the chunk's last frame first.</summary>
    internal (float[] Conditions, Tensor ChunkPlucker) Chunk(int firstLatentFrame, int latentFrames)
    {
        int poses = (firstLatentFrame + latentFrames - 1) * SanaWmCameraConditioning.TemporalStride + 1;
        IReadOnlyList<float[]> path = _live?.ExtendTo(poses) ?? _fixedPoses!;
        (float[] conditions, Tensor plucker) = SanaWmCameraConditioning.PackRange(path, _latentIntrinsics, _latentHeight, _latentWidth,
            firstLatentFrame, latentFrames);
        SanaWmPipeline.RoundToBf16(conditions);
        SanaWmPipeline.RoundToBf16(plucker.AsSpan<float>());
        return (conditions, plucker);
    }
}
