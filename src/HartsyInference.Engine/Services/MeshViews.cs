using HartsyInference.Engine.Requests;
using HartsyInference.ThreeD.Pipelines.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Where each <see cref="MeshView"/> sits for the models that take extra views: the one place the request's sides are matched to their cameras.</summary>
internal static class MeshViews
{
    /// <summary>The Hunyuan3D-2mv view slot of a side.</summary>
    public static Hunyuan3DView ToHunyuan(MeshView view) => view switch
    {
        MeshView.Left => Hunyuan3DView.Left,
        MeshView.Back => Hunyuan3DView.Back,
        MeshView.Right => Hunyuan3DView.Right,
        _ => throw new ArgumentOutOfRangeException(nameof(view), view, "Not a side view."),
    };

    /// <summary>The Paint camera azimuth (degrees, elevation 0) a side's picture is taken from. Paint's front camera (azimuth 0) looks along world -y with
    /// +z up, so its screen right is world -x, where a subject facing the camera has its left hand; azimuth 90 puts the camera at world -x.</summary>
    public static double PaintAzimuth(MeshView view) => view switch
    {
        MeshView.Left => 90,
        MeshView.Back => 180,
        MeshView.Right => 270,
        _ => throw new ArgumentOutOfRangeException(nameof(view), view, "Not a side view."),
    };
}
