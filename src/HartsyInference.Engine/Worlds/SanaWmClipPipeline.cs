using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Requests;
using HartsyInference.World.Camera;
using HartsyInference.World.Pipelines;
using HartsyInference.World.Pipelines.SanaWm;

namespace HartsyInference.Engine.Worlds;

/// <summary>The engine's one-shot adapter for SANA-WM: maps a resolved <see cref="WorldPlan"/> onto a
/// <see cref="SanaWmClipRequest"/>, renders it through <see cref="SanaWmPipeline"/> with components loaded one stage at a time,
/// and hands back the frames. The camera path comes from the plan (poses or an expanded action script), so session actions
/// are refused.</summary>
internal sealed class SanaWmClipPipeline : IOneShotClipPipeline
{
    private readonly Func<ISanaWmComponents> _components;
    private readonly Func<WorldPlan, Tensor?>? _initialLatent;

    /// <summary>Renders with components from <paramref name="components"/>, created once per run.</summary>
    /// <param name="initialLatent">Supplies an injected initial latent for a plan (parity runs); null draws and encodes one.</param>
    public SanaWmClipPipeline(Func<ISanaWmComponents> components, Func<WorldPlan, Tensor?>? initialLatent = null)
    {
        ArgumentNullException.ThrowIfNull(components);
        _components = components;
        _initialLatent = initialLatent;
    }

    /// <inheritdoc/>
    public IReadOnlyList<VideoFrame> GenerateClip(OneShotClipRequest request, CancellationToken cancel)
    {
        ArgumentNullException.ThrowIfNull(request);
        if (request.Actions.Count > 0)
        {
            throw new InvalidOperationException(
                $"'{request.Plan.ModelId}' takes its camera path from the request (poses or an action script); session actions are not accepted.");
        }
        cancel.ThrowIfCancellationRequested();
        Tensor? initialLatent = _initialLatent?.Invoke(request.Plan);
        if (initialLatent is not null)
        {
            request.Phase.Own(initialLatent);
        }
        SanaWmClip clip = new SanaWmPipeline(_components()).Generate(ToClipRequest(request.Plan, initialLatent), cancel);
        VideoFrame[] frames = new VideoFrame[clip.Frames.Count];
        for (int i = 0; i < frames.Length; i++)
        {
            frames[i] = new VideoFrame { Rgb = clip.Frames[i], Width = clip.Width, Height = clip.Height, Index = i };
        }
        return frames;
    }

    /// <summary>The pipeline request for <paramref name="plan"/>: the model defaults fill guidance and shift when the plan names none.</summary>
    internal static SanaWmClipRequest ToClipRequest(WorldPlan plan, Tensor? initialLatent)
    {
        ArgumentNullException.ThrowIfNull(plan);
        WorldRequest request = plan.Request;
        ImageData image = request.InitImage ?? throw new InvalidOperationException("The SANA-WM plan carries no seed image.");
        WorldCameraPath camera = request.Camera ?? throw new InvalidOperationException("The SANA-WM plan carries no camera path.");
        return new SanaWmClipRequest
        {
            ImageRgb = image.Rgb,
            ImageWidth = image.Width,
            ImageHeight = image.Height,
            Prompt = request.Prompt,
            CameraToWorld = camera.Poses,
            Intrinsics = Intrinsics(camera),
            Frames = plan.Frames,
            Steps = plan.Steps > 0 ? plan.Steps : SanaWmPipeline.DefaultSteps,
            Guidance = plan.Guidance ?? SanaWmPipeline.DefaultGuidance,
            FlowShift = plan.FlowShift ?? SanaWmPipeline.DefaultFlowShift,
            Seed = plan.Seed,
            SequentialCfg = plan.SequentialCfg,
            InitialLatent = initialLatent,
        };
    }

    private static PinholeIntrinsics[] Intrinsics(WorldCameraPath camera)
    {
        IReadOnlyList<WorldIntrinsics> source = camera.FrameIntrinsics
            ?? (camera.Intrinsics is WorldIntrinsics single ? [single]
                : throw new InvalidOperationException("The SANA-WM plan carries no intrinsics."));
        PinholeIntrinsics[] result = new PinholeIntrinsics[source.Count];
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = new PinholeIntrinsics((float)source[i].Fx, (float)source[i].Fy, (float)source[i].Cx, (float)source[i].Cy);
        }
        return result;
    }
}
