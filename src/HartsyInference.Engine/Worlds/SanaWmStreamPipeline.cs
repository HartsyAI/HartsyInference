using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Requests;
using HartsyInference.World.Camera;
using HartsyInference.World.Pipelines;
using HartsyInference.World.Pipelines.SanaWm;
using HartsyInference.World.Sessions;

namespace HartsyInference.Engine.Worlds;

/// <summary>The engine's streaming adapter for SANA-WM: maps a resolved <see cref="WorldPlan"/> onto a
/// <see cref="SanaWmStreamRequest"/> and opens it through <see cref="SanaWmStreamingPipeline"/>. A plan with camera poses
/// streams along that fixed path; otherwise the camera is live, starting with the plan's action script when it has one and
/// then following the actions the session is sent. A plan with the refiner runs the LTX-2 refiner stage, sized by the
/// <c>vram.sanaWmRefiner*</c> settings.</summary>
internal sealed class SanaWmStreamPipeline : IChunkedWorldPipeline
{
    private readonly Func<ISanaWmStreamingComponents> _components;
    private readonly Func<WorldPlan, SanaWmStreamInjection?>? _injection;

    /// <summary>Streams with components from <paramref name="components"/>, created once per stream.</summary>
    /// <param name="injection">Supplies recorded upstream tensors for a plan (parity runs); null draws and encodes them.</param>
    public SanaWmStreamPipeline(Func<ISanaWmStreamingComponents> components, Func<WorldPlan, SanaWmStreamInjection?>? injection = null)
    {
        ArgumentNullException.ThrowIfNull(components);
        _components = components;
        _injection = injection;
    }

    /// <inheritdoc/>
    public IChunkedWorldStream OpenStream(ChunkedWorldRequest request, CancellationToken cancel)
    {
        ArgumentNullException.ThrowIfNull(request);
        cancel.ThrowIfCancellationRequested();
        SanaWmStreamInjection? injection = _injection?.Invoke(request.Plan);
        if (injection?.FirstLatent is Tensor firstLatent)
        {
            request.Phase.Own(firstLatent);
        }
        SanaWmStream stream = new SanaWmStreamingPipeline(_components()).Open(ToStreamRequest(request.Plan, injection), cancel);
        try
        {
            foreach (string action in request.Actions)
            {
                stream.SendAction(action);
            }
            return stream;
        }
        catch
        {
            stream.Dispose();
            throw;
        }
    }

    /// <summary>The stream request for <paramref name="plan"/>: the model defaults fill what the plan leaves unset.</summary>
    internal static SanaWmStreamRequest ToStreamRequest(WorldPlan plan, SanaWmStreamInjection? injection)
    {
        ArgumentNullException.ThrowIfNull(plan);
        WorldRequest request = plan.Request;
        ImageData image = request.InitImage ?? throw new InvalidOperationException("The SANA-WM plan carries no seed image.");
        WorldCameraPath camera = request.Camera ?? throw new InvalidOperationException("The SANA-WM plan carries no camera input.");
        bool fixedPath = camera.Poses.Count > 0;
        return new SanaWmStreamRequest
        {
            ImageRgb = image.Rgb,
            ImageWidth = image.Width,
            ImageHeight = image.Height,
            Prompt = request.Prompt,
            Intrinsics = Intrinsics(camera),
            CameraToWorld = fixedPath ? camera.Poses : null,
            InitialActions = fixedPath ? null : camera.Script,
            Frames = plan.Unbounded ? null : plan.Frames,
            Guidance = plan.Guidance ?? 1.0,
            Seed = plan.Seed,
            Refine = plan.Refiner,
            Refiner = new SanaWmRefinerSettings
            {
                WindowFrames = EngineKnobs.SanaWmRefinerWindowFrames.Value,
                HistoryLayerStride = EngineKnobs.SanaWmRefinerLayerStride.Value,
                Noise = injection?.RefinerNoise,
                Tap = injection?.RefinerTap,
            },
            FirstLatent = injection?.FirstLatent,
            ChunkNoise = injection?.ChunkNoise,
            Tap = injection?.Tap,
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
