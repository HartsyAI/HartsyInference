using System.Collections.ObjectModel;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Planning;

/// <summary>Freezes request collection shape while preserving opaque Extra values and leaving large media
/// payloads caller-owned and zero-copy.</summary>
internal sealed class VideoRequestExecutionBinding
{
    private readonly VideoRequest _source;
    private readonly MediaBufferStamp[] _mediaBuffers;

    private VideoRequestExecutionBinding(ModelSpec model, VideoRequest request, VideoRequest source,
        MediaBufferStamp[] mediaBuffers)
    {
        Model = model;
        Request = request;
        _source = source;
        _mediaBuffers = mediaBuffers;
    }

    /// <summary>Small model metadata snapshot used by planning and execution.</summary>
    internal ModelSpec Model { get; }

    /// <summary>Shallow request snapshot with stable collection membership and shared media payloads.</summary>
    internal VideoRequest Request { get; }

    /// <summary>Captures collection membership and media identity without reading or copying media contents.</summary>
    internal static VideoRequestExecutionBinding Create(ModelSpec model, VideoRequest request)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(request);
        if (request.Prompt is null)
        {
            throw new ArgumentException("VideoRequest.Prompt cannot be null.", nameof(request));
        }

        List<MediaBufferStamp> mediaBuffers = [];
        VideoRequest snapshot = SnapshotRequest(request, mediaBuffers);
        return new VideoRequestExecutionBinding(FreezeModel(model), snapshot, request, mediaBuffers.ToArray());
    }

    /// <summary>Rejects request swaps and replaced media buffers, then returns the shallow execution
    /// snapshot.</summary>
    internal static VideoRequest RequireUnchanged(VideoPlan plan, VideoRequest source)
    {
        ArgumentNullException.ThrowIfNull(plan);
        ArgumentNullException.ThrowIfNull(source);
        VideoRequestExecutionBinding binding = plan.RequestBinding ?? throw new ArgumentException(
            "This VideoPlan has no execution binding. Create it through IVideoPlanningService.PlanAsync.",
            nameof(plan));
        if (!ReferenceEquals(plan.SourceRequest, source) || !ReferenceEquals(binding._source, source))
        {
            throw new ArgumentException(
                "A VideoPlan can execute only the same in-memory VideoRequest instance it validated. "
                    + "Re-plan the modified or deserialized request.",
                nameof(source));
        }
        foreach (MediaBufferStamp media in binding._mediaBuffers)
        {
            if (!media.Matches())
            {
                throw new ArgumentException(
                    "A media buffer was replaced after planning. Re-plan before generation; media owner, "
                        + "buffer reference, and length are plan-bound.",
                    nameof(source));
            }
        }
        return binding.Request;
    }

    /// <summary>Freezes public collection surfaces and retains an inaccessible execution snapshot.</summary>
    internal static VideoPlan BindPlan(VideoPlan plan)
    {
        ArgumentNullException.ThrowIfNull(plan);
        VideoPlan frozen = plan with
        {
            Model = FreezeModel(plan.Model),
            Profile = plan.Profile with
            {
                Defaults = plan.Profile.Defaults with { },
                CheckpointMetadata = FreezeStringDictionary(plan.Profile.CheckpointMetadata,
                    nameof(VideoModelProfile.CheckpointMetadata)),
            },
            EffectiveSettings = plan.EffectiveSettings with { },
            Issues = Array.AsReadOnly(plan.Issues.Select(issue => issue with { }).ToArray()),
            ComponentPaths = FreezeStringDictionary(plan.ComponentPaths, nameof(VideoPlan.ComponentPaths)),
            ComponentFormats = FreezeStringDictionary(plan.ComponentFormats, nameof(VideoPlan.ComponentFormats)),
            ArtifactHashes = FreezeStringDictionary(plan.ArtifactHashes, nameof(VideoPlan.ArtifactHashes)),
            ArtifactMetadata = FreezeNestedStringDictionary(plan.ArtifactMetadata),
        };
        frozen = frozen with { ArtifactFileStamps = VideoArtifactFileBinding.Capture(frozen.ComponentPaths) };
        return frozen with { ExecutionPlan = frozen };
    }

    /// <summary>Returns the verified bound plan after rejecting altered public fields. The returned outer plan keeps
    /// its execution marker so first-party construction layers can independently re-verify the same handoff.</summary>
    internal static VideoPlan RequirePlannedState(VideoPlan plan)
    {
        ArgumentNullException.ThrowIfNull(plan);
        VideoPlan execution = plan.ExecutionPlan ?? throw new ArgumentException(
            "This VideoPlan has no immutable execution binding. Create it through IVideoPlanningService.PlanAsync.",
            nameof(plan));
        if (!ReferenceEquals(plan.RequestBinding, execution.RequestBinding)
            || !ReferenceEquals(plan.Model, execution.Model) || !ReferenceEquals(plan.Profile, execution.Profile)
            || !ReferenceEquals(plan.EffectiveSettings, execution.EffectiveSettings)
            || !ReferenceEquals(plan.Issues, execution.Issues)
            || !string.Equals(plan.CacheIdentity, execution.CacheIdentity, StringComparison.Ordinal)
            || !ReferenceEquals(plan.ComponentPaths, execution.ComponentPaths)
            || !ReferenceEquals(plan.ComponentFormats, execution.ComponentFormats)
            || !ReferenceEquals(plan.ArtifactHashes, execution.ArtifactHashes)
            || !ReferenceEquals(plan.ArtifactMetadata, execution.ArtifactMetadata))
        {
            throw new ArgumentException(
                "The VideoPlan was altered after planning. Re-plan and execute the returned plan unchanged.",
                nameof(plan));
        }
        return plan;
    }

    private static ModelSpec FreezeModel(ModelSpec model) => model with
    {
        Catalog = FreezeCatalog(model.Catalog),
        Aux = FreezeStringDictionary(model.Aux, nameof(ModelSpec.Aux)),
    };

    private static CatalogEntry? FreezeCatalog(CatalogEntry? catalog)
    {
        if (catalog is null)
        {
            return null;
        }
        ModelAsset[] assets = new ModelAsset[catalog.Assets.Count];
        for (int i = 0; i < assets.Length; i++)
        {
            ModelAsset asset = catalog.Assets[i]
                ?? throw new ArgumentException($"{nameof(CatalogEntry.Assets)} contains null at index {i}.",
                    nameof(catalog));
            assets[i] = asset with { };
        }
        return catalog with { Assets = Array.AsReadOnly(assets) };
    }

    private static VideoRequest SnapshotRequest(VideoRequest request, List<MediaBufferStamp> mediaBuffers)
    {
        CaptureMedia(request.InitImage, mediaBuffers);
        CaptureMedia(request.VideoEndFrame, mediaBuffers);
        CaptureMedia(request.VideoAudioInput, mediaBuffers);
        CaptureMedia(request.VideoAudioReference, mediaBuffers);
        CaptureMedia(request.VideoDenoiseMask, mediaBuffers);
        CaptureMedia(request.DrivingVideo, mediaBuffers);
        CaptureMedia(request.DrivingPoseVideo, mediaBuffers);
        CaptureMedia(request.DrivingFaceVideo, mediaBuffers);
        CaptureMedia(request.DrivingBackgroundVideo, mediaBuffers);
        CaptureMedia(request.DrivingMaskVideo, mediaBuffers);
        return request with
        {
            ReferenceImages = SnapshotList(request.ReferenceImages, image => CaptureMedia(image, mediaBuffers),
                nameof(VideoRequest.ReferenceImages)),
            ReferenceVideos = SnapshotList(request.ReferenceVideos, reference => CaptureMedia(reference, mediaBuffers),
                nameof(VideoRequest.ReferenceVideos)),
            ReferenceAudios = SnapshotList(request.ReferenceAudios, audio => CaptureMedia(audio, mediaBuffers),
                nameof(VideoRequest.ReferenceAudios)),
            Guides = SnapshotList(request.Guides, guide => CaptureMedia(guide, mediaBuffers),
                nameof(VideoRequest.Guides)),
            AudioDenoiseMask = SnapshotAudioMask(request.AudioDenoiseMask, mediaBuffers),
            Controls = SnapshotList(request.Controls, control => CaptureMedia(control, mediaBuffers),
                nameof(VideoRequest.Controls)),
            Loras = SnapshotLoras(request.Loras),
            Extra = SnapshotExtra(request.Extra, mediaBuffers),
            Settings = SnapshotSettings(request.Settings),
        };
    }

    private static AudioDenoiseMask? SnapshotAudioMask(AudioDenoiseMask? mask,
        List<MediaBufferStamp> mediaBuffers)
    {
        if (mask is null)
        {
            return null;
        }
        IReadOnlyList<float> values = mask.Values
            ?? throw new ArgumentException("AudioDenoiseMask.Values cannot be null.", nameof(mask));
        float[] snapshot = new float[values.Count];
        for (int i = 0; i < snapshot.Length; i++)
        {
            snapshot[i] = values[i];
        }
        CaptureMedia(mask.Source, mediaBuffers);
        return mask with
        {
            Values = Array.AsReadOnly(snapshot),
            Source = mask.Source,
        };
    }

    private static LoraStack? SnapshotLoras(LoraStack? stack)
    {
        if (stack is null)
        {
            return null;
        }
        if (stack.Entries is null)
        {
            throw new ArgumentException("LoraStack.Entries cannot be null.", nameof(VideoRequest.Loras));
        }
        return stack with
        {
            Entries = SnapshotList(stack.Entries, static _ => { }, nameof(LoraStack.Entries))!,
        };
    }

    private static RequestSettings? SnapshotSettings(RequestSettings? settings) =>
        settings is null ? null : settings with
    {
        Set = settings.Set is null ? null : FreezeStringDictionary(settings.Set, nameof(RequestSettings.Set)),
    };

    private static IReadOnlyList<T>? SnapshotList<T>(IReadOnlyList<T>? values, Action<T> capture, string field)
        where T : class
    {
        if (values is null)
        {
            return null;
        }
        int count = values.Count;
        T[] snapshot = new T[count];
        for (int i = 0; i < count; i++)
        {
            T value = values[i] ?? throw new ArgumentException(
                $"Video request field '{field}' contains null at index {i}.", field);
            capture(value);
            snapshot[i] = value;
        }
        if (values.Count != count)
        {
            throw new ArgumentException($"Video request field '{field}' changed while planning copied it.", field);
        }
        return Array.AsReadOnly(snapshot);
    }

    private static IReadOnlyDictionary<string, object> SnapshotExtra(IReadOnlyDictionary<string, object> extra,
        List<MediaBufferStamp> mediaBuffers)
    {
        ArgumentNullException.ThrowIfNull(extra);
        Dictionary<string, object> snapshot = new Dictionary<string, object>(extra.Count, StringComparer.Ordinal);
        foreach ((string key, object value) in extra)
        {
            if (key is null)
            {
                throw new ArgumentException("VideoRequest.Extra contains a null key.", nameof(extra));
            }
            snapshot.Add(key, value);
            CaptureMedia(value, mediaBuffers);
        }
        return new ReadOnlyDictionary<string, object>(snapshot);
    }

    private static void CaptureMedia(object? value, List<MediaBufferStamp> mediaBuffers)
    {
        switch (value)
        {
            case byte[] bytes:
                mediaBuffers.Add(new MediaBufferStamp(bytes, bytes, bytes.LongLength));
                break;
            case ImageData image:
                byte[] pixels = image.Rgb
                    ?? throw new ArgumentException("ImageData.Rgb cannot be null.", nameof(value));
                mediaBuffers.Add(new MediaBufferStamp(image, pixels, pixels.LongLength));
                break;
            case VideoClip video:
                byte[] videoData = video.Data
                    ?? throw new ArgumentException("VideoClip.Data cannot be null.", nameof(value));
                mediaBuffers.Add(new MediaBufferStamp(video, videoData, videoData.LongLength));
                break;
            case AudioClip audio:
                byte[] audioData = audio.Data
                    ?? throw new ArgumentException("AudioClip.Data cannot be null.", nameof(value));
                mediaBuffers.Add(new MediaBufferStamp(audio, audioData, audioData.LongLength));
                break;
            case ReferenceVideo reference:
                if (reference.Video is null)
                {
                    throw new ArgumentException("A reference-video entry has a null Video payload.",
                        nameof(VideoRequest.ReferenceVideos));
                }
                CaptureMedia(reference.Video, mediaBuffers);
                CaptureMedia(reference.Audio, mediaBuffers);
                break;
            case VideoGuide guide:
                CaptureMedia(guide.Image, mediaBuffers);
                CaptureMedia(guide.Video, mediaBuffers);
                CaptureMedia(guide.Audio, mediaBuffers);
                break;
            case VideoDenoiseMask mask:
                CaptureMedia(mask.MaskImage, mediaBuffers);
                CaptureMedia(mask.MaskVideo, mediaBuffers);
                CaptureMedia(mask.SourceImage, mediaBuffers);
                CaptureMedia(mask.SourceVideo, mediaBuffers);
                break;
            case AudioDenoiseMask mask:
                CaptureMedia(mask.Source, mediaBuffers);
                break;
            case VideoControl control:
                if (control.Video is null)
                {
                    throw new ArgumentException("A control entry has a null Video payload.",
                        nameof(VideoRequest.Controls));
                }
                CaptureMedia(control.Video, mediaBuffers);
                CaptureMedia(control.VisibilityMask, mediaBuffers);
                CaptureMedia(control.MaskedSource, mediaBuffers);
                break;
        }
    }

    private static IReadOnlyDictionary<string, string> FreezeStringDictionary(
        IReadOnlyDictionary<string, string> values, string field)
    {
        Dictionary<string, string> snapshot = new Dictionary<string, string>(values.Count, StringComparer.Ordinal);
        foreach ((string key, string value) in values)
        {
            if (key is null || value is null)
            {
                throw new ArgumentException($"Video request/model dictionary '{field}' contains a null key or value.",
                    field);
            }
            snapshot.Add(key, value);
        }
        return new ReadOnlyDictionary<string, string>(snapshot);
    }

    private static IReadOnlyDictionary<string, IReadOnlyDictionary<string, string>> FreezeNestedStringDictionary(
        IReadOnlyDictionary<string, IReadOnlyDictionary<string, string>> values)
    {
        Dictionary<string, IReadOnlyDictionary<string, string>> snapshot =
            new Dictionary<string, IReadOnlyDictionary<string, string>>(values.Count, StringComparer.Ordinal);
        foreach ((string key, IReadOnlyDictionary<string, string> value) in values)
        {
            snapshot.Add(key, FreezeStringDictionary(value, $"{nameof(VideoPlan.ArtifactMetadata)}.{key}"));
        }
        return new ReadOnlyDictionary<string, IReadOnlyDictionary<string, string>>(snapshot);
    }

    private readonly record struct MediaBufferStamp(object Owner, byte[] Buffer, long Length)
    {
        internal bool Matches() => Owner switch
        {
            ImageData image => ReferenceEquals(image.Rgb, Buffer) && image.Rgb.LongLength == Length,
            VideoClip video => ReferenceEquals(video.Data, Buffer) && video.Data.LongLength == Length,
            AudioClip audio => ReferenceEquals(audio.Data, Buffer) && audio.Data.LongLength == Length,
            byte[] bytes => ReferenceEquals(bytes, Buffer) && bytes.LongLength == Length,
            _ => false,
        };
    }
}
