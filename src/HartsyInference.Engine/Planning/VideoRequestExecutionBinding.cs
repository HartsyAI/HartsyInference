using System.Buffers.Binary;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.ObjectModel;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Planning;

/// <summary>Creates the immutable request/model graph a <see cref="VideoPlan"/> validates and executes, plus a
/// structural binding to the caller-owned request. No JSON serialization is involved: binary media is copied once
/// and fed directly to SHA-256, while <see cref="VideoRequest.Extra"/> retains its supported runtime value types.</summary>
internal sealed record VideoRequestExecutionBinding
{
    private static readonly ConcurrentDictionary<Type, PropertyInfo[]> FingerprintProperties = new();

    private static readonly HashSet<Type> SupportedRecordTypes =
    [
        typeof(VideoRequest),
        typeof(ImageData),
        typeof(VideoClip),
        typeof(AudioClip),
        typeof(ReferenceVideo),
        typeof(VideoGuide),
        typeof(VideoDenoiseMask),
        typeof(AudioDenoiseMask),
        typeof(VideoControl),
        typeof(LoraStack),
        typeof(LoraEntry),
        typeof(ComponentOverrides),
        typeof(VramOverrides),
        typeof(RequestSettings),
    ];

    internal required ModelSpec Model { get; init; }

    internal required VideoRequest Request { get; init; }

    internal required string Fingerprint { get; init; }

    /// <summary>Freezes before profile resolution so every validation read observes the same graph execution will.</summary>
    internal static VideoRequestExecutionBinding Create(ModelSpec model, VideoRequest request)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(request);
        try
        {
            VideoRequest frozen = FreezeRequest(request);
            return new VideoRequestExecutionBinding
            {
                Model = FreezeModel(model),
                Request = frozen,
                Fingerprint = ComputeFingerprint(frozen),
            };
        }
        catch (Exception error) when (error is InvalidOperationException or IndexOutOfRangeException
            or NullReferenceException)
        {
            throw new ArgumentException(
                "The video request/model graph changed while planning copied it, or contains a null required payload. "
                + "Retry with a stable request graph.", nameof(request), error);
        }
    }

    /// <summary>Verifies identity and every nested mutable value, then returns only the inaccessible frozen graph.</summary>
    internal static VideoRequest RequireUnchanged(VideoPlan plan, VideoRequest source)
    {
        ArgumentNullException.ThrowIfNull(plan);
        ArgumentNullException.ThrowIfNull(source);
        if (!ReferenceEquals(plan.SourceRequest, source))
        {
            throw new ArgumentException(
                "A VideoPlan can execute only the same in-memory VideoRequest instance it validated. Re-plan the modified or deserialized request.",
                nameof(source));
        }
        if (plan.ExecutionRequest is null || string.IsNullOrWhiteSpace(plan.SourceRequestFingerprint))
        {
            throw new ArgumentException(
                "This VideoPlan has no immutable execution binding. Create it through IVideoPlanningService.PlanAsync.",
                nameof(plan));
        }

        string current;
        try
        {
            current = ComputeFingerprint(source);
        }
        catch (Exception error) when (error is ArgumentException or InvalidOperationException)
        {
            throw new ArgumentException(
                "The VideoRequest changed while its execution binding was being verified. Re-plan the request.",
                nameof(source), error);
        }
        if (!string.Equals(current, plan.SourceRequestFingerprint, StringComparison.Ordinal))
        {
            throw new ArgumentException(
                "The VideoRequest was mutated after planning. Re-plan before generation; nested media, lists, settings, and Extra values are plan-bound.",
                nameof(source));
        }
        return plan.ExecutionRequest;
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

    /// <summary>Returns the inaccessible plan snapshot after rejecting altered public plan fields.</summary>
    internal static VideoPlan RequirePlannedState(VideoPlan plan)
    {
        ArgumentNullException.ThrowIfNull(plan);
        VideoPlan execution = plan.ExecutionPlan ?? throw new ArgumentException(
            "This VideoPlan has no immutable execution binding. Create it through IVideoPlanningService.PlanAsync.",
            nameof(plan));
        if (!ReferenceEquals(plan.Model, execution.Model) || !ReferenceEquals(plan.Profile, execution.Profile)
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
        return execution;
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
            assets[i] = catalog.Assets[i] with { };
        }
        return catalog with { Assets = Array.AsReadOnly(assets) };
    }

    private static VideoRequest FreezeRequest(VideoRequest request)
    {
        if (request.Prompt is null)
        {
            throw new ArgumentException("VideoRequest.Prompt cannot be null.", nameof(VideoRequest.Prompt));
        }
        return request with
        {
            InitImage = FreezeImage(request.InitImage),
            VideoEndFrame = FreezeImage(request.VideoEndFrame),
            VideoAudioInput = FreezeAudio(request.VideoAudioInput),
            VideoAudioReference = FreezeAudio(request.VideoAudioReference),
            ReferenceImages = FreezeList(request.ReferenceImages, image => FreezeImage(image)!,
                nameof(VideoRequest.ReferenceImages)),
            ReferenceVideos = FreezeList(request.ReferenceVideos, FreezeReferenceVideo,
                nameof(VideoRequest.ReferenceVideos)),
            ReferenceAudios = FreezeList(request.ReferenceAudios, audio => FreezeAudio(audio)!,
                nameof(VideoRequest.ReferenceAudios)),
            Guides = FreezeList(request.Guides, FreezeGuide, nameof(VideoRequest.Guides)),
            VideoDenoiseMask = FreezeVideoMask(request.VideoDenoiseMask),
            AudioDenoiseMask = FreezeAudioMask(request.AudioDenoiseMask),
            Controls = FreezeList(request.Controls, FreezeControl, nameof(VideoRequest.Controls)),
            DrivingVideo = FreezeVideo(request.DrivingVideo),
            DrivingPoseVideo = FreezeVideo(request.DrivingPoseVideo),
            DrivingFaceVideo = FreezeVideo(request.DrivingFaceVideo),
            DrivingBackgroundVideo = FreezeVideo(request.DrivingBackgroundVideo),
            DrivingMaskVideo = FreezeVideo(request.DrivingMaskVideo),
            Components = request.Components is null ? null : request.Components with { },
            Loras = FreezeLoras(request.Loras),
            Extra = FreezeExtra(request.Extra),
            Vram = request.Vram is null ? null : request.Vram with { },
            Settings = FreezeSettings(request.Settings),
        };
    }

    private static ImageData? FreezeImage(ImageData? image) => image is null ? null : image with
    {
        Rgb = (byte[])image.Rgb.Clone(),
    };

    private static VideoClip? FreezeVideo(VideoClip? clip) => clip is null ? null : clip with
    {
        Data = (byte[])clip.Data.Clone(),
    };

    private static AudioClip? FreezeAudio(AudioClip? clip) => clip is null ? null : clip with
    {
        Data = (byte[])clip.Data.Clone(),
    };

    private static ReferenceVideo FreezeReferenceVideo(ReferenceVideo reference)
    {
        if (reference.Video is null)
        {
            throw new ArgumentException("A reference-video entry has a null Video payload.",
                nameof(VideoRequest.ReferenceVideos));
        }
        return reference with
        {
            Video = FreezeVideo(reference.Video)!,
            Audio = FreezeAudio(reference.Audio),
        };
    }

    private static VideoGuide FreezeGuide(VideoGuide guide) => guide with
    {
        Image = FreezeImage(guide.Image),
        Video = FreezeVideo(guide.Video),
        Audio = FreezeAudio(guide.Audio),
    };

    private static VideoDenoiseMask? FreezeVideoMask(VideoDenoiseMask? mask) => mask is null ? null : mask with
    {
        MaskImage = FreezeImage(mask.MaskImage),
        MaskVideo = FreezeVideo(mask.MaskVideo),
        SourceImage = FreezeImage(mask.SourceImage),
        SourceVideo = FreezeVideo(mask.SourceVideo),
    };

    private static AudioDenoiseMask? FreezeAudioMask(AudioDenoiseMask? mask)
    {
        if (mask is null)
        {
            return null;
        }
        if (mask.Values is null)
        {
            throw new ArgumentException("AudioDenoiseMask.Values cannot be null.",
                nameof(VideoRequest.AudioDenoiseMask));
        }
        return mask with
        {
            Values = Array.AsReadOnly(mask.Values.ToArray()),
            Source = FreezeAudio(mask.Source),
        };
    }

    private static VideoControl FreezeControl(VideoControl control)
    {
        if (control.Video is null)
        {
            throw new ArgumentException("A control entry has a null Video payload.", nameof(VideoRequest.Controls));
        }
        return control with
        {
            Video = FreezeVideo(control.Video)!,
            VisibilityMask = FreezeVideo(control.VisibilityMask),
            MaskedSource = FreezeVideo(control.MaskedSource),
        };
    }

    private static LoraStack? FreezeLoras(LoraStack? stack)
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
            Entries = FreezeList(stack.Entries, entry => entry with { }, nameof(LoraStack.Entries))!,
        };
    }

    private static RequestSettings? FreezeSettings(RequestSettings? settings) => settings is null ? null : settings with
    {
        Set = settings.Set is null ? null : FreezeStringDictionary(settings.Set, nameof(RequestSettings.Set)),
    };

    private static IReadOnlyList<TOutput>? FreezeList<TInput, TOutput>(
        IReadOnlyList<TInput>? values, Func<TInput, TOutput> clone, string field)
    {
        if (values is null)
        {
            return null;
        }
        int count;
        try
        {
            count = values.Count;
        }
        catch (Exception error) when (error is InvalidOperationException or IndexOutOfRangeException)
        {
            throw new ArgumentException($"Video request field '{field}' changed while planning copied it.",
                field, error);
        }
        TOutput[] copy = new TOutput[count];
        for (int i = 0; i < copy.Length; i++)
        {
            TInput value;
            try
            {
                value = values[i];
            }
            catch (Exception error) when (error is ArgumentOutOfRangeException or IndexOutOfRangeException
                or InvalidOperationException)
            {
                throw new ArgumentException(
                    $"Video request field '{field}' changed while planning copied index {i}.", field, error);
            }
            if (value is null)
            {
                throw new ArgumentException($"Video request field '{field}' contains null at index {i}.", field);
            }
            copy[i] = clone(value);
        }
        return Array.AsReadOnly(copy);
    }

    private static IReadOnlyDictionary<string, string> FreezeStringDictionary(
        IReadOnlyDictionary<string, string> values, string field)
    {
        Dictionary<string, string> copy = new Dictionary<string, string>(values.Count, StringComparer.Ordinal);
        try
        {
            foreach ((string key, string value) in values)
            {
                if (key is null || value is null)
                {
                    throw new ArgumentException($"Video request/model dictionary '{field}' contains a null key or value.",
                        field);
                }
                copy.Add(key, value);
            }
        }
        catch (InvalidOperationException error)
        {
            throw new ArgumentException(
                $"Video request/model dictionary '{field}' changed while planning copied it.", field, error);
        }
        return new ReadOnlyDictionary<string, string>(copy);
    }

    private static IReadOnlyDictionary<string, IReadOnlyDictionary<string, string>> FreezeNestedStringDictionary(
        IReadOnlyDictionary<string, IReadOnlyDictionary<string, string>> values)
    {
        Dictionary<string, IReadOnlyDictionary<string, string>> copy =
            new Dictionary<string, IReadOnlyDictionary<string, string>>(values.Count, StringComparer.Ordinal);
        foreach ((string key, IReadOnlyDictionary<string, string> value) in values)
        {
            copy.Add(key, FreezeStringDictionary(value, $"{nameof(VideoPlan.ArtifactMetadata)}.{key}"));
        }
        return new ReadOnlyDictionary<string, IReadOnlyDictionary<string, string>>(copy);
    }

    private static IReadOnlyDictionary<string, object> FreezeExtra(IReadOnlyDictionary<string, object> extra)
    {
        Dictionary<string, object> copy = new Dictionary<string, object>(extra.Count, StringComparer.Ordinal);
        foreach ((string key, object value) in extra)
        {
            copy.Add(key, FreezeExtraValue(key, value));
        }
        return new ReadOnlyDictionary<string, object>(copy);
    }

    private static object FreezeExtraValue(string key, object? value)
    {
        if (value is null)
        {
            return null!;
        }
        Type type = value.GetType();
        if (type.IsEnum || type.IsPrimitive || value is string or decimal or Guid or DateTime
            or DateTimeOffset or DateOnly or TimeOnly or TimeSpan)
        {
            return value;
        }
        return value switch
        {
            byte[] bytes => (byte[])bytes.Clone(),
            JsonElement json => json.Clone(),
            ImageData image => FreezeImage(image)!,
            VideoClip video => FreezeVideo(video)!,
            AudioClip audio => FreezeAudio(audio)!,
            ReferenceVideo reference => FreezeReferenceVideo(reference),
            VideoGuide guide => FreezeGuide(guide),
            VideoDenoiseMask mask => FreezeVideoMask(mask)!,
            AudioDenoiseMask mask => FreezeAudioMask(mask)!,
            VideoControl control => FreezeControl(control),
            LoraEntry lora => lora with { },
            LoraStack stack => FreezeLoras(stack)!,
            ComponentOverrides components => components with { },
            VramOverrides vram => vram with { },
            RequestSettings settings => FreezeSettings(settings)!,
            _ => throw new ArgumentException(
                $"VideoRequest.Extra['{key}'] has unsupported mutable type '{type.FullName}'. "
                + "Use strings, primitive values, JsonElement, or an Engine video request/media DTO.",
                nameof(VideoRequest.Extra)),
        };
    }

    private static string ComputeFingerprint(VideoRequest request)
    {
        using FingerprintWriter writer = new FingerprintWriter();
        AppendValue(writer, request);
        return writer.Finish();
    }

    private static void AppendValue(FingerprintWriter writer, object? value)
    {
        if (value is null)
        {
            writer.Tag("null");
            return;
        }

        Type type = value.GetType();
        if (type.IsEnum)
        {
            writer.Tag("enum");
            writer.String(type.AssemblyQualifiedName ?? type.FullName ?? type.Name);
            AppendValue(writer, Convert.ChangeType(value, Enum.GetUnderlyingType(type),
                System.Globalization.CultureInfo.InvariantCulture));
            return;
        }
        switch (value)
        {
            case string text: writer.Tag("string"); writer.String(text); return;
            case bool item: writer.Tag("bool"); writer.Byte(item ? (byte)1 : (byte)0); return;
            case byte item: writer.Tag("u8"); writer.Byte(item); return;
            case sbyte item: writer.Tag("i8"); writer.Byte(unchecked((byte)item)); return;
            case short item: writer.Tag("i16"); writer.Int64(item); return;
            case ushort item: writer.Tag("u16"); writer.UInt64(item); return;
            case int item: writer.Tag("i32"); writer.Int64(item); return;
            case uint item: writer.Tag("u32"); writer.UInt64(item); return;
            case long item: writer.Tag("i64"); writer.Int64(item); return;
            case ulong item: writer.Tag("u64"); writer.UInt64(item); return;
            case nint item: writer.Tag("nint"); writer.Int64(item); return;
            case nuint item: writer.Tag("nuint"); writer.UInt64(item); return;
            case char item: writer.Tag("char"); writer.UInt64(item); return;
            case Half item: writer.Tag("half"); writer.Int64(BitConverter.HalfToInt16Bits(item)); return;
            case float item: writer.Tag("f32"); writer.Int64(BitConverter.SingleToInt32Bits(item)); return;
            case double item: writer.Tag("f64"); writer.Int64(BitConverter.DoubleToInt64Bits(item)); return;
            case decimal item:
                writer.Tag("decimal");
                foreach (int part in decimal.GetBits(item)) writer.Int64(part);
                return;
            case Guid item:
                writer.Tag("guid");
                Span<byte> guid = stackalloc byte[16];
                item.TryWriteBytes(guid);
                writer.Bytes(guid);
                return;
            case DateTime item: writer.Tag("datetime"); writer.Int64(item.ToBinary()); return;
            case DateTimeOffset item:
                writer.Tag("datetimeoffset"); writer.Int64(item.Ticks); writer.Int64(item.Offset.Ticks); return;
            case DateOnly item: writer.Tag("dateonly"); writer.Int64(item.DayNumber); return;
            case TimeOnly item: writer.Tag("timeonly"); writer.Int64(item.Ticks); return;
            case TimeSpan item: writer.Tag("timespan"); writer.Int64(item.Ticks); return;
            case byte[] bytes:
                writer.Tag("bytes"); writer.Int64(bytes.LongLength); writer.Bytes(bytes); return;
            case JsonElement json:
                writer.Tag("json"); writer.Int64((int)json.ValueKind);
                if (json.ValueKind != JsonValueKind.Undefined) writer.String(json.GetRawText());
                return;
            case IReadOnlyDictionary<string, string> strings:
                AppendDictionary(writer, strings.Select(pair =>
                    new KeyValuePair<string, object?>(pair.Key, pair.Value)));
                return;
            case IReadOnlyDictionary<string, object> objects:
                AppendDictionary(writer, objects.Select(pair =>
                    new KeyValuePair<string, object?>(pair.Key, pair.Value)));
                return;
        }

        if (SupportedRecordTypes.Contains(type))
        {
            writer.Tag("record");
            writer.String(type.FullName ?? type.Name);
            PropertyInfo[] properties = FingerprintProperties.GetOrAdd(type, static recordType =>
                recordType.GetProperties(BindingFlags.Instance | BindingFlags.Public)
                    .Where(property => property.GetMethod is not null && property.SetMethod is not null
                        && property.GetIndexParameters().Length == 0)
                    .OrderBy(property => property.Name, StringComparer.Ordinal)
                    .ToArray());
            writer.Int64(properties.Length);
            foreach (PropertyInfo property in properties)
            {
                writer.String(property.Name);
                AppendValue(writer, property.GetValue(value));
            }
            return;
        }

        if (value is IEnumerable sequence)
        {
            writer.Tag("list");
            List<object?> items = [];
            foreach (object? item in sequence) items.Add(item);
            writer.Int64(items.Count);
            foreach (object? item in items) AppendValue(writer, item);
            return;
        }

        throw new ArgumentException(
            $"A VideoRequest value has unsupported mutable type '{type.FullName}'.", nameof(value));
    }

    private static void AppendDictionary(FingerprintWriter writer,
        IEnumerable<KeyValuePair<string, object?>> values)
    {
        KeyValuePair<string, object?>[] ordered = values.OrderBy(pair => pair.Key, StringComparer.Ordinal).ToArray();
        writer.Tag("dictionary");
        writer.Int64(ordered.Length);
        foreach ((string key, object? value) in ordered)
        {
            writer.String(key);
            AppendValue(writer, value);
        }
    }

    private sealed class FingerprintWriter : IDisposable
    {
        private readonly IncrementalHash _hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);

        internal void Tag(string value) => String(value);

        internal void Byte(byte value)
        {
            Span<byte> bytes = stackalloc byte[1];
            bytes[0] = value;
            _hash.AppendData(bytes);
        }

        internal void Int64(long value)
        {
            Span<byte> bytes = stackalloc byte[8];
            BinaryPrimitives.WriteInt64LittleEndian(bytes, value);
            _hash.AppendData(bytes);
        }

        internal void UInt64(ulong value)
        {
            Span<byte> bytes = stackalloc byte[8];
            BinaryPrimitives.WriteUInt64LittleEndian(bytes, value);
            _hash.AppendData(bytes);
        }

        internal void String(string value)
        {
            int count = Encoding.UTF8.GetByteCount(value);
            Int64(count);
            if (count == 0) return;
            byte[] bytes = Encoding.UTF8.GetBytes(value);
            _hash.AppendData(bytes);
        }

        internal void Bytes(ReadOnlySpan<byte> bytes) => _hash.AppendData(bytes);

        internal string Finish() => Convert.ToHexString(_hash.GetHashAndReset());

        public void Dispose() => _hash.Dispose();
    }
}
