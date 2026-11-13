using HartsyInference.Core.Backends;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Requests;
using HartsyInference.ThreeD.Pipelines.Stages;
using HartsyInference.ThreeD.Preprocessing;

namespace HartsyInference.Engine.Services;

/// <summary>Prepares the input image of a 3D model: the matte for an image without alpha, then the model's own framing and compositing. The matte is the BiRefNet checkpoint TRELLIS.2 and HairDeck also use.</summary>
public sealed class MeshPreparationService : IMeshPreparationService, IDisposable
{
    private readonly InferenceEngine _engine;
    private readonly Func<ISharedMatter> _createMatter;
    private readonly object _gate = new();
    private readonly SerializedMatter _shared;
    private ISharedMatter? _matter;

    internal MeshPreparationService(InferenceEngine engine)
        : this(engine, static () => new SharedBiRefNetMatter(new Trellis2LazyMatter(RepoPaths.ModelsRoot())))
    {
    }

    /// <param name="createMatter">Builds the matting model on first use, and again after it was released; a test stands in for the checkpoint.</param>
    internal MeshPreparationService(InferenceEngine engine, Func<ISharedMatter> createMatter)
    {
        _engine = engine;
        _createMatter = createMatter;
        _shared = new SerializedMatter(this);
    }

    /// <summary>The one matting model: the pipelines that matte their own input (TRELLIS.2, Pixal3D) take it too, so the checkpoint is never resident twice. It loads on first use, runs one image at a time, and is released by <see cref="ReleaseMatter"/>.</summary>
    internal IForegroundMatter SharedMatter => _shared;

    /// <summary>The checkpoint the resident matting model was read from; null while none is resident. Does not wait for a matte in progress.</summary>
    internal string? MatterLoadedFrom => Volatile.Read(ref _matter)?.LoadedFrom;

    /// <summary>Releases the matting model, waiting for a matte in progress, and says whether one was resident. The next matte loads it again.</summary>
    internal bool ReleaseMatter()
    {
        lock (_gate)
        {
            ISharedMatter? matter = _matter;
            _matter = null;
            if (matter is null)
                return false;
            bool resident = matter.LoadedFrom is not null;
            matter.Dispose();
            return resident;
        }
    }

    /// <inheritdoc/>
    public Task<PreparedMeshImage> PrepareImageAsync(ModelSpec spec, ImageData image, CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(spec);
        ArgumentNullException.ThrowIfNull(image);
        string id = MeshService.IdOf(spec);
        if (MeshModelProfiles.Find(id) is null)
            throw new NotSupportedException($"3D model '{id}' has no wired pipeline; supported: {string.Join(", ", MeshModelIds.Described)}.");
        return Task.Run(() => Prepare(id, image, Matter(), _engine.Backend, cancel), cancel);
    }

    /// <summary>Prepares for a generation on this engine.</summary>
    internal PreparedMeshImage Prepare(string id, ImageData image, CancellationToken cancel) => Prepare(id, image, Matter(), _engine.Backend, cancel);

    /// <summary>The preparation itself, with the matter and backend supplied so a test can stand in for them.</summary>
    internal static PreparedMeshImage Prepare(string id, ImageData image, IForegroundMatter matter, IBackend backend, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        if (!UsesPreparation(id))
        {
            string reason = MeshModelProfiles.Find(id)?.Input is MeshInputKind.SceneMesh
                ? $"'{id}' takes no image."
                : $"'{id}' mattes and frames its own input; the image is used as given.";
            return new PreparedMeshImage { Image = image, Notes = [reason] };
        }

        List<string> notes = [];
        ImageData source = image;
        bool matted = false;
        if (!HasUsableAlpha(image))
        {
            if (!matter.IsAvailable)
            {
                throw new NotSupportedException(
                    "Preparing an image without alpha needs the BiRefNet matting checkpoint (Vision/BiRefNet-matting/model.safetensors); install it or send an image with alpha.");
            }
            float[] coverage = matter.Alpha(backend, image.Rgb, image.Width, image.Height);
            cancel.ThrowIfCancellationRequested();
            source = image with { Alpha = [.. coverage.Select(a => (byte)Math.Clamp((int)MathF.Round(a * 255f), 0, 255))] };
            matted = true;
            notes.Add("The image had no alpha; BiRefNet matted it.");
        }

        PreparedImage prepared = id switch
        {
            "triposr" => TripoSrImageProcessor.Process(source.Rgb, source.Alpha, source.Width, source.Height),
            "trellis" => TrellisImageProcessor.Process(source.Rgb, source.Alpha, source.Width, source.Height),
            _ when MeshModelIds.Hunyuan3D.Contains(id) => Hunyuan3DImageProcessor.Process(source.Rgb, source.Alpha, source.Width, source.Height),
            _ => new PreparedImage(source.Rgb, source.Width, source.Height, source.Alpha),
        };
        notes.Add(id switch
        {
            "triposr" => "Foreground cropped, padded to a square at ratio 0.85 and composited on mid-gray.",
            "trellis" => "Foreground cropped to 1.2 times its larger side, resized to 518 px and premultiplied by alpha.",
            _ when MeshModelIds.Hunyuan3D.Contains(id) => "Foreground recentred with a 15 % border on white at 512 px.",
            _ => "The cutout is composited on white by the model.",
        });
        return new PreparedMeshImage
        {
            Image = new ImageData { Rgb = prepared.Rgb, Width = prepared.Width, Height = prepared.Height, Alpha = prepared.Alpha },
            Cutout = source,
            Matted = matted,
            Notes = notes,
        };
    }

    /// <summary>Whether <paramref name="id"/> takes an image the engine frames: the models that do not matte and frame their own input.</summary>
    internal static bool UsesPreparation(string id) =>
        id == "triposr" || id == "trellis" || MeshModelIds.Hunyuan3D.Contains(id) || MeshModelIds.PaintMesh.Contains(id);

    /// <summary>Whether the image carries a cutout to use: an alpha plane with some transparency. An RGBA image whose alpha is entirely opaque (every ordinary RGBA export of a photo) has no foreground to find, as in upstream TRELLIS (<c>np.all(alpha == 255)</c>), so it is matted; a plane that is nearly but not entirely opaque is taken as a cutout.</summary>
    internal static bool HasUsableAlpha(ImageData image) => image.HasAlpha && image.Alpha!.Any(a => a != 255);

    private IForegroundMatter Matter() => _shared;

    /// <summary>Releases the matting model, waiting for a matte in progress.</summary>
    public void Dispose() => ReleaseMatter();

    /// <summary>The service's one matting model behind the service's lock: it loads once, runs one image at a time, and cannot be released under a run.</summary>
    private sealed class SerializedMatter(MeshPreparationService owner) : IForegroundMatter
    {
        public bool IsAvailable
        {
            get
            {
                lock (owner._gate)
                    return (owner._matter ??= owner._createMatter()).IsAvailable;
            }
        }

        public float[] Alpha(IBackend backend, ReadOnlySpan<byte> rgb, int width, int height)
        {
            lock (owner._gate)
                return (owner._matter ??= owner._createMatter()).Alpha(backend, rgb, width, height);
        }
    }
}
