using System.Globalization;
using HartsyInference.Core.Backends;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Requests;
using HartsyInference.ThreeD.Pipelines;
using HartsyInference.ThreeD.Pipelines.Requests;
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
        if (UsesOwnCrop(id))
            return PrepareOwnCrop(id, image, matter, backend, cancel);
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
            _ when UsesHunyuan3DFraming(id) => Hunyuan3DImageProcessor.Process(source.Rgb, source.Alpha, source.Width, source.Height),
            _ => new PreparedImage(source.Rgb, source.Width, source.Height, source.Alpha),
        };
        notes.Add(id switch
        {
            "triposr" => "Foreground cropped, padded to a square at ratio 0.85 and composited on mid-gray.",
            "trellis" => "Foreground cropped to 1.2 times its larger side, resized to 518 px and premultiplied by alpha.",
            _ when UsesHunyuan3DFraming(id) => "Foreground recentred with a 15 % border on white at 512 px.",
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

    /// <summary>TRELLIS.2 and Pixal3D: the image is the pipeline's own preprocessing, run here on its own. <see cref="PreparedMeshImage.Image"/> is the premultiplied square the conditioner resizes and <see cref="PreparedMeshImage.Cutout"/> the image it was cropped from with the alpha the pipeline used, shrunk to 1024 px on the long edge as the pipeline does, so a corrected cutout sent back keeps its alpha and is not matted again.</summary>
    private static PreparedMeshImage PrepareOwnCrop(string id, ImageData image, IForegroundMatter matter, IBackend backend, CancellationToken cancel)
    {
        Trellis2InputImage input = new() { Rgb = image.Rgb, Alpha = image.Alpha, Width = image.Width, Height = image.Height };
        // The pipeline would refuse with its own exception type; this keeps the contract of the other models (NotSupportedException).
        if (!input.HasForegroundAlpha && !matter.IsAvailable)
        {
            throw new NotSupportedException(
                "Preparing an image without alpha needs the BiRefNet matting checkpoint (Vision/BiRefNet-matting/model.safetensors); install it or send an image with alpha.");
        }
        double cropScale = id == MeshModelIds.Pixal3D ? Pixal3DImageTo3DPipeline.CropScale : 1.0;
        Trellis2PreparedCutout prepared = Trellis2ImagePreprocessor.PrepareWithCutout(backend, input, matter, cropScale);
        cancel.ThrowIfCancellationRequested();
        List<string> notes = [];
        if (prepared.Matted)
            notes.Add("The image had no alpha; BiRefNet matted it.");
        if (prepared.Width != image.Width || prepared.Height != image.Height)
            notes.Add($"Shrunk to {prepared.Width} x {prepared.Height}, the {Trellis2ImagePreprocessor.MaxInputEdge} px long edge the pipeline works at.");
        notes.Add($"Foreground square (alpha above 80 %{(cropScale > 1.0 ? $", widened by {cropScale.ToString(CultureInfo.InvariantCulture)}" : string.Empty)}) cropped and premultiplied by alpha; the conditioner resizes it.");
        return new PreparedMeshImage
        {
            Image = new ImageData { Rgb = prepared.Crop.Rgb, Width = prepared.Crop.Width, Height = prepared.Crop.Height },
            Cutout = new ImageData { Rgb = prepared.Rgb, Alpha = prepared.Alpha, Width = prepared.Width, Height = prepared.Height },
            Matted = prepared.Matted,
            Notes = notes,
        };
    }

    /// <summary>Whether <paramref name="id"/> mattes and crops its own input, which <see cref="IMeshPreparationService.PrepareImageAsync"/> reproduces: TRELLIS.2 and Pixal3D.</summary>
    internal static bool UsesOwnCrop(string id) => id == MeshModelIds.Trellis2 || id == MeshModelIds.Pixal3D;

    /// <summary>Whether <paramref name="id"/> takes an image the engine frames: the models that do not matte and frame their own input.</summary>
    internal static bool UsesPreparation(string id) =>
        id == "triposr" || id == "trellis" || UsesHunyuan3DFraming(id) || MeshModelIds.PaintMesh.Contains(id);

    /// <summary>The Hunyuan3D shape builds and Omni, which uses the same <c>ImageProcessorV2</c> (foreground recentred with a 15 % border on white at 512 px).</summary>
    private static bool UsesHunyuan3DFraming(string id) => MeshModelIds.Hunyuan3D.Contains(id) || id == MeshModelIds.Omni;

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
