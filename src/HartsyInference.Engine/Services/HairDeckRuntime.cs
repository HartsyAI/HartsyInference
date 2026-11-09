using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ThreeD.Models.HairDeck;
using HartsyInference.ThreeD.Models.HairDeck.Preprocessing;
using HartsyInference.Vision.Matting.BiRefNet;
using HartsyInference.Vision.Segmentation.Sam3;

namespace HartsyInference.Engine.Services;

/// <summary>The loaded HairDeck stack: the image model and the two Vision models its preprocessing runs (SAM 3 and BiRefNet-matting).</summary>
internal sealed class HairDeckRuntime : IDisposable
{
    private const string Sam3FileName = "sam3.pt";
    private const string BiRefNetFileName = "model.safetensors";
    private const string DinoFileName = "model.safetensors";

    private readonly Sam3Pipeline _segmenter;
    private readonly BiRefNetMatter _matter;
    private readonly SafeTensorsLoader _matteWeights;
    private int _disposed;

    /// <summary>Serialises generations on this runtime; held for the whole preprocessing and generation of one request.</summary>
    public SemaphoreSlim Gate { get; } = new(1, 1);

    private HairDeckRuntime(Sam3Pipeline segmenter, BiRefNetMatter matter, SafeTensorsLoader matteWeights, HairDeckModel model, IBackend backend)
    {
        _segmenter = segmenter;
        _matter = matter;
        _matteWeights = matteWeights;
        Model = model;
        Preprocessor = new HairDeckPreprocessor(segmenter, matter, backend);
    }

    /// <summary>The image model.</summary>
    public HairDeckModel Model { get; }

    /// <summary>The preprocessing that turns a character image into the model input.</summary>
    public HairDeckPreprocessor Preprocessor { get; }

    /// <summary>Loads the image model from <paramref name="modelRoot"/> and the Vision models from <paramref name="visionRoot"/>.</summary>
    /// <exception cref="FileNotFoundException">A Vision checkpoint is missing; the names of the expected files are in the message.</exception>
    public static HairDeckRuntime Load(IBackend backend, string modelRoot, string visionRoot)
    {
        string sam3Path = Path.Combine(visionRoot, "sam3", Sam3FileName);
        string birefnetPath = Path.Combine(visionRoot, "BiRefNet-matting", BiRefNetFileName);
        string dinoPath = Path.Combine(visionRoot, "Dinov3", DinoFileName);
        RequireFile(sam3Path);
        RequireFile(birefnetPath);
        RequireFile(dinoPath);

        Sam3Pipeline segmenter = Sam3Pipeline.Load(new Sam3Config(), sam3Path);
        SafeTensorsLoader matteWeights = new();
        BiRefNetMatter matter = new();
        try
        {
            matteWeights.Load(birefnetPath);
            matter.LoadWeights(BiRefNetCheckpointConverter.Convert(matteWeights.GetAllTensors()));
            HairDeckModel model = HairDeckModel.Load(backend, modelRoot, dinoPath);
            return new HairDeckRuntime(segmenter, matter, matteWeights, model, backend);
        }
        catch (Exception ex)
        {
            Logs.Error($"Failed to load HairDeck from {modelRoot}", ex);
            matter.Dispose();
            matteWeights.Dispose();
            segmenter.Dispose();
            throw;
        }
    }

    /// <summary>True once <see cref="Dispose"/> has started. A request that won the gate after this must not run.</summary>
    public bool IsDisposed => Volatile.Read(ref _disposed) != 0;

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        // Wait for a generation in progress: the models must not be freed under it. The gate is released again once the
        // models are gone, so a request queued on it wakes, sees IsDisposed and gives up instead of blocking forever.
        Gate.Wait();
        try
        {
            Model.Dispose();
            _matter.Dispose();
            _matteWeights.Dispose();
            _segmenter.Dispose();
        }
        finally
        {
            Gate.Release();
        }
    }

    private static void RequireFile(string path)
    {
        if (!File.Exists(path))
        {
            throw new FileNotFoundException(
                $"HairDeck needs the Vision checkpoint '{path}' (the catalog's SAM 3, BiRefNet-matting and DINOv3 assets). "
                + "Vision checkpoints are read from the Models/Vision folder under the models root.",
                path);
        }
    }
}
