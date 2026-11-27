using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ThreeD.Models.TrellisV2;
using HartsyInference.Vision.Matting.BiRefNet;

namespace HartsyInference.ThreeD.Pipelines.Stages;

/// <summary>The BiRefNet matter TRELLIS.2 uses for images without alpha, in place of upstream's RMBG-2.0. It looks for its checkpoint and
/// reads it on the first <see cref="Alpha"/> call, so a request whose image already has alpha never pays for it, and a checkpoint installed
/// after the pipeline loaded is still found.</summary>
public sealed class Trellis2LazyMatter : IForegroundMatter, IDisposable
{
    private readonly string _modelsRoot;
    private SafeTensorsLoader? _loader;
    private BiRefNetMatter? _matter;
    private volatile string? _loadedFrom;

    /// <param name="modelsRoot">The models root the checkpoint is looked up under, through <see cref="Trellis2ModelFiles.FindMatterFile"/>.</param>
    public Trellis2LazyMatter(string modelsRoot)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelsRoot);
        _modelsRoot = modelsRoot;
    }

    /// <summary>The checkpoint the resident model was read from; null while nothing is loaded.</summary>
    public string? LoadedFrom => _loadedFrom;

    /// <summary>Whether the checkpoint is installed right now, or already loaded.</summary>
    public bool IsAvailable => _matter is not null || Trellis2ModelFiles.FindMatterFile(_modelsRoot) is not null;

    /// <exception cref="FileNotFoundException">The checkpoint is not installed.</exception>
    public float[] Alpha(IBackend backend, ReadOnlySpan<byte> rgb, int width, int height)
    {
        if (_matter is null)
        {
            string path = Trellis2ModelFiles.FindMatterFile(_modelsRoot)
                ?? throw new FileNotFoundException(
                    "TRELLIS.2 needs the BiRefNet matting checkpoint (Vision/BiRefNet-matting/model.safetensors) for an image without an "
                    + $"alpha channel, and it is not installed under '{_modelsRoot}'. Install it or supply an image with alpha.");
            Logs.Info($"TRELLIS.2: loading the BiRefNet matting stand-in from {path} (upstream uses RMBG-2.0).");
            SafeTensorsLoader loader = new();
            try
            {
                loader.Load(path);
                BiRefNetMatter matter = new();
                matter.LoadWeights(BiRefNetCheckpointConverter.Convert(loader.GetAllTensors()));
                _loader = loader;
                _matter = matter;
                _loadedFrom = path;
            }
            catch
            {
                loader.Dispose();
                throw;
            }
        }

        return _matter.Matte(backend, rgb, width, height);
    }

    public void Dispose()
    {
        _loadedFrom = null;
        _matter?.Dispose();
        _loader?.Dispose();
    }
}
