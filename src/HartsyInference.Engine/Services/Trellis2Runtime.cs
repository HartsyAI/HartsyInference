using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ThreeD.Pipelines;
using HartsyInference.ThreeD.Pipelines.Stages;
using HartsyInference.Vision.Matting.BiRefNet;

namespace HartsyInference.Engine.Services;

/// <summary>A loaded TRELLIS.2 pipeline with the matter it uses for images that carry no alpha. The matter reads its checkpoint on first
/// use, so a request whose image already has alpha never pays for it.</summary>
internal sealed class Trellis2Runtime : IDisposable
{
    /// <summary>Models-root relative location of the BiRefNet-matting checkpoint the catalog downloads.</summary>
    internal static readonly string[] MatterRelativePath = ["Vision", "BiRefNet-matting", "model.safetensors"];

    private readonly LazyMatter? _matter;
    private int _disposed;

    private Trellis2Runtime(Trellis2ImageTo3DPipeline pipeline, LazyMatter? matter)
    {
        Pipeline = pipeline;
        _matter = matter;
    }

    /// <summary>The pipeline.</summary>
    public Trellis2ImageTo3DPipeline Pipeline { get; }

    /// <summary>Serialises generations on this runtime; the stages share one set of resident weights.</summary>
    public SemaphoreSlim Gate { get; } = new(1, 1);

    /// <summary>Whether a matting checkpoint is installed under <paramref name="modelsRoot"/>.</summary>
    public static bool MatterAvailable(string modelsRoot) => File.Exists(MatterPath(modelsRoot));

    /// <summary>Loads the pipeline from <paramref name="localPath"/>; the matting checkpoint is looked up under <paramref name="modelsRoot"/>.</summary>
    public static Trellis2Runtime Load(IBackend backend, string localPath, string modelsRoot)
    {
        LazyMatter? matter = MatterAvailable(modelsRoot) ? new LazyMatter(MatterPath(modelsRoot)) : null;
        try
        {
            return new Trellis2Runtime(Trellis2ImageTo3DPipeline.LoadFromPath(backend, localPath, matter), matter);
        }
        catch
        {
            matter?.Dispose();
            throw;
        }
    }

    /// <summary>True once <see cref="Dispose"/> has started.</summary>
    public bool IsDisposed => Volatile.Read(ref _disposed) != 0;

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        // A generation in progress holds the gate; the weights must not go under it.
        Gate.Wait();
        try
        {
            Pipeline.Dispose();
            _matter?.Dispose();
        }
        finally
        {
            Gate.Release();
        }
    }

    private static string MatterPath(string modelsRoot) => Path.Combine([modelsRoot, .. MatterRelativePath]);

    private sealed class LazyMatter : IForegroundMatter, IDisposable
    {
        private readonly string _path;
        private SafeTensorsLoader? _loader;
        private BiRefNetMatter? _matter;

        internal LazyMatter(string path) => _path = path;

        public float[] Alpha(IBackend backend, ReadOnlySpan<byte> rgb, int width, int height)
        {
            if (_matter is null)
            {
                Logs.Info($"TRELLIS.2: loading the BiRefNet matting stand-in from {_path} (upstream uses RMBG-2.0).");
                SafeTensorsLoader loader = new();
                try
                {
                    loader.Load(_path);
                    BiRefNetMatter matter = new();
                    matter.LoadWeights(BiRefNetCheckpointConverter.Convert(loader.GetAllTensors()));
                    _loader = loader;
                    _matter = matter;
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
            _matter?.Dispose();
            _loader?.Dispose();
        }
    }
}
