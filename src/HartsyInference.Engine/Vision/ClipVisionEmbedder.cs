using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Requests;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Vision.Clip;

namespace HartsyInference.Engine.Vision;

/// <summary>Pooled CLIP-Vision image embedding: the projected CLS vector of the CLIP-ViT-H/14 image tower — the same
/// encoder IP-Adapter consumes, so an explicit <c>clip_vision</c> checkpoint on disk is reused as-is.</summary>
public sealed class ClipVisionEmbedder : IDisposable
{
    private readonly Dictionary<string, Entry> _cache = new(StringComparer.Ordinal);
    private readonly object _lock = new();

    /// <summary>Encodes <paramref name="image"/> into the contrastive-space embedding vector.</summary>
    public float[] Embed(IBackend backend, string checkpointPath, ImageData image)
    {
        ArgumentNullException.ThrowIfNull(image);
        Entry entry = GetOrLoad(checkpointPath);
        ClipImagePreprocessor preprocessor = new ClipImagePreprocessor(entry.Encoder.Config.ImageSize);
        Tensor pixels = preprocessor.Preprocess(image.Rgb, image.Width, image.Height);
        try
        {
            Tensor embeds = entry.Encoder.EncodeImageEmbeds(backend, pixels);
            try
            {
                return embeds.AsReadOnlySpan<float>().ToArray();
            }
            finally
            {
                embeds.Dispose();
            }
        }
        catch (Exception ex)
        {
            Logs.Error($"[Vision] CLIP-Vision embed failed for '{checkpointPath}'.", ex);
            throw;
        }
        finally
        {
            pixels.Dispose();
        }
    }

    /// <summary>Resolves the CLIP-Vision checkpoint: an explicit path wins, else the conventional side-model folders.</summary>
    public static string? ResolvePath(string? explicitPath) =>
        ModelFileLocator.Find(explicitPath, "clip_vision", "ClipVision", "text_encoders")
        ?? VisionModelPaths.FindCheckpoint(explicitPath, "clip_vision")
        ?? VisionModelPaths.FindCheckpoint(explicitPath, "ClipVision");

    private Entry GetOrLoad(string checkpointPath)
    {
        lock (_lock)
        {
            if (_cache.TryGetValue(checkpointPath, out Entry? cached))
            {
                return cached;
            }
            Logs.Info($"[Vision] Loading CLIP-Vision: {checkpointPath}");
            SafeTensorsLoader loader = new SafeTensorsLoader();
            try
            {
                loader.Load(checkpointPath);
                Dictionary<string, Tensor> weights = loader.GetAllTensors();
                // Some image-encoder files ship under a "vision_model." prefix, others ship rooted.
                string prefix = weights.ContainsKey("vision_model.embeddings.patch_embedding.weight")
                    ? "vision_model"
                    : (weights.ContainsKey("embeddings.patch_embedding.weight") ? "" : "vision_model");
                ClipVisionEncoder encoder = new ClipVisionEncoder(ClipVisionEncoderConfig.ViTH14);
                encoder.LoadWeights(weights, prefix: prefix);
                Entry created = new Entry { Encoder = encoder, Loader = loader };
                _cache[checkpointPath] = created;
                return created;
            }
            catch (Exception ex)
            {
                Logs.Error($"[Vision] CLIP-Vision load failed for '{checkpointPath}'.", ex);
                loader.Dispose();
                throw;
            }
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        lock (_lock)
        {
            foreach (Entry entry in _cache.Values)
            {
                try
                {
                    entry.Loader.Dispose();
                }
                catch (Exception ex)
                {
                    Logs.Error("[Vision] CLIP-Vision loader dispose failed.", ex);
                }
            }
            _cache.Clear();
        }
    }

    /// <summary>A loaded image tower plus the loader owning its weight tensors.</summary>
    private sealed class Entry
    {
        /// <summary>The CLIP vision tower.</summary>
        public required ClipVisionEncoder Encoder { get; init; }

        /// <summary>Owner of the encoder's weight tensors; must outlive the encoder.</summary>
        public required SafeTensorsLoader Loader { get; init; }
    }
}
