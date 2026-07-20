using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Requests;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Vision.Detection.GroundingDino;

namespace HartsyInference.Engine.Vision;

/// <summary>Open-vocabulary, text-prompted detection with the pure-C# Grounding DINO (<c>grounding-dino-tiny</c>):
/// DETR-style preprocessing → BERT tokenization → Swin backbone + cross-modality encoder + two-stage decoder.</summary>
public sealed class GroundingDinoObjectDetector : IDisposable
{
    private const int ShortestEdge = 800;
    private const int LongestEdge = 1333;

    private readonly Dictionary<string, Entry> _cache = new(StringComparer.Ordinal);
    private readonly object _lock = new();
    private IBackend? _boundBackend;

    /// <summary>Detects <paramref name="query"/> in <paramref name="image"/>, returning boxes in source-image pixels.</summary>
    public IReadOnlyList<Detection> Detect(IBackend backend, string checkpointPath, string vocabPath, ImageData image, string query, float threshold)
    {
        ArgumentNullException.ThrowIfNull(image);
        string subject = (query ?? "").Trim();
        string phrase = subject;
        if (phrase.Length == 0)
        {
            throw new InvalidOperationException("Grounding DINO needs a text query; supply VisionRequest.Prompt.");
        }
        // GDINO convention: the phrase is period-terminated (the period is a query separator token).
        if (!phrase.EndsWith('.'))
        {
            phrase += " .";
        }
        Entry entry = GetOrLoad(backend, checkpointPath, vocabPath);
        int[] ids = entry.Tokenizer.EncodeWithSpecial(phrase).ToArray();
        Tensor pixels = BuildPixels(image);
        GroundingDinoDetector.Output output;
        try
        {
            output = entry.Model.Forward(backend, pixels, ids);
        }
        catch (Exception ex)
        {
            Logs.Error($"[Vision] Grounding DINO forward failed for query '{phrase}'.", ex);
            throw;
        }
        finally
        {
            pixels.Dispose();
        }
        try
        {
            float textThreshold = Math.Min(0.3f, threshold);
            List<GroundingDinoDetection> raw = GroundingDinoPipeline.PostProcess(
                output.Logits, output.PredBoxes, ids, entry.Vocab, image.Height, image.Width, threshold, textThreshold);
            List<Detection> results = new List<Detection>(raw.Count);
            foreach (GroundingDinoDetection d in raw)
            {
                results.Add(new Detection
                {
                    Label = string.IsNullOrWhiteSpace(d.Label) ? subject : d.Label,
                    Score = d.Score,
                    X = (int)MathF.Round(d.X0),
                    Y = (int)MathF.Round(d.Y0),
                    Width = (int)MathF.Round(d.X1 - d.X0),
                    Height = (int)MathF.Round(d.Y1 - d.Y0),
                });
            }
            return results;
        }
        finally
        {
            output.Logits.Dispose();
            output.PredBoxes.Dispose();
        }
    }

    /// <summary>Builds the DETR-preprocessed pixel tensor: aspect-preserving resize to shortest edge 800 / longest
    /// ≤ 1333, snapped even, then ImageNet-normalized. Boxes come back in the source frame via post-processing.</summary>
    private static Tensor BuildPixels(ImageData image)
    {
        double scale = (double)ShortestEdge / Math.Min(image.Width, image.Height);
        if (Math.Round(Math.Max(image.Width, image.Height) * scale) > LongestEdge)
        {
            scale = (double)LongestEdge / Math.Max(image.Width, image.Height);
        }
        int newW = Math.Max(2, (int)Math.Round(image.Width * scale));
        int newH = Math.Max(2, (int)Math.Round(image.Height * scale));
        newW -= newW % 2;
        newH -= newH % 2;
        byte[] rgb = FeatureImaging.ResizeRgb24(image, newW, newH);
        return VisionTensors.ImageNetNormalized(rgb, newW, newH);
    }

    private Entry GetOrLoad(IBackend backend, string checkpointPath, string vocabPath)
    {
        lock (_lock)
        {
            if (!ReferenceEquals(_boundBackend, backend))
            {
                DisposeAll();
                _boundBackend = backend;
            }
            if (_cache.TryGetValue(checkpointPath, out Entry? cached))
            {
                return cached;
            }
            Logs.Info($"[Vision] Loading Grounding DINO (tiny): {checkpointPath}");
            SafeTensorsLoader loader = new SafeTensorsLoader();
            try
            {
                loader.Load(checkpointPath);
                GroundingDinoModel model = new GroundingDinoModel(GroundingDinoConfig.Tiny);
                model.LoadWeights(loader.GetAllTensors());
                Entry created = new Entry
                {
                    Model = model,
                    Loader = loader,
                    Tokenizer = new BertWordPieceTokenizer(vocabPath, lowerCase: true),
                    Vocab = GroundingDinoPipeline.LoadVocab(vocabPath),
                };
                _cache[checkpointPath] = created;
                return created;
            }
            catch (Exception ex)
            {
                Logs.Error($"[Vision] Grounding DINO load failed for '{checkpointPath}'.", ex);
                loader.Dispose();
                throw;
            }
        }
    }

    private void DisposeAll()
    {
        foreach (Entry entry in _cache.Values)
        {
            try
            {
                entry.Model.Dispose();
                entry.Loader.Dispose();
                entry.Tokenizer.Dispose();
            }
            catch (Exception ex)
            {
                Logs.Error("[Vision] Grounding DINO dispose failed.", ex);
            }
        }
        _cache.Clear();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        lock (_lock)
        {
            DisposeAll();
            _boundBackend = null;
        }
    }

    /// <summary>A loaded detector: the model plus the loader that owns its weight tensors and the BERT tokenizer.</summary>
    private sealed class Entry
    {
        /// <summary>The loaded detector.</summary>
        public required GroundingDinoModel Model { get; init; }

        /// <summary>Owner of the model's weight tensors; must outlive the model.</summary>
        public required SafeTensorsLoader Loader { get; init; }

        /// <summary>BERT WordPiece tokenizer for the text query.</summary>
        public required BertWordPieceTokenizer Tokenizer { get; init; }

        /// <summary>Vocabulary lines used to map matched token spans back to label text.</summary>
        public required string[] Vocab { get; init; }
    }
}
