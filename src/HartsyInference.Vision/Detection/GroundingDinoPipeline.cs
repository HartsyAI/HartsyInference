using HartsyInference.Core.Backends;
using HartsyInference.Core.Pipelines;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Codec;

namespace HartsyInference.Vision.Detection;

/// <summary>Text-prompted open-vocabulary detection pipeline for Grounding DINO. Unlike <see cref="IDetectionPipeline"/>
/// (whose fixed signature has no place for a caption), detection here takes a free-text prompt: the caption is
/// tokenized, encoded by the injected <see cref="IGroundingTextEncoder"/>, and grounded against the image. Reuses
/// <see cref="YoloPreprocessor"/> (letterbox + normalize + CHW) and <see cref="PngDecoder"/> for the image path.
///
/// <para><b>Preprocessing note.</b> Real Grounding DINO normalizes with ImageNet mean/std; <see cref="YoloPreprocessor"/>
/// only divides by 255. That divergence is fine for this structural phase but must be corrected before real-weight
/// parity.</para></summary>
public sealed class GroundingDinoPipeline : IVisionPipeline
{
    private readonly IBackend _backend;
    private readonly GroundingDinoModel _model;
    private readonly IGroundingTextEncoder _textEncoder;
    private readonly IGroundingCaptionTokenizer _tokenizer;
    private readonly YoloPreprocessor _preprocessor;
    private readonly GroundingDinoPhraseAggregation _aggregation;
    private int _disposed;

    public string ModelName => "grounding-dino";

    /// <summary>The underlying model — exposed for diagnostics and weight preloading.</summary>
    public GroundingDinoModel Model => _model;

    /// <summary>Constructs the pipeline from an already-weight-loaded model plus the text encoder and caption
    /// tokenizer. <paramref name="inputSize"/> must be a multiple of 32 (the backbone's total stride).</summary>
    public GroundingDinoPipeline(
        IBackend backend,
        GroundingDinoModel model,
        IGroundingTextEncoder textEncoder,
        IGroundingCaptionTokenizer tokenizer,
        int inputSize = 800,
        GroundingDinoPhraseAggregation aggregation = GroundingDinoPhraseAggregation.Max)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(textEncoder);
        ArgumentNullException.ThrowIfNull(tokenizer);
        if (textEncoder.TextDim != model.Config.TextDim)
            throw new ArgumentException($"Text encoder dim {textEncoder.TextDim} != model text dim {model.Config.TextDim}.", nameof(textEncoder));

        _backend = backend;
        _model = model;
        _textEncoder = textEncoder;
        _tokenizer = tokenizer;
        _preprocessor = new YoloPreprocessor(inputSize, stride: 32, stridePadAlign: false);
        _aggregation = aggregation;
    }

    /// <summary>Decodes PNG bytes, then grounds the caption in the image.</summary>
    public IReadOnlyList<GroundedDetection> Detect(
        ReadOnlyMemory<byte> pngBytes,
        string caption,
        float boxThreshold = 0.35f,
        float textThreshold = 0.25f)
    {
        (byte[] rgb, int width, int height) = PngDecoder.Decode(pngBytes.Span);
        return Detect(rgb, width, height, caption, boxThreshold, textThreshold);
    }

    /// <summary>Grounds <paramref name="caption"/> in an HWC-packed RGB u8 image, returning one
    /// <see cref="GroundedDetection"/> per query that clears <paramref name="boxThreshold"/> and whose best phrase
    /// clears <paramref name="textThreshold"/>. Boxes are normalized xywh in <c>[0,1]</c>.</summary>
    public IReadOnlyList<GroundedDetection> Detect(
        ReadOnlySpan<byte> rgbPixels,
        int width,
        int height,
        string caption,
        float boxThreshold = 0.35f,
        float textThreshold = 0.25f)
    {
        ThrowIfDisposed();
        GroundingCaptionTokens tokens = _tokenizer.Tokenize(caption);
        if (tokens.TokenIds.Length == 0 || tokens.Phrases.Count == 0)
            return [];

        Tensor textEmbeddings = _textEncoder.Encode(_backend, tokens.TokenIds);
        (Tensor input, YoloPreprocessor.Transform _) = _preprocessor.Preprocess(rgbPixels, width, height);
        Tensor? logits = null;
        Tensor? boxes = null;
        try
        {
            (logits, boxes) = _model.Forward(_backend, input, textEmbeddings, null);
            return Decode(logits, boxes, tokens.Phrases, boxThreshold, textThreshold);
        }
        finally
        {
            textEmbeddings.Dispose();
            input.Dispose();
            logits?.Dispose();
            boxes?.Dispose();
        }
    }

    private List<GroundedDetection> Decode(
        Tensor logits,
        Tensor boxes,
        IReadOnlyList<GroundingPhraseSpan> phrases,
        float boxThreshold,
        float textThreshold)
    {
        int nq = (int)logits.Shape[1];
        int t = (int)logits.Shape[2];
        ReadOnlySpan<float> logitData = logits.AsSpan<float>();
        ReadOnlySpan<float> boxData = boxes.AsSpan<float>();

        List<GroundedDetection> results = new();
        float[] probRow = new float[t];
        for (int q = 0; q < nq; q++)
        {
            long lb = (long)q * t;
            float boxScore = 0f;
            for (int j = 0; j < t; j++)
            {
                float p = Sigmoid(logitData[(int)(lb + j)]);
                probRow[j] = p;
                if (p > boxScore)
                    boxScore = p;
            }
            if (boxScore <= boxThreshold)
                continue;

            int bestPhrase = -1;
            float bestScore = textThreshold;
            for (int pi = 0; pi < phrases.Count; pi++)
            {
                GroundingPhraseSpan span = phrases[pi];
                float agg = GroundingDinoHead.AggregateSpanScore(probRow, span.Start, span.End, _aggregation);
                if (agg > bestScore)
                {
                    bestScore = agg;
                    bestPhrase = pi;
                }
            }
            if (bestPhrase < 0)
                continue;

            long bb = (long)q * 4;
            float cx = boxData[(int)(bb + 0)];
            float cy = boxData[(int)(bb + 1)];
            float w = boxData[(int)(bb + 2)];
            float h = boxData[(int)(bb + 3)];
            float x = Math.Clamp(cx - w * 0.5f, 0f, 1f);
            float y = Math.Clamp(cy - h * 0.5f, 0f, 1f);

            GroundingPhraseSpan chosen = phrases[bestPhrase];
            results.Add(new GroundedDetection
            {
                Phrase = chosen.Phrase,
                Score = bestScore,
                Box = new DetectionResult
                {
                    X = x,
                    Y = y,
                    Width = w,
                    Height = h,
                    Confidence = boxScore,
                    ClassIndex = bestPhrase,
                    Label = chosen.Phrase,
                },
            });
        }
        return results;
    }

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    private void ThrowIfDisposed() =>
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    public void Dispose()
    {
        Interlocked.Exchange(ref _disposed, 1);
    }
}
