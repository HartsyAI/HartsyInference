using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Vision.Detection;
using Xunit;

namespace HartsyInference.Vision.Tests;

/// <summary>Grounding DINO structural bring-up tests: synthetic-weight forward passes (shape + plumbing), a pure
/// unit test of the phrase-span aggregation math, and an env-gated real-weight parity stub. The synthetic forward
/// tests are <c>SyntheticSmoke</c> — the model is not yet real-weight-validated, so they run on the GPU/manual lane,
/// not the hosted CI Unit gate.</summary>
public sealed class GroundingDinoForwardTests
{
    /// <summary>Tiny config so a full forward runs in milliseconds on CPU: 32-d hidden, 16-d text, one enhancer and
    /// one decoder layer, 8 queries, a 5-channel backbone.</summary>
    private static GroundingDinoConfig TinyConfig => new()
    {
        ImageHiddenDim = 32,
        TextDim = 16,
        NumHeads = 8,
        NumFeatureLevels = 3,
        NumEnhancerLayers = 1,
        NumDecoderLayers = 1,
        NumQueries = 8,
        FeedForwardDim = 64,
        TextFeedForwardDim = 32,
        MaxTextTokens = 32,
        BackboneChannels = [16, 16, 32, 32, 32],
    };

    [Fact]
    [Trait("Category", "SyntheticSmoke")]
    public void FeatureEnhancer_PreservesImageAndTextShapes()
    {
        GroundingDinoConfig config = TinyConfig;
        int nImg = 6;
        int numText = 5;

        GroundingDinoFeatureEnhancer enhancer = new(config);
        enhancer.Build(new SyntheticWeightBag(seed: 11), "enhancer");

        using IBackend backend = new CpuBackend();
        Tensor image = MakeDeterministic(new TensorShape(1, nImg, config.ImageHiddenDim), 1);
        Tensor text = MakeDeterministic(new TensorShape(1, numText, config.TextDim), 2);

        (Tensor enhImage, Tensor enhText) = enhancer.Forward(backend, image, text, null, null);
        try
        {
            Assert.Equal(3, enhImage.Shape.Rank);
            Assert.Equal(1, enhImage.Shape[0]);
            Assert.Equal(nImg, enhImage.Shape[1]);
            Assert.Equal(config.ImageHiddenDim, enhImage.Shape[2]);

            Assert.Equal(3, enhText.Shape.Rank);
            Assert.Equal(numText, enhText.Shape[1]);
            Assert.Equal(config.TextDim, enhText.Shape[2]);

            AssertAllFinite(enhImage);
            AssertAllFinite(enhText);
        }
        finally
        {
            image.Dispose();
            text.Dispose();
            enhImage.Dispose();
            enhText.Dispose();
        }
    }

    [Fact]
    [Trait("Category", "SyntheticSmoke")]
    public void Model_Forward_ReturnsQueryTextLogitsAndNormalizedBoxes()
    {
        GroundingDinoConfig config = TinyConfig;
        int numText = 5;
        GroundingDinoModel model = GroundingDinoModel.CreateWithSyntheticWeights(config, seed: 5);

        using IBackend backend = new CpuBackend();
        Tensor pixels = MakeDeterministic(new TensorShape(1, 3, 64, 64), 3);
        Tensor textEmb = MakeDeterministic(new TensorShape(1, numText, config.TextDim), 4);

        (Tensor logits, Tensor boxes) = model.Forward(backend, pixels, textEmb, null);
        try
        {
            // Nq = min(NumQueries, image tokens). A 64×64 image → 8×8 + 4×4 + 2×2 = 84 tokens, so Nq = 8.
            Assert.Equal(3, logits.Shape.Rank);
            Assert.Equal(1, logits.Shape[0]);
            Assert.Equal(config.NumQueries, logits.Shape[1]);
            Assert.Equal(numText, logits.Shape[2]);

            Assert.Equal(config.NumQueries, boxes.Shape[1]);
            Assert.Equal(4, boxes.Shape[2]);

            AssertAllFinite(logits);
            ReadOnlySpan<float> boxData = boxes.AsSpan<float>();
            for (int i = 0; i < boxData.Length; i++)
                Assert.InRange(boxData[i], 0f, 1f);
        }
        finally
        {
            pixels.Dispose();
            textEmb.Dispose();
            logits.Dispose();
            boxes.Dispose();
        }
    }

    [Fact]
    [Trait("Category", "SyntheticSmoke")]
    public void Pipeline_Detect_ReturnsNormalizedBoxesWithoutThrowing()
    {
        GroundingDinoConfig config = TinyConfig;
        GroundingDinoModel model = GroundingDinoModel.CreateWithSyntheticWeights(config, seed: 9);

        LinearGroundingTextEncoder textEncoder = new(vocabSize: 64, embedDim: config.TextDim, textDim: config.TextDim);
        textEncoder.Build(new SyntheticWeightBag(seed: 21), "text_encoder");

        using IBackend backend = new CpuBackend();
        using GroundingDinoPipeline pipeline = new(
            backend, model, textEncoder, new WhitespaceCaptionTokenizer(vocabSize: 64), inputSize: 64);

        byte[] rgb = new byte[64 * 64 * 3];
        for (int i = 0; i < rgb.Length; i++)
            rgb[i] = (byte)((i * 37 + 11) & 0xFF);

        // Very low thresholds so the untrained model still surfaces some detections to exercise the box mapping.
        IReadOnlyList<GroundedDetection> detections =
            pipeline.Detect(rgb, 64, 64, "a cat . a remote control .", boxThreshold: 0.0f, textThreshold: 0.0f);

        Assert.NotNull(detections);
        foreach (GroundedDetection det in detections)
        {
            Assert.InRange(det.Box.X, 0f, 1f);
            Assert.InRange(det.Box.Y, 0f, 1f);
            Assert.InRange(det.Box.Width, 0f, 1f);
            Assert.InRange(det.Box.Height, 0f, 1f);
            Assert.False(string.IsNullOrEmpty(det.Phrase));
        }
    }

    [Fact]
    public void AggregateSpanScore_MaxAndMean_OverKnownScores()
    {
        // Per-token scores for one query: 4 tokens. Phrase "A" spans [0,2), phrase "B" spans [2,4).
        float[] scores = [0.10f, 0.70f, 0.30f, 0.20f];

        float aMax = GroundingDinoHead.AggregateSpanScore(scores, 0, 2, GroundingDinoPhraseAggregation.Max);
        float aMean = GroundingDinoHead.AggregateSpanScore(scores, 0, 2, GroundingDinoPhraseAggregation.Mean);
        float bMax = GroundingDinoHead.AggregateSpanScore(scores, 2, 4, GroundingDinoPhraseAggregation.Max);

        Assert.Equal(0.70f, aMax, 5);
        Assert.Equal(0.40f, aMean, 5);
        Assert.Equal(0.30f, bMax, 5);

        // Empty / clamped spans.
        Assert.Equal(float.NegativeInfinity, GroundingDinoHead.AggregateSpanScore(scores, 3, 3, GroundingDinoPhraseAggregation.Max));
        Assert.Equal(0.20f, GroundingDinoHead.AggregateSpanScore(scores, 3, 99, GroundingDinoPhraseAggregation.Max), 5);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void RealWeights_Parity_SkipsWhenCheckpointUnset()
    {
        string? path = Environment.GetEnvironmentVariable("GROUNDING_DINO_PATH");
        if (string.IsNullOrEmpty(path) || !File.Exists(path))
            return; // No checkpoint → skip cleanly (real-weight parity is not yet established for Grounding DINO).

        // A real safetensors checkpoint would load here; the key remap from HF Grounding DINO names to this
        // engine's naming is not implemented yet, so this path is a placeholder for future parity work.
        Assert.True(File.Exists(path));
    }

    private static Tensor MakeDeterministic(TensorShape shape, int seed)
    {
        Tensor tensor = new(shape, DType.F32);
        Span<float> data = tensor.AsSpan<float>();
        uint state = (uint)(seed * 2654435761u + 1u);
        for (int i = 0; i < data.Length; i++)
        {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            data[i] = ((state & 0xFFFF) / 65535f - 0.5f) * 0.2f;
        }
        return tensor;
    }

    private static void AssertAllFinite(Tensor tensor)
    {
        ReadOnlySpan<float> data = tensor.AsSpan<float>();
        for (int i = 0; i < data.Length; i++)
            Assert.True(float.IsFinite(data[i]), $"Non-finite value at index {i}: {data[i]}");
    }

    /// <summary>Trivial caption tokenizer for tests — splits on '.', then whitespace, mapping each word to a
    /// bounded pseudo-id. No vocab file needed, so the pipeline runs on the synthetic text encoder.</summary>
    private sealed class WhitespaceCaptionTokenizer : IGroundingCaptionTokenizer
    {
        private readonly int _vocabSize;

        public WhitespaceCaptionTokenizer(int vocabSize) => _vocabSize = vocabSize;

        public GroundingCaptionTokens Tokenize(string caption)
        {
            List<int> ids = new() { 1 }; // pseudo-[CLS]
            List<GroundingPhraseSpan> phrases = new();
            foreach (string rawPhrase in (caption ?? "").Split('.'))
            {
                string phrase = rawPhrase.Trim();
                if (phrase.Length == 0)
                    continue;
                int start = ids.Count;
                foreach (string word in phrase.Split(' ', StringSplitOptions.RemoveEmptyEntries))
                    ids.Add(2 + (Math.Abs(word.GetHashCode()) % (_vocabSize - 2)));
                int end = ids.Count;
                if (end > start)
                    phrases.Add(new GroundingPhraseSpan(phrase, start, end));
                ids.Add(0); // pseudo-[SEP]
            }
            return new GroundingCaptionTokens { TokenIds = ids.ToArray(), Phrases = phrases };
        }
    }
}
