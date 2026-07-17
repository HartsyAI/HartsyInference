using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Detection;

/// <summary>End-to-end Grounding DINO forward pass: CNN backbone → flatten multi-scale features → cross-modality
/// feature enhancer → language-guided query selection → cross-modality decoder → contrastive class head + box head.
/// The text tower is supplied externally (see <see cref="IGroundingTextEncoder"/>); <see cref="Forward"/> takes the
/// already-encoded text token embeddings so the model never references a concrete BERT.
///
/// <para>The model owns the text→hidden projection used by both language-guided query selection and the contrastive
/// class head — the enhancer keeps the text stream at the BERT width, so a projection to the shared hidden width is
/// needed before dotting queries with text tokens.</para></summary>
public sealed class GroundingDinoModel
{
    private const float SigmoidEps = 1e-4f;

    private readonly GroundingDinoConfig _config;
    private readonly GroundingDinoBackbone _backbone;
    private readonly GroundingDinoFeatureEnhancer _enhancer;
    private readonly GroundingDinoDecoder _decoder;
    private readonly GroundingDinoHead _head;

    private Tensor? _textProjW;   // [hidden, textDim]
    private Tensor? _textProjB;   // [hidden]

    /// <summary>The configuration this model was built for.</summary>
    public GroundingDinoConfig Config => _config;

    public GroundingDinoModel(GroundingDinoConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        _config = config;
        _backbone = new GroundingDinoBackbone(config);
        _enhancer = new GroundingDinoFeatureEnhancer(config);
        _decoder = new GroundingDinoDecoder(config);
        _head = new GroundingDinoHead(config.ImageHiddenDim);
    }

    /// <summary>Resolves every parameter through the shared weight bag under <paramref name="prefix"/>.</summary>
    public void Build(IGroundingWeightBag bag, string prefix = "grounding_dino")
    {
        ArgumentNullException.ThrowIfNull(bag);
        _backbone.Build(bag, $"{prefix}.backbone");
        _textProjW = bag.Get($"{prefix}.text_proj.weight", _config.ImageHiddenDim, _config.TextDim);
        _textProjB = bag.Get($"{prefix}.text_proj.bias", _config.ImageHiddenDim);
        _enhancer.Build(bag, $"{prefix}.enhancer");
        _decoder.Build(bag, $"{prefix}.decoder");
        _head.Build(bag, $"{prefix}.head");
    }

    /// <summary>Loads weights from a safetensors dictionary.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix = "grounding_dino")
        => Build(new DictionaryWeightBag(weights), prefix);

    /// <summary>Constructs a model with deterministic synthetic weights — for shape / plumbing tests before a real
    /// checkpoint exists. Every parameter is generated from its key name, so the same config + seed is reproducible.</summary>
    public static GroundingDinoModel CreateWithSyntheticWeights(GroundingDinoConfig config, int seed = 1234)
    {
        GroundingDinoModel model = new(config);
        model.Build(new SyntheticWeightBag(seed));
        return model;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _backbone.EnumerateWeights()) yield return t;
        if (_textProjW is not null) yield return _textProjW;
        if (_textProjB is not null) yield return _textProjB;
        foreach (Tensor t in _enhancer.EnumerateWeights()) yield return t;
        foreach (Tensor t in _decoder.EnumerateWeights()) yield return t;
        foreach (Tensor t in _head.EnumerateWeights()) yield return t;
    }

    /// <summary>Runs the full forward pass. <paramref name="pixelValues"/> is <c>[1, 3, H, W]</c> (H, W multiples of
    /// 32). <paramref name="textTokenEmbeddings"/> is <c>[1, T, textDim]</c> from the text encoder. Optional
    /// <paramref name="textTokenMask"/> is <c>[1, T]</c> (1 = real token, 0 = pad). Returns the per-query,
    /// per-text-token contrastive logits <c>[1, Nq, T]</c> and boxes <c>[1, Nq, 4]</c> (cxcywh in [0,1]), where Nq =
    /// min(NumQueries, available image tokens). Caller owns both returned tensors.</summary>
    public (Tensor queryTextLogits, Tensor boxes) Forward(IBackend backend, Tensor pixelValues, Tensor textTokenEmbeddings, Tensor? textTokenMask)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(pixelValues);
        ArgumentNullException.ThrowIfNull(textTokenEmbeddings);
        if (_textProjW is null)
            throw new InvalidOperationException("GroundingDinoModel weights not loaded.");

        int hidden = _config.ImageHiddenDim;
        int t = (int)textTokenEmbeddings.Shape[1];

        // Backbone → flatten multi-scale features into one token sequence + per-token reference boxes.
        Tensor[] levels = _backbone.Forward(backend, pixelValues);
        Tensor imageTokens = FlattenLevels(backend, levels, out float[] referenceBoxes, out int nImg);
        for (int i = 0; i < levels.Length; i++)
            levels[i].Dispose();

        // Text-key additive masks (null when no padding).
        Tensor? textSelfMask = BuildKeyMask(t, t, textTokenMask);
        Tensor? imageToTextMask = BuildKeyMask(nImg, t, textTokenMask);

        // Cross-modality enhancer.
        (Tensor enhImage, Tensor enhText) = _enhancer.Forward(backend, imageTokens, textTokenEmbeddings, textSelfMask, imageToTextMask);
        imageTokens.Dispose();
        textSelfMask?.Dispose();
        imageToTextMask?.Dispose();

        // Project enhanced text to the shared hidden width; used for both query selection and the class head.
        Tensor textHidden = new Tensor(new TensorShape(1, t, hidden), DType.F32);
        backend.Linear(textHidden, enhText, _textProjW!, _textProjB);
        Tensor textHiddenT = new Tensor(new TensorShape(1, hidden, t), DType.F32);
        backend.Transpose2D(textHiddenT, textHidden, t, hidden);

        // Language-guided query selection: score each image token by its best match to any text token.
        int nq = Math.Min(_config.NumQueries, nImg);
        Tensor similarity = new Tensor(new TensorShape(1, nImg, t), DType.F32);
        backend.BatchedMatMul(similarity, enhImage, textHiddenT);
        int[] selected = SelectTopQueries(similarity, nImg, t, nq);
        similarity.Dispose();

        // Gather the selected image tokens as initial queries; their reference boxes seed the box head.
        Tensor queryContent = GatherQueries(backend, enhImage, selected, nq, hidden);
        Tensor queryRefLogit = BuildReferenceLogits(referenceBoxes, selected, nq);

        // Decoder.
        Tensor? queryToTextMask = BuildKeyMask(nq, t, textTokenMask);
        Tensor finalQueries = _decoder.Forward(backend, queryContent, enhImage, enhText, queryToTextMask);
        queryContent.Dispose();
        queryToTextMask?.Dispose();

        // Heads: contrastive class logits (query · text) + boxes.
        Tensor logits = new Tensor(new TensorShape(1, nq, t), DType.F32);
        backend.BatchedMatMul(logits, finalQueries, textHiddenT);
        Tensor boxes = _head.ForwardBoxes(backend, finalQueries, queryRefLogit);

        enhImage.Dispose();
        enhText.Dispose();
        textHidden.Dispose();
        textHiddenT.Dispose();
        finalQueries.Dispose();
        queryRefLogit.Dispose();
        return (logits, boxes);
    }

    private Tensor FlattenLevels(IBackend backend, Tensor[] levels, out float[] referenceBoxes, out int nImg)
    {
        int hidden = _config.ImageHiddenDim;
        nImg = 0;
        for (int i = 0; i < levels.Length; i++)
            nImg += (int)(levels[i].Shape[2] * levels[i].Shape[3]);
        referenceBoxes = new float[(long)nImg * 4];

        Tensor[] levelTokens = new Tensor[levels.Length];
        int tokenBase = 0;
        for (int l = 0; l < levels.Length; l++)
        {
            int hl = (int)levels[l].Shape[2];
            int wl = (int)levels[l].Shape[3];
            int hw = hl * wl;
            Tensor view = levels[l].Reshape(new TensorShape(1, hidden, hw));
            Tensor tokens = new Tensor(new TensorShape(1, hw, hidden), DType.F32);
            backend.Transpose2D(tokens, view, hidden, hw);
            view.Dispose();
            levelTokens[l] = tokens;

            // Reference boxes: cell centers as cx/cy, one-cell extent as w/h (level-appropriate scale).
            float bw = 1f / wl;
            float bh = 1f / hl;
            for (int y = 0; y < hl; y++)
            {
                for (int x = 0; x < wl; x++)
                {
                    long idx = (long)(tokenBase + y * wl + x) * 4;
                    referenceBoxes[idx + 0] = (x + 0.5f) * bw;
                    referenceBoxes[idx + 1] = (y + 0.5f) * bh;
                    referenceBoxes[idx + 2] = bw;
                    referenceBoxes[idx + 3] = bh;
                }
            }
            tokenBase += hw;
        }

        Tensor imageTokens = new Tensor(new TensorShape(1, nImg, hidden), DType.F32);
        backend.Concat(imageTokens, levelTokens, dim: 1);
        for (int l = 0; l < levelTokens.Length; l++)
            levelTokens[l].Dispose();
        return imageTokens;
    }

    /// <summary>Per-image-token max similarity to any text token, then the top-<paramref name="nq"/> token indices
    /// (descending). Reads the similarity tensor on the host — a selection over a small [Nimg, T] map, not model math.</summary>
    private static int[] SelectTopQueries(Tensor similarity, int nImg, int t, int nq)
    {
        ReadOnlySpan<float> sim = similarity.AsSpan<float>();
        float[] score = new float[nImg];
        for (int i = 0; i < nImg; i++)
        {
            float best = float.NegativeInfinity;
            long baseOff = (long)i * t;
            for (int j = 0; j < t; j++)
                if (sim[(int)(baseOff + j)] > best)
                    best = sim[(int)(baseOff + j)];
            score[i] = best;
        }

        int[] order = new int[nImg];
        for (int i = 0; i < nImg; i++)
            order[i] = i;
        Array.Sort(order, (a, b) => score[b].CompareTo(score[a]));

        int[] selected = new int[nq];
        Array.Copy(order, selected, nq);
        return selected;
    }

    private static Tensor GatherQueries(IBackend backend, Tensor enhImage, int[] selected, int nq, int hidden)
    {
        int nImg = (int)enhImage.Shape[1];
        Tensor imageFlat = enhImage.Reshape(new TensorShape(nImg, hidden));
        Tensor queries = new Tensor(new TensorShape(1, nq, hidden), DType.F32);
        Tensor queriesFlat = queries.Reshape(new TensorShape(nq, hidden));
        backend.GatherRows(queriesFlat, imageFlat, selected);
        queriesFlat.Dispose();
        imageFlat.Dispose();
        return queries;
    }

    private static Tensor BuildReferenceLogits(float[] referenceBoxes, int[] selected, int nq)
    {
        Tensor refLogit = new Tensor(new TensorShape(1, nq, 4), DType.F32);
        Span<float> dst = refLogit.AsSpan<float>();
        for (int i = 0; i < nq; i++)
        {
            long src = (long)selected[i] * 4;
            for (int c = 0; c < 4; c++)
                dst[i * 4 + c] = InverseSigmoid(referenceBoxes[src + c]);
        }
        return refLogit;
    }

    /// <summary>Builds a <c>[rows, T]</c> additive key mask (0 keep / −inf drop) from a <c>[1, T]</c> keep mask, or
    /// null when <paramref name="keepMask"/> is null (no padding → attend everything).</summary>
    private static Tensor? BuildKeyMask(int rows, int t, Tensor? keepMask)
    {
        if (keepMask is null)
            return null;
        ReadOnlySpan<float> keep = keepMask.AsSpan<float>();
        Tensor mask = new Tensor(new TensorShape(rows, t), DType.F32);
        Span<float> dst = mask.AsSpan<float>();
        for (int j = 0; j < t; j++)
        {
            float add = keep[j] > 0.5f ? 0f : float.NegativeInfinity;
            for (int r = 0; r < rows; r++)
                dst[r * t + j] = add;
        }
        return mask;
    }

    private static float InverseSigmoid(float p)
    {
        float clamped = Math.Clamp(p, SigmoidEps, 1f - SigmoidEps);
        return MathF.Log(clamped / (1f - clamped));
    }
}
