using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Segmentation.Sam3;

/// <summary>Detector decoder with iterative box refinement and a presence logit; inference only (apply_dac off).</summary>
public sealed class Sam3DetectorDecoder
{
    private const float TwoPi = 2f * MathF.PI;
    private const float RelativeBiasScale = 8f;
    private const float RelativeBiasLog2Range = 3f;
    private const float InverseSigmoidEps = 1e-3f;
    private static readonly float Ln2 = MathF.Log(2f);

    private readonly int _dModel;
    private readonly int _heads;
    private readonly int _numQueries;
    private readonly int _grid;
    private readonly float _eps;
    private readonly float[] _sineFrequencies;
    private readonly Sam3DecoderLayer[] _layers;
    private readonly Sam3Mlp _bboxEmbed = new();
    private readonly Sam3Mlp _referencePointHead = new();
    private readonly Sam3Mlp _relativeBiasX = new();
    private readonly Sam3Mlp _relativeBiasY = new();
    private readonly Sam3Mlp _presenceHead = new();
    private Tensor? _queryEmbed;
    private Tensor? _referencePoints;
    private Tensor? _normWeight;
    private Tensor? _normBias;
    private Tensor? _presenceToken;
    private Tensor? _presenceNormWeight;
    private Tensor? _presenceNormBias;

    /// <summary>Creates the decoder from the configuration.</summary>
    public Sam3DetectorDecoder(Sam3Config config)
    {
        ArgumentNullException.ThrowIfNull(config);
        if (config.DecoderLayers < 1 || config.NumQueries < 1 || config.PatchGrid < 1 || config.DModel % 2 != 0)
            throw new ArgumentException("Decoder needs layers, queries and a grid of at least one, and an even width.", nameof(config));

        _dModel = config.DModel;
        _heads = config.DecoderHeads;
        _numQueries = config.NumQueries;
        _grid = config.PatchGrid;
        _eps = config.LayerNormEps;

        int perCoordinate = _dModel / 2;
        _sineFrequencies = new float[perCoordinate];
        for (int i = 0; i < perCoordinate; i++)
            _sineFrequencies[i] = MathF.Pow(10000f, 2f * (i / 2) / perCoordinate);

        _layers = new Sam3DecoderLayer[config.DecoderLayers];
        for (int i = 0; i < _layers.Length; i++)
            _layers[i] = new Sam3DecoderLayer(config);
    }

    /// <summary>Loads query, reference, norm, box, relative-bias, presence heads and every decoder layer; fails naming the first missing key.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        int d = _dModel;
        _queryEmbed = Sam3WeightAccess.Require(weights, "query_embed.weight", _numQueries, d);
        _referencePoints = Sam3WeightAccess.Require(weights, "reference_points.weight", _numQueries, 4);
        _normWeight = Sam3WeightAccess.Require(weights, "norm.weight", d);
        _normBias = Sam3WeightAccess.Require(weights, "norm.bias", d);
        _bboxEmbed.LoadWeights(weights, "bbox_embed.", d, d, d, 4);
        _referencePointHead.LoadWeights(weights, "ref_point_head.", 2 * d, d, d);
        _relativeBiasX.LoadWeights(weights, "boxRPB_embed_x.", 2, d, _heads);
        _relativeBiasY.LoadWeights(weights, "boxRPB_embed_y.", 2, d, _heads);
        _presenceToken = Sam3WeightAccess.Require(weights, "presence_token.weight", 1, d);
        _presenceHead.LoadWeights(weights, "presence_token_head.", d, d, d, 1);
        _presenceNormWeight = Sam3WeightAccess.Require(weights, "presence_token_out_norm.weight", d);
        _presenceNormBias = Sam3WeightAccess.Require(weights, "presence_token_out_norm.bias", d);
        for (int i = 0; i < _layers.Length; i++)
            _layers[i].LoadWeights(weights, $"layers.{i}.");
    }

    /// <summary>Decodes the prompt-conditioned image memory <paramref name="memory"/> <c>[1, HW, D]</c> (with position
    /// <paramref name="memoryPos"/>). Returns owned tensors: boxes <c>[1, Nq, 4]</c> (normalized cxcywh), final query hidden
    /// states <c>[1, Nq, D]</c> and the presence logit <c>[1, 1]</c>.</summary>
    public (Tensor Boxes, Tensor QueryHidden, Tensor PresenceLogit) Forward(IBackend backend, Tensor memory, Tensor memoryPos, Tensor prompt, bool[] promptPadding)
    {
        ArgumentNullException.ThrowIfNull(backend);
        Tensor queryEmbed = _queryEmbed ?? throw new InvalidOperationException("SAM 3 decoder weights are not loaded.");
        Tensor referencePoints = _referencePoints ?? throw new InvalidOperationException("SAM 3 decoder weights are not loaded.");
        Tensor presenceToken = _presenceToken ?? throw new InvalidOperationException("SAM 3 decoder weights are not loaded.");
        Tensor normWeight = _normWeight ?? throw new InvalidOperationException("SAM 3 decoder weights are not loaded.");
        Tensor normBias = _normBias ?? throw new InvalidOperationException("SAM 3 decoder weights are not loaded.");
        Tensor presenceNormWeight = _presenceNormWeight ?? throw new InvalidOperationException("SAM 3 decoder weights are not loaded.");
        Tensor presenceNormBias = _presenceNormBias ?? throw new InvalidOperationException("SAM 3 decoder weights are not loaded.");
        Sam3Ops.RequireSequence(memory, "decoder memory", _dModel);
        Sam3Ops.RequireSequence(memoryPos, "decoder memory position", _dModel);
        if (memory.Shape[1] != (long)_grid * _grid)
            throw new ArgumentException($"Decoder memory must have {(long)_grid * _grid} rows, found {memory.Shape[1]}.", nameof(memory));

        int nq = _numQueries;
        Tensor queries = Sam3Ops.NewF32(1, nq, _dModel);
        queryEmbed.AsReadOnlySpan<float>().CopyTo(queries.AsSpan<float>());
        Tensor presence = Sam3Ops.NewF32(1, 1, _dModel);
        presenceToken.AsReadOnlySpan<float>().CopyTo(presence.AsSpan<float>());

        float[] refBoxes = new float[nq * 4];
        ReadOnlySpan<float> initial = referencePoints.AsReadOnlySpan<float>();
        for (int i = 0; i < refBoxes.Length; i++)
            refBoxes[i] = Sigmoid(initial[i]);

        Tensor? finalHidden = null;
        int last = _layers.Length - 1;
        for (int l = 0; l < _layers.Length; l++)
        {
            Tensor queryPos = BuildQueryPos(backend, refBoxes);
            Tensor crossBias = BuildRelativeBias(backend, refBoxes);
            (Tensor nextQueries, Tensor nextPresence) = _layers[l].Forward(
                backend, queries, presence, queryPos, prompt, promptPadding, memory, memoryPos, crossBias);
            queryPos.Dispose();
            crossBias.Dispose();
            queries.Dispose();
            presence.Dispose();
            queries = nextQueries;
            presence = nextPresence;

            Tensor normed = Sam3Ops.LayerNorm(backend, queries, normWeight, normBias, _eps);
            Tensor delta = _bboxEmbed.Forward(backend, normed);
            RefineBoxes(delta, refBoxes);
            delta.Dispose();
            if (l == last)
                finalHidden = normed;
            else
                normed.Dispose();
        }

        Tensor hidden = finalHidden ?? throw new InvalidOperationException("SAM 3 decoder has no layers.");
        Tensor presenceNorm = Sam3Ops.LayerNorm(backend, presence, presenceNormWeight, presenceNormBias, _eps);
        Tensor presenceRaw = _presenceHead.Forward(backend, presenceNorm);
        presenceNorm.Dispose();
        Tensor presenceLogit = Sam3Ops.NewF32(1, 1);
        presenceLogit.AsSpan<float>()[0] = presenceRaw.AsReadOnlySpan<float>()[0];
        presenceRaw.Dispose();
        presence.Dispose();
        queries.Dispose();

        Tensor boxes = Sam3Ops.NewF32(1, nq, 4);
        refBoxes.AsSpan().CopyTo(boxes.AsSpan<float>());
        return (boxes, hidden, presenceLogit);
    }

    /// <summary>Positional query <c>ref_point_head(sine(reference box))</c> for every query, <c>[1, Nq, D]</c>.</summary>
    private Tensor BuildQueryPos(IBackend backend, float[] refBoxes)
    {
        int width = 4 * _sineFrequencies.Length;
        Tensor sine = Sam3Ops.NewF32(1, _numQueries, width);
        Span<float> values = sine.AsSpan<float>();
        for (int q = 0; q < _numQueries; q++)
            EncodeBox(refBoxes.AsSpan(q * 4, 4), values.Slice(q * width, width));
        Tensor queryPos = _referencePointHead.Forward(backend, sine);
        sine.Dispose();
        return queryPos;
    }

    /// <summary>Sine embedding of a normalized cxcywh box, segments ordered y, x, w, h as upstream concatenates them.</summary>
    private void EncodeBox(ReadOnlySpan<float> box, Span<float> destination)
    {
        int perCoordinate = _sineFrequencies.Length;
        for (int segment = 0; segment < 4; segment++)
        {
            int coordinate = segment switch
            {
                0 => 1,
                1 => 0,
                _ => segment,
            };
            float angle = box[coordinate] * TwoPi;
            int offset = segment * perCoordinate;
            for (int i = 0; i < perCoordinate; i++)
            {
                float arg = angle / _sineFrequencies[i];
                destination[offset + i] = (i & 1) == 0 ? MathF.Sin(arg) : MathF.Cos(arg);
            }
        }
    }

    /// <summary>Additive image bias <c>[1, heads, Nq + 1, HW]</c> from the reference boxes; row 0 is the presence row and is zero.</summary>
    private Tensor BuildRelativeBias(IBackend backend, float[] refBoxes)
    {
        int nq = _numQueries;
        int g = _grid;
        int hw = g * g;
        int nh = _heads;
        int rowsPerHead = nq + 1;

        Tensor deltaX = Sam3Ops.NewF32(1, nq * g, 2);
        Tensor deltaY = Sam3Ops.NewF32(1, nq * g, 2);
        Span<float> dx = deltaX.AsSpan<float>();
        Span<float> dy = deltaY.AsSpan<float>();
        for (int q = 0; q < nq; q++)
        {
            float halfW = refBoxes[q * 4 + 2] * 0.5f;
            float halfH = refBoxes[q * 4 + 3] * 0.5f;
            float x0 = refBoxes[q * 4] - halfW;
            float x1 = refBoxes[q * 4] + halfW;
            float y0 = refBoxes[q * 4 + 1] - halfH;
            float y1 = refBoxes[q * 4 + 1] + halfH;
            for (int i = 0; i < g; i++)
            {
                float coord = i / (float)g;
                int offset = (q * g + i) * 2;
                dx[offset] = LogDelta(coord - x0);
                dx[offset + 1] = LogDelta(coord - x1);
                dy[offset] = LogDelta(coord - y0);
                dy[offset + 1] = LogDelta(coord - y1);
            }
        }

        Tensor embedX = _relativeBiasX.Forward(backend, deltaX);
        Tensor embedY = _relativeBiasY.Forward(backend, deltaY);
        deltaX.Dispose();
        deltaY.Dispose();

        Tensor bias = Sam3Ops.NewF32(1, nh, rowsPerHead, hw);
        Span<float> values = bias.AsSpan<float>();
        ReadOnlySpan<float> ex = embedX.AsReadOnlySpan<float>();
        ReadOnlySpan<float> ey = embedY.AsReadOnlySpan<float>();
        for (int h = 0; h < nh; h++)
        {
            for (int q = 0; q < nq; q++)
            {
                int rowBase = (h * rowsPerHead + q + 1) * hw;
                for (int yy = 0; yy < g; yy++)
                {
                    float yv = ey[(q * g + yy) * nh + h];
                    int colBase = rowBase + yy * g;
                    for (int xx = 0; xx < g; xx++)
                        values[colBase + xx] = yv + ex[(q * g + xx) * nh + h];
                }
            }
        }
        embedX.Dispose();
        embedY.Dispose();
        return bias;
    }

    /// <summary>Iterative refinement: <c>sigmoid(delta + inverse_sigmoid(box))</c> written back into <paramref name="refBoxes"/>.</summary>
    private static void RefineBoxes(Tensor delta, float[] refBoxes)
    {
        ReadOnlySpan<float> offsets = delta.AsReadOnlySpan<float>();
        for (int i = 0; i < refBoxes.Length; i++)
            refBoxes[i] = Sigmoid(offsets[i] + InverseSigmoid(refBoxes[i]));
    }

    /// <summary>Signed log-scaled offset used by the relative bias, <c>sign(8d) * log2(|8d| + 1) / log2(8)</c>.</summary>
    private static float LogDelta(float delta)
    {
        float scaled = delta * RelativeBiasScale;
        float magnitude = MathF.Log(MathF.Abs(scaled) + 1f) / Ln2 / RelativeBiasLog2Range;
        return scaled > 0f ? magnitude : (scaled < 0f ? -magnitude : 0f);
    }

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    private static float InverseSigmoid(float x)
    {
        float clamped = Math.Clamp(x, 0f, 1f);
        float numerator = MathF.Max(clamped, InverseSigmoidEps);
        float denominator = MathF.Max(1f - clamped, InverseSigmoidEps);
        return MathF.Log(numerator / denominator);
    }
}
