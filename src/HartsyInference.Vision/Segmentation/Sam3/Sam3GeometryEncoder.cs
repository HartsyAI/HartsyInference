using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Segmentation.Sam3;

/// <summary>Geometry encoder CLS path for text-only prompts; the point and box submodules are not loaded.</summary>
public sealed class Sam3GeometryEncoder
{
    private readonly int _dModel;
    private readonly float _eps;
    private readonly Sam3FusionEncoderLayer[] _layers;
    private Tensor? _clsEmbed;
    private Tensor? _finalProjWeight;
    private Tensor? _finalProjBias;
    private Tensor? _normWeight;
    private Tensor? _normBias;
    private Tensor? _encodeNormWeight;
    private Tensor? _encodeNormBias;

    /// <summary>Creates the CLS-path encoder from the configuration.</summary>
    public Sam3GeometryEncoder(Sam3Config config)
    {
        ArgumentNullException.ThrowIfNull(config);
        if (config.GeometryLayers < 1)
            throw new ArgumentException("The geometry encoder needs at least one layer.", nameof(config));
        _dModel = config.DModel;
        _eps = config.LayerNormEps;
        _layers = new Sam3FusionEncoderLayer[config.GeometryLayers];
        for (int i = 0; i < _layers.Length; i++)
            _layers[i] = new Sam3FusionEncoderLayer(config.DModel, config.EncoderHeads, config.EncoderFfn, config.LayerNormEps);
    }

    /// <summary>Loads <c>cls_embed</c>, <c>final_proj</c>, <c>norm</c>, <c>encode.{i}.*</c> and <c>encode_norm</c>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        int d = _dModel;
        _clsEmbed = Sam3WeightAccess.Require(weights, "cls_embed.weight", 1, d);
        _finalProjWeight = Sam3WeightAccess.Require(weights, "final_proj.weight", d, d);
        _finalProjBias = Sam3WeightAccess.Require(weights, "final_proj.bias", d);
        _normWeight = Sam3WeightAccess.Require(weights, "norm.weight", d);
        _normBias = Sam3WeightAccess.Require(weights, "norm.bias", d);
        for (int i = 0; i < _layers.Length; i++)
            _layers[i].LoadWeights(weights, $"encode.{i}.");
        _encodeNormWeight = Sam3WeightAccess.Require(weights, "encode_norm.weight", d);
        _encodeNormBias = Sam3WeightAccess.Require(weights, "encode_norm.bias", d);
    }

    /// <summary>Returns the CLS token <c>[1, 1, D]</c> refined against the image tokens <c>[1, HW, D]</c> and their position
    /// encoding. The returned tensor is owned by the caller; the inputs are not disposed.</summary>
    public Tensor Forward(IBackend backend, Tensor imageTokens, Tensor imagePos)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(imageTokens);
        ArgumentNullException.ThrowIfNull(imagePos);
        Tensor clsEmbed = _clsEmbed ?? throw new InvalidOperationException("SAM 3 geometry encoder weights are not loaded.");
        Tensor finalProjWeight = _finalProjWeight ?? throw new InvalidOperationException("SAM 3 geometry encoder weights are not loaded.");
        Tensor finalProjBias = _finalProjBias ?? throw new InvalidOperationException("SAM 3 geometry encoder weights are not loaded.");
        Tensor normWeight = _normWeight ?? throw new InvalidOperationException("SAM 3 geometry encoder weights are not loaded.");
        Tensor normBias = _normBias ?? throw new InvalidOperationException("SAM 3 geometry encoder weights are not loaded.");
        Tensor encodeNormWeight = _encodeNormWeight ?? throw new InvalidOperationException("SAM 3 geometry encoder weights are not loaded.");
        Tensor encodeNormBias = _encodeNormBias ?? throw new InvalidOperationException("SAM 3 geometry encoder weights are not loaded.");
        Sam3Ops.RequireSequence(imageTokens, "geometry image tokens", _dModel);
        Sam3Ops.RequireSequence(imagePos, "geometry image position", _dModel);

        Tensor cls = Sam3Ops.NewF32(1, 1, _dModel);
        clsEmbed.AsReadOnlySpan<float>().CopyTo(cls.AsSpan<float>());
        Tensor projected = Sam3Ops.Linear(backend, cls, finalProjWeight, finalProjBias);
        cls.Dispose();
        Tensor current = Sam3Ops.LayerNorm(backend, projected, normWeight, normBias, _eps);
        projected.Dispose();

        for (int i = 0; i < _layers.Length; i++)
        {
            Tensor next = _layers[i].Forward(backend, current, null, imageTokens, imagePos, null);
            current.Dispose();
            current = next;
        }

        Tensor output = Sam3Ops.LayerNorm(backend, current, encodeNormWeight, encodeNormBias, _eps);
        current.Dispose();
        return output;
    }
}
