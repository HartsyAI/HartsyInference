using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Segmentation.Sam3;

/// <summary>SAM 3 ViT trunk: strided patch embed, tiled absolute position, pre-norm, then the block stack.</summary>
public sealed class Sam3VitTrunk
{
    private readonly int _imageSize;
    private readonly int _patchSize;
    private readonly int _grid;
    private readonly int _tokens;
    private readonly int _dim;
    private readonly float _eps;
    private readonly Tensor _patchWeight;
    private readonly Tensor _posEmbed;
    private readonly int[] _positionRows;
    private readonly Tensor _lnPreWeight;
    private readonly Tensor _lnPreBias;
    private readonly Sam3VitBlock[] _blocks;

    /// <summary>Loads the patch embed, position table, ln_pre and every block from trunk-relative keys.</summary>
    /// <param name="config">Trunk hyperparameters.</param>
    /// <param name="weights">Trunk weights keyed relative to <c>...vision_backbone.trunk.</c>.</param>
    public Sam3VitTrunk(Sam3Config config, IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(weights);

        _imageSize = config.ImageSize;
        _patchSize = config.PatchSize;
        _grid = config.PatchGrid;
        _tokens = _grid * _grid;
        _dim = config.VitEmbedDim;
        _eps = config.LayerNormEps;

        int pretrain = config.VitPretrainGrid;
        _patchWeight = Sam3WeightAccess.Require(weights, "patch_embed.proj.weight", _dim, 3, _patchSize, _patchSize);
        _posEmbed = Sam3WeightAccess.Require(weights, "pos_embed", 1, 1 + pretrain * pretrain, _dim);
        _lnPreWeight = Sam3WeightAccess.Require(weights, "ln_pre.weight", _dim);
        _lnPreBias = Sam3WeightAccess.Require(weights, "ln_pre.bias", _dim);
        _positionRows = BuildPositionRows(_grid, pretrain);

        int[] gather = Sam3WindowOrder.CreateGather(_grid, config.VitWindowSize);
        int[] scatter = Sam3WindowOrder.Invert(gather);
        _blocks = new Sam3VitBlock[config.VitDepth];
        for (int i = 0; i < _blocks.Length; i++)
        {
            string prefix = $"blocks.{i}.";
            bool global = Array.IndexOf(config.VitGlobalAttentionBlocks, i) >= 0;
            _blocks[i] = global
                ? new Sam3VitBlock(config, true, null, null, weights, prefix)
                : new Sam3VitBlock(config, false, gather, scatter, weights, prefix);
        }
    }

    /// <summary>Runs the trunk on a normalized image and returns the last-block map <c>[1, dim, grid, grid]</c>.</summary>
    /// <param name="backend">Backend that runs the ops.</param>
    /// <param name="image">F32 <c>[1, 3, ImageSize, ImageSize]</c> in [-1, 1].</param>
    /// <param name="rope">RoPE tables built for this configuration.</param>
    /// <returns>A new owned tensor; the caller disposes it.</returns>
    public Tensor Forward(IBackend backend, Tensor image, Sam3Rope rope)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(image);
        ArgumentNullException.ThrowIfNull(rope);
        if (image.DType != DType.F32 || image.Shape.Rank != 4 || image.Shape[0] != 1 || image.Shape[1] != 3
            || image.Shape[2] != _imageSize || image.Shape[3] != _imageSize)
        {
            throw new ArgumentException(
                $"Sam3 trunk expects F32 image [1,3,{_imageSize},{_imageSize}]; got {image.DType} {image.Shape}.", nameof(image));
        }

        using Tensor patches = new Tensor(new TensorShape(1, _dim, _grid, _grid), DType.F32);
        backend.Conv2D(patches, image, _patchWeight, null, _patchSize, _patchSize, 0, 0);
        using Tensor patchTokens = new Tensor(new TensorShape(1, _tokens, _dim), DType.F32);
        backend.Transpose2D(patchTokens, patches.Reshape(new TensorShape(1, _dim, _tokens)), _dim, _tokens);

        // Row 0 of the table is the class token, which upstream drops before tiling.
        Tensor positionTable = _posEmbed.Reshape(new TensorShape(_posEmbed.Shape[1], _dim));
        using Tensor positions = new Tensor(new TensorShape(1, _tokens, _dim), DType.F32);
        backend.GatherRows(positions, positionTable, _positionRows);
        using Tensor withPositions = new Tensor(new TensorShape(1, _tokens, _dim), DType.F32);
        backend.Add(withPositions, patchTokens, positions);

        Tensor state = new Tensor(new TensorShape(1, _tokens, _dim), DType.F32);
        backend.LayerNorm(state, withPositions, _lnPreWeight, _lnPreBias, _eps);
        foreach (Sam3VitBlock block in _blocks)
        {
            Tensor next = block.Forward(backend, state, rope);
            state.Dispose();
            state = next;
        }

        Tensor features = new Tensor(new TensorShape(1, _dim, _grid, _grid), DType.F32);
        // Write into the owned output: a backend op that writes through a reshape view leaves the parent's device result unread.
        backend.Transpose2D(features, state, _tokens, _dim);
        state.Dispose();
        return features;
    }

    // Tiles the pretrained grid across the token grid by wrapping indices, as upstream does for ImageSize/PatchSize.
    private static int[] BuildPositionRows(int grid, int pretrain)
    {
        int[] rows = new int[grid * grid];
        for (int y = 0; y < grid; y++)
        {
            for (int x = 0; x < grid; x++)
            {
                rows[y * grid + x] = 1 + (y % pretrain) * pretrain + (x % pretrain);
            }
        }

        return rows;
    }
}
