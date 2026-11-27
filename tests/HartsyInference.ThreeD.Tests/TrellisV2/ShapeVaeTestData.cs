using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.TrellisV2;

namespace HartsyInference.ThreeD.Tests.TrellisV2;

/// <summary>Tiny synthetic shape-VAE architecture and random weights for CPU unit tests.</summary>
internal static unsafe class ShapeVaeTestData
{
    /// <summary>Three levels (4x downsample), 4 latent channels: 8 → 16 → 32 encoder, 32 → 16 → 8 decoder.</summary>
    internal static ShapeVaeEncoderConfig Encoder() => new()
    {
        ModelChannels = [8, 16, 32],
        LatentChannels = 4,
        NumBlocks = [1, 1, 2],
        BlockType = ["SparseConvNeXtBlock3d", "SparseConvNeXtBlock3d", "SparseConvNeXtBlock3d"],
        DownBlockType = ["SparseResBlockS2C3d", "SparseResBlockS2C3d"],
        UseCheckpoint = [false, false, false],
        UseFp16 = false,
    };

    internal static ShapeVaeDecoderConfig Decoder() => new()
    {
        Resolution = 16,
        ModelChannels = [32, 16, 8],
        LatentChannels = 4,
        NumBlocks = [2, 1, 0],
        BlockType = ["SparseConvNeXtBlock3d", "SparseConvNeXtBlock3d", "SparseConvNeXtBlock3d"],
        UpBlockType = ["SparseResBlockC2S3d", "SparseResBlockC2S3d"],
        UseCheckpoint = [false, false, false],
        UseFp16 = false,
    };

    internal static Dictionary<string, Tensor> RandomWeights(IReadOnlyList<ShapeVaeKeyMap.TensorSpec> specs, int seed)
    {
        Random rng = new(seed);
        Dictionary<string, Tensor> weights = new(specs.Count);
        foreach (ShapeVaeKeyMap.TensorSpec spec in specs)
        {
            Tensor t = new(new TensorShape(spec.Shape), DType.F32);
            FillUniform(rng, t);
            weights[spec.CheckpointKey] = t;
        }
        return weights;
    }

    internal static void FillUniform(Random rng, Tensor t)
    {
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 0.2 - 0.1);
    }

    /// <summary>Distinct random voxels in <c>[0, res)³</c>, batch 0, as flat (b, x, y, z) quads.</summary>
    internal static int[] RandomCoords(Random rng, int n, int res)
    {
        HashSet<(int, int, int)> seen = new();
        List<int> coords = new();
        while (seen.Count < n)
        {
            int x = rng.Next(res), y = rng.Next(res), z = rng.Next(res);
            if (seen.Add((x, y, z))) coords.AddRange(new[] { 0, x, y, z });
        }
        return coords.ToArray();
    }

    internal static Tensor RandomFeats(Random rng, int n, int channels, double low, double high)
    {
        Tensor t = new(new TensorShape(n, channels), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = (float)(low + rng.NextDouble() * (high - low));
        return t;
    }

    internal static SparseTensor Sparse(int[] coords, int resolution, Tensor feats) => new(feats, coords, resolution);

    internal static float[] Read(Tensor t)
    {
        float* p = (float*)t.DataPointer;
        float[] values = new float[t.ElementCount];
        for (long i = 0; i < values.Length; i++) values[i] = p[i];
        return values;
    }

    /// <summary>Random encoder and decoder weights for the given configs, each keyed by its own module's checkpoint keys.</summary>
    internal static (Dictionary<string, Tensor> Encoder, Dictionary<string, Tensor> Decoder) RandomPair(
        ShapeVaeEncoderConfig encoder, ShapeVaeDecoderConfig decoder, int seed)
        => (RandomWeights(ShapeVaeKeyMap.Encoder(encoder), seed), RandomWeights(ShapeVaeKeyMap.Decoder(decoder), seed + 1));
}
