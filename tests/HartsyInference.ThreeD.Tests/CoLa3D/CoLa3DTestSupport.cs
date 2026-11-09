using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.CoLa3D;

namespace HartsyInference.ThreeD.Tests.CoLa3D;

/// <summary>Configs, synthetic weights and fixture readers shared by the CoLa3D tests.</summary>
internal static class CoLa3DTestSupport
{
    /// <summary>The architecture of <c>cola3d_free_q20.pt</c>, read from <c>configs/train/cola3d/free_q20.json</c>.</summary>
    internal static CoLa3DConfig FreeQ20() => new()
    {
        LatentChannels = 32,
        LatentResolution = 64,
        UtoniaFeatureDim = 576,
        HiddenDim = 768,
        NumLayers = 6,
        NumHeads = 12,
        MaxParts = 20,
        QuerySource = CoLa3DConfig.QuerySourceKind.Fixed,
        QueryInit = CoLa3DConfig.QueryInitKind.Positional,
    };

    /// <summary>The architecture of <c>cola3d_mask_q20.pt</c>.</summary>
    internal static CoLa3DConfig MaskQ20() => FreeQ20() with
    {
        QuerySource = CoLa3DConfig.QuerySourceKind.PromptMask,
        MaskEncoder = CoLa3DConfig.MaskEncoderKind.CnnCenterPe,
        MaskEncoderHidden = 128,
        MaskPePerAxis = 128,
    };

    /// <summary>The architecture of <c>cola3d_point_q20.pt</c>.</summary>
    internal static CoLa3DConfig PointQ20() => FreeQ20() with
    {
        QuerySource = CoLa3DConfig.QuerySourceKind.PromptPoint,
        PointFeatureSource = CoLa3DConfig.PointFeatureSourceKind.LatentUtonia,
        PointFourierFeatures = 64,
    };

    /// <summary>A small configuration that exercises every code path: head dim 8 (one identity RoPE pair), two blocks.</summary>
    internal static CoLa3DConfig Tiny(CoLa3DConfig.QuerySourceKind source) => new()
    {
        LatentChannels = 4,
        LatentResolution = 8,
        UtoniaFeatureDim = 5,
        HiddenDim = 16,
        NumLayers = 2,
        NumHeads = 2,
        MaxParts = 3,
        MlpRatio = 2.0,
        QuerySource = source,
        QueryInit = CoLa3DConfig.QueryInitKind.Positional,
        PointFeatureSource = CoLa3DConfig.PointFeatureSourceKind.Latent,
        MaskEncoder = CoLa3DConfig.MaskEncoderKind.CnnCenterPe,
        MaskEncoderHidden = 8,
        MaskPePerAxis = 4,
        PointFourierFeatures = 4,
    };

    /// <summary>Random parameters for every name the config uses. Norm scales start near 1, RMS gains near 1, and the
    /// temperature at 0.07 as in the checkpoints; everything else is small and centred on zero.</summary>
    internal static Dictionary<string, Tensor> RandomWeights(CoLa3DConfig config, int seed)
    {
        Random random = new(seed);
        Dictionary<string, Tensor> weights = new(StringComparer.Ordinal);
        foreach (KeyValuePair<string, long[]> entry in CoLa3DKeyMap.ExpectedShapes(config))
        {
            Tensor tensor = new(new TensorShape(entry.Value), DType.F32);
            Span<float> values = tensor.AsSpan<float>();
            bool unit = entry.Key.Contains("Norm", StringComparison.Ordinal) || entry.Key.EndsWith(".gamma", StringComparison.Ordinal);
            for (int i = 0; i < values.Length; i++)
                values[i] = unit ? 1f + (float)(random.NextDouble() - 0.5) * 0.1f : (float)(random.NextDouble() - 0.5) * 0.2f;
            if (entry.Key == "head.temperature")
                values[0] = 0.07f;
            weights[entry.Key] = tensor;
        }

        return weights;
    }

    internal static void Dispose(IEnumerable<Tensor> tensors)
    {
        foreach (Tensor tensor in tensors)
            tensor.Dispose();
    }

    /// <summary>Reads a fixture of <c>key&lt;TAB&gt;[d0, d1, …]</c> lines; lines starting with <c>#</c> are comments.</summary>
    internal static Dictionary<string, long[]> ReadFixture(string fileName)
    {
        string path = Path.Combine(AppContext.BaseDirectory, "CoLa3D", "Fixtures", fileName);
        if (!File.Exists(path))
            throw new FileNotFoundException($"CoLa3D fixture '{fileName}' is missing from the test output; expected at {path}.", path);

        Dictionary<string, long[]> shapes = new(StringComparer.Ordinal);
        foreach (string line in File.ReadLines(path))
        {
            if (line.Length == 0 || line[0] == '#')
                continue;
            string[] columns = line.Split('\t');
            string[] dims = columns[1].Trim('[', ']').Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
            shapes.Add(columns[0], dims.Select(long.Parse).ToArray());
        }

        return shapes;
    }

    /// <summary>A latent with <paramref name="voxelsPerSample"/> voxels per sample on an 8³ grid, with random features and
    /// Utonia features. Voxel coordinates are distinct within each sample and sorted by batch.</summary>
    internal static (SparseTensor Latent, Tensor Utonia) RandomLatent(CoLa3DConfig config, int[] voxelsPerSample, int seed)
    {
        Random random = new(seed);
        int n = voxelsPerSample.Sum();
        int[] coords = new int[n * 4];
        int row = 0;
        for (int b = 0; b < voxelsPerSample.Length; b++)
        {
            for (int v = 0; v < voxelsPerSample[b]; v++, row++)
            {
                coords[row * 4] = b;
                coords[row * 4 + 1] = (v * 3 + b) % config.LatentResolution;
                coords[row * 4 + 2] = (v * 5 + 1) % config.LatentResolution;
                coords[row * 4 + 3] = (v * 7 + 2 * b) % config.LatentResolution;
            }
        }

        Tensor feats = new(new TensorShape(n, config.LatentChannels), DType.F32);
        Fill(feats, random);
        Tensor utonia = new(new TensorShape(n, config.UtoniaFeatureDim), DType.F32);
        Fill(utonia, random);
        return (new SparseTensor(feats, coords, config.LatentResolution), utonia);
    }

    private static void Fill(Tensor tensor, Random random)
    {
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++)
            values[i] = (float)(random.NextDouble() * 2.0 - 1.0);
    }
}
