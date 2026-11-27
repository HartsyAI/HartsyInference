using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.Utonia;

namespace HartsyInference.ThreeD.Tests.Utonia;

/// <summary>Tiny synthetic Utonia configs and random weights. No checkpoint is read.</summary>
internal static class UtoniaTestData
{
    public static UtoniaConfig TinyConfig(bool encoderOnly = true, bool qkvBias = true) => new()
    {
        InChannels = 9,
        Order = ["z", "z-trans"],
        Stride = [2, 2],
        EncDepths = [1, 1, 1],
        EncChannels = [12, 24, 36],
        EncNumHeads = [2, 4, 6],
        EncPatchSize = [8, 8, 8],
        EncoderOnly = encoderOnly,
        DecDepths = [1, 1],
        DecChannels = [12, 24],
        DecNumHeads = [2, 4],
        DecPatchSize = [8, 8],
        MlpRatio = 2.0,
        QkvBias = qkvBias,
        LayerScaleInit = 0.1f,
    };

    /// <summary>Random weights for every parameter the config implies.</summary>
    public static UtoniaWeights RandomWeights(UtoniaConfig config, int seed) => new(RandomTensors(config, seed));

    /// <summary>Random tensors keyed by C# name. Norm scales are 1; LayerScale gammas are 0.1.</summary>
    public static Dictionary<string, Tensor> RandomTensors(UtoniaConfig config, int seed)
    {
        Random rng = new(seed);
        Dictionary<string, Tensor> tensors = new(StringComparer.Ordinal);
        foreach (UtoniaParameterSpec spec in UtoniaCheckpointKeys.Parameters(config))
        {
            bool unitScale = spec.CSharpName.Contains("Norm", StringComparison.Ordinal) && spec.CSharpName.EndsWith(".Weight", StringComparison.Ordinal);
            bool gamma = spec.CSharpName.EndsWith(".Gamma", StringComparison.Ordinal);
            long count = 1;
            foreach (long d in spec.Shape)
            {
                count *= d;
            }

            float[] values = new float[count];
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = unitScale ? 1f : gamma ? 0.1f : (float)(rng.NextDouble() * 0.2 - 0.1);
            }

            tensors[spec.CSharpName] = Make(spec.Shape, values);
        }

        return tensors;
    }

    public static unsafe Tensor Make(long[] shape, float[] values)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < values.Length; i++)
        {
            p[i] = values[i];
        }

        return t;
    }

    public static unsafe float[] Read(Tensor t)
    {
        long count = t.ElementCount;
        float[] values = new float[count];
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < count; i++)
        {
            values[i] = p[i];
        }

        return values;
    }
}
