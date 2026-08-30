using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Lora;
using HartsyInference.ModelAssets.MiniMaxH3;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

public sealed class MiniMaxH3PddKeyConverterTests
{
    private static readonly HashSet<string> _headKeys = new(StringComparer.Ordinal)
    {
        "proj_out.weight", "proj_out.bias", "audio_proj_out.weight", "audio_proj_out.bias",
    };

    [Fact]
    public void ConvertOfficial_FusesQkvSwapsSwiGluAndConsumesEveryTensor()
    {
        Dictionary<string, Tensor> source = CreateOfficialSource();
        try
        {
            using MiniMaxH3PddTrunkConversion converted =
                MiniMaxH3PddKeyConverter.Convert(source, _headKeys, 64.0f);

            Assert.Equal(258, converted.Layers.Count);
            LoraLayer qkv = Assert.Single(converted.Layers,
                layer => layer.TargetKey == "blocks.0.attn.qkv_proj.weight");
            Assert.Equal(new float[] { 1, 2, 3, 4, 5, 6 }, qkv.LoraDown.AsSpan<float>().ToArray());
            Assert.Equal(3, qkv.Rank);
            Assert.Equal(192.0f, qkv.Alpha);
            Assert.Equal(new float[]
            {
                10, 0, 0,
                11, 0, 0,
                0, 20, 0,
                0, 21, 0,
                0, 0, 30,
                0, 0, 31,
            }, qkv.LoraUp.AsSpan<float>().ToArray());

            LoraLayer fc1 = Assert.Single(converted.Layers,
                layer => layer.TargetKey == "blocks.0.mlp.fc1.weight");
            Assert.Equal(new float[] { 3, 4, 1, 2 }, fc1.LoraUp.AsSpan<float>().ToArray());
        }
        finally
        {
            DisposeAll(source);
        }
    }

    [Fact]
    public void ConvertOfficial_FailsWhenAnyPddTensorWouldBeSkipped()
    {
        Dictionary<string, Tensor> source = CreateOfficialSource();
        Tensor removed = source["transformer_blocks.49.adaln_proj.linear.lora_up"];
        source.Remove("transformer_blocks.49.adaln_proj.linear.lora_up");
        try
        {
            HartsyInferenceException exception = Assert.Throws<HartsyInferenceException>(() =>
                MiniMaxH3PddKeyConverter.Convert(source, _headKeys, 64.0f));
            Assert.Contains("missing", exception.Message, StringComparison.OrdinalIgnoreCase);
        }
        finally
        {
            removed.Dispose();
            DisposeAll(source);
        }
    }

    private static Dictionary<string, Tensor> CreateOfficialSource()
    {
        Dictionary<string, Tensor> tensors = new(StringComparer.Ordinal);
        for (int i = 0; i < 50; i++)
        {
            string root = $"transformer_blocks.{i}";
            AddQkv(tensors, root, i == 0);
            AddPair(tensors, $"{root}.attn.to_out.0", 2, 2, [1, 1]);
            AddPair(tensors, $"{root}.ff.net.0.proj", 2, 4, i == 0 ? [1, 2, 3, 4] : [1, 1, 1, 1]);
            AddPair(tensors, $"{root}.ff.net.2", 2, 2, [1, 1]);
            AddPair(tensors, $"{root}.adaln_proj.linear", 2, 6, [1, 1, 1, 1, 1, 1]);
        }
        for (int i = 0; i < 2; i++)
        {
            string root = $"token_refiner.refiner_blocks.{i}";
            AddQkv(tensors, root, false);
            AddPair(tensors, $"{root}.attn.to_out.0", 2, 2, [1, 1]);
            AddPair(tensors, $"{root}.ff.net.0.proj", 2, 4, [1, 1, 1, 1]);
            AddPair(tensors, $"{root}.ff.net.2", 2, 2, [1, 1]);
        }
        foreach (string key in _headKeys) tensors[key] = Filled(new TensorShape(1), [0]);
        return tensors;
    }

    private static void AddQkv(Dictionary<string, Tensor> tensors, string root, bool known)
    {
        float[][] down = known ? [[1, 2], [3, 4], [5, 6]] : [[1, 1], [1, 1], [1, 1]];
        float[][] up = known ? [[10, 11], [20, 21], [30, 31]] : [[1, 1], [1, 1], [1, 1]];
        string[] names = ["q", "k", "v"];
        for (int i = 0; i < names.Length; i++)
        {
            tensors[$"{root}.attn.to_{names[i]}.lora_down"] = Filled(new TensorShape(1, 2), down[i]);
            tensors[$"{root}.attn.to_{names[i]}.lora_up"] = Filled(new TensorShape(2, 1), up[i]);
        }
    }

    private static void AddPair(Dictionary<string, Tensor> tensors, string root, int input, int output,
        float[] upValues)
    {
        tensors[root + ".lora_down"] = Filled(new TensorShape(1, input), Enumerable.Repeat(1.0f, input).ToArray());
        tensors[root + ".lora_up"] = Filled(new TensorShape(output, 1), upValues);
    }

    private static Tensor Filled(TensorShape shape, float[] values)
    {
        Tensor tensor = new Tensor(shape, DType.F32);
        values.CopyTo(tensor.AsSpan<float>());
        return tensor;
    }

    private static void DisposeAll(Dictionary<string, Tensor> tensors)
    {
        foreach (Tensor tensor in tensors.Values) tensor.Dispose();
    }
}
