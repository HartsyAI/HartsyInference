using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Diffusion.Tests;

/// <summary>A tiny LongCat LoRA written in the reference's own state-dict layout
/// (<c>lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___attn___lorahyphen___qkv.lora_down.weight</c>,
/// split <c>lora_up.blocks.i</c> on the fused projections, an <c>alpha_scale</c> scalar), plus the merged-weight delta computed
/// straight from the reference's
/// forward (<c>y += strength · alpha_scale · up(chunk_i(down(x)))</c>) rather than through the mapper.</summary>
internal sealed class LongCatLoraFixture : IDisposable
{
    public const int Rank = 4;

    private readonly Dictionary<string, Tensor> _tensors = new(StringComparer.Ordinal);
    private readonly Dictionary<string, (Tensor Down, Tensor[] Ups)> _modules = new(StringComparer.Ordinal);

    public LongCatLoraFixture(string path, LongCatConfig config, int seed, float alphaScale = 0.5f, float firstBlockAlphaScale = float.NaN)
    {
        AlphaScale = alphaScale;
        int h = config.HiddenSize, f = config.FfnHiddenDim;
        for (int b = 0; b < config.Depth; b++)
        {
            float scale = b == 0 && !float.IsNaN(firstBlockAlphaScale) ? firstBlockAlphaScale : alphaScale;
            (string Name, int In, int[] Outs)[] specs =
            [
                ("attn.qkv", h, [h, h, h]), ("attn.proj", h, [h]), ("cross_attn.q_linear", h, [h]), ("cross_attn.kv_linear", h, [h, h]),
                ("ffn.w1", h, [f]), ("ffn.w2", f, [h]), ("ffn.w3", h, [f]),
            ];
            int salt = 0;
            foreach ((string name, int input, int[] outs) in specs)
            {
                string module = $"blocks.{b}.{name}";
                Tensor down = LongCatTestData.Random(new TensorShape(outs.Length * Rank, input), seed + 31 * b + salt++, 0.4f);
                Tensor[] ups = outs.Select((o, i) =>
                    LongCatTestData.Random(new TensorShape(o, Rank), seed + 977 * b + 13 * salt + i, 0.4f)).ToArray();
                _modules[module] = (down, ups);
                string key = "lora___lorahyphen___" + module.Replace(".", "___lorahyphen___", StringComparison.Ordinal);
                _tensors[key + ".lora_down.weight"] = down;
                if (outs.Length == 1)
                    _tensors[key + ".lora_up.weight"] = ups[0];
                else
                    for (int i = 0; i < outs.Length; i++)
                        _tensors[$"{key}.lora_up.blocks.{i}.weight"] = ups[i];
                Tensor alpha = new(new TensorShape(Array.Empty<long>()), DType.F32);
                alpha.AsSpan<float>()[0] = scale;
                _tensors[key + ".alpha_scale"] = alpha;
            }
        }
        SafeTensorsWriter.Save(path, _tensors);
        Path = path;
    }

    public string Path { get; }

    public float AlphaScale { get; }

    public IEnumerable<string> Modules => _modules.Keys;

    public Tensor Alpha(string module) =>
        _tensors["lora___lorahyphen___" + module.Replace(".", "___lorahyphen___", StringComparison.Ordinal) + ".alpha_scale"];

    /// <summary>The weight delta for <paramref name="module"/> at <paramref name="strength"/>, from the reference's chunked
    /// forward: rows of up block <c>i</c> read
    /// chunk <c>i</c> of the down output.</summary>
    public float[] Delta(string module, float strength)
    {
        (Tensor down, Tensor[] ups) = _modules[module];
        float scale = strength * Alpha(module).AsReadOnlySpan<float>()[0];
        int input = (int)down.Shape[1];
        int rows = ups.Sum(static u => (int)u.Shape[0]);
        float[] delta = new float[rows * input];
        ReadOnlySpan<float> d = down.AsReadOnlySpan<float>();
        int rowOffset = 0;
        for (int i = 0; i < ups.Length; i++)
        {
            ReadOnlySpan<float> u = ups[i].AsReadOnlySpan<float>();
            int outRows = (int)ups[i].Shape[0];
            for (int r = 0; r < outRows; r++)
                for (int c = 0; c < input; c++)
                {
                    double sum = 0;
                    for (int k = 0; k < Rank; k++)
                        sum += u[r * Rank + k] * (double)d[(i * Rank + k) * input + c];
                    delta[(rowOffset + r) * input + c] = (float)(scale * sum);
                }
            rowOffset += outRows;
        }
        return delta;
    }

    /// <summary>A copy of <paramref name="weights"/> with every adapted weight replaced by the weight plus this LoRA's delta at
    /// <paramref name="strength"/>.</summary>
    public Dictionary<string, Tensor> MergedInto(IReadOnlyDictionary<string, Tensor> weights, float strength)
    {
        Dictionary<string, Tensor> merged = new(weights, StringComparer.Ordinal);
        foreach (string module in Modules)
        {
            string key = module + ".weight";
            Tensor baseWeight = weights[key];
            Tensor result = new(baseWeight.Shape, DType.F32);
            ReadOnlySpan<float> w = baseWeight.AsReadOnlySpan<float>();
            float[] delta = Delta(module, strength);
            Span<float> output = result.AsSpan<float>();
            for (int i = 0; i < output.Length; i++)
                output[i] = w[i] + delta[i];
            merged[key] = result;
        }
        return merged;
    }

    public void Dispose()
    {
        foreach (Tensor t in _tensors.Values)
            t.Dispose();
    }
}
