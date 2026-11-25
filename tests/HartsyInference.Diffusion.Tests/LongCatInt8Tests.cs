using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Memory.Vram.Streaming;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ModelAssets.Checkpoints;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The per-row INT8 release of the LongCat DiT: the loader, the policy that widens it for a backend without a resident int8 path, and the
/// forward it gives against the dense weights it was quantized from.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class LongCatInt8Tests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "longcat-int8-" + Guid.NewGuid().ToString("N"));
    private readonly LongCatConfig _config = LongCatTestData.Tiny;
    private readonly Dictionary<string, Tensor> _dense;
    private readonly Dictionary<string, Tensor> _quantized;

    public LongCatInt8Tests()
    {
        Directory.CreateDirectory(_directory);
        _dense = LongCatTestData.Weights(_config, 21);
        _quantized = LongCatTestData.QuantizePerRow(_dense);
        LongCatTestData.WriteCheckpoint(_directory, _quantized, indexFileName: LongCatCheckpoint.Int8IndexFileName);
    }

    public void Dispose()
    {
        foreach ((string key, Tensor t) in _quantized)
            if (!_dense.ContainsKey(key))
                t.Dispose();
        LongCatTestData.DisposeAll(_dense);
        Directory.Delete(_directory, recursive: true);
    }

    [Fact]
    public void LoaderReturnsDenseKeyNamesWithI8WeightsCarryingTheirRowScale()
    {
        using LongCatCheckpoint checkpoint = LongCatCheckpoint.Open(_directory);

        Assert.True(checkpoint.IsInt8);
        Assert.Equal(_config, checkpoint.Config);
        Assert.Equal(_dense.Keys.Order(), checkpoint.Weights.Keys.Order());
        foreach ((string key, Tensor tensor) in checkpoint.Weights)
        {
            bool linear = tensor.DType == DType.I8;
            Assert.Equal(linear, tensor.QuantInfo is not null);
            if (!linear)
                continue;
            Assert.Equal("int8_tensorwise", tensor.QuantInfo!.Format);
            Assert.Equal(0, tensor.QuantInfo.ConvRotGroupSize);
            Assert.Equal(tensor.Shape[0], tensor.QuantInfo.RowScale!.ElementCount);
            Assert.Equal(_dense[key].Shape, tensor.Shape);
        }
        Assert.Equal(DType.F32, checkpoint.Weights["final_layer.linear.weight"].DType);
        Assert.Equal(DType.F32, checkpoint.Weights["x_embedder.proj.weight"].DType);
        // Two timestep MLP linears, two caption linears, the final adaLN projection and nine linears per block; final_layer.linear stays dense.
        Assert.Equal(2 + 2 + 1 + _config.Depth * 9, checkpoint.Weights.Values.Count(static t => t.DType == DType.I8));
    }

    [Fact]
    public void WideningRestoresEachWeightWithinHalfAQuantizationStep()
    {
        using LongCatCheckpoint checkpoint = LongCatCheckpoint.Open(_directory);
        Dictionary<string, Tensor> weights = new(checkpoint.Weights, StringComparer.Ordinal);
        using CpuBackend backend = new();

        using QuantizedWeightPolicy.PreparedWeights prepared = QuantizedWeightPolicy.PrepareForBackends(weights, [backend], DType.F32);

        int checkedWeights = 0;
        foreach ((string key, Tensor wide) in weights)
        {
            if (checkpoint.Weights[key].DType != DType.I8)
                continue;
            Assert.Equal(DType.F32, wide.DType);
            ReadOnlySpan<float> got = wide.AsReadOnlySpan<float>(), original = _dense[key].AsReadOnlySpan<float>();
            long cols = wide.Shape[1];
            for (long r = 0; r < wide.Shape[0]; r++)
            {
                float max = 0f;
                for (long c = 0; c < cols; c++)
                    max = MathF.Max(max, MathF.Abs(original[(int)(r * cols + c)]));
                // Half a step (max / 254) plus the bf16 rounding of the policy's widening, which rounds the scale and the product.
                float tolerance = max / 254f + max * 0.0125f + 1e-9f;
                for (long c = 0; c < cols; c++)
                    Assert.True(MathF.Abs(got[(int)(r * cols + c)] - original[(int)(r * cols + c)]) <= tolerance, $"{key}[{r},{c}]");
            }
            checkedWeights++;
        }
        Assert.Equal(2 + 2 + 1 + _config.Depth * 9, checkedWeights);
    }

    [Fact]
    public void Int8ForwardTracksTheDenseForward()
    {
        using LongCatCheckpoint checkpoint = LongCatCheckpoint.Open(_directory);
        Dictionary<string, Tensor> weights = new(checkpoint.Weights, StringComparer.Ordinal);
        using CpuBackend backend = new();
        using QuantizedWeightPolicy.PreparedWeights prepared = QuantizedWeightPolicy.PrepareForBackends(weights, [backend], DType.F32);
        using Tensor latent = LongCatTestData.Latent(_config, 5, 4, 4, 2);
        using Tensor text = LongCatTestData.Encoder(_config, 5, 3);
        LongCatConditioning conditioning = new(NumCondLatents: 1);

        float[] reference = Forward(backend, _dense, latent, text, conditioning);
        float[] int8 = Forward(backend, weights, latent, text, conditioning);

        double error = LongCatTestData.RelRms(int8, reference);
        Assert.True(error < 1.5e-2, $"int8 forward drifted {error} from the dense forward (measured 6.0e-3 on this fixture)");
        Assert.True(error > 1e-6, "an int8 forward should not equal the dense one");
    }

    [Fact]
    public void Int8BlocksAreAQuarterOfTheDenseBlockBytesAndStayStreamable()
    {
        using LongCatCheckpoint checkpoint = LongCatCheckpoint.Open(_directory);
        using LongCatTransformer int8 = new(_config);
        int8.LoadWeights(checkpoint.Weights);
        using LongCatTransformer dense = new(_config);
        dense.LoadWeights(_dense);
        IStreamableDenoiser streamed = int8;

        long int8Bytes = streamed.GetBlock(0).EstimatedWeightBytes, denseBytes = ((IStreamableDenoiser)dense).GetBlock(0).EstimatedWeightBytes;

        Assert.True(int8Bytes < denseBytes / 3, $"{int8Bytes} int8 bytes against {denseBytes} dense");
        Assert.Equal(21, streamed.GetBlock(0).EnumerateWeights().Count());
    }

    private float[] Forward(CpuBackend backend, IReadOnlyDictionary<string, Tensor> weights, Tensor latent, Tensor text,
        LongCatConditioning conditioning)
    {
        using LongCatTransformer transformer = new(_config);
        transformer.LoadWeights(weights);
        using Tensor velocity = transformer.Forward(backend, latent, text, conditioning.FrameTimesteps(300f, 5), in conditioning);
        return velocity.AsReadOnlySpan<float>().ToArray();
    }
}
