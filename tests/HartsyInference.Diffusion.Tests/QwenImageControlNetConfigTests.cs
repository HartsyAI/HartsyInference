using System.Text.Json;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Adapters;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary><see cref="QwenImageControlNetConfig.FromDescriptors"/> on header-only descriptors: the Union layout derives its
/// geometry, and an inpainting ControlNet (a control input wider than the packed latent) is refused by name rather than failing
/// on a shape mismatch inside the first projection.</summary>
public sealed class QwenImageControlNetConfigTests
{
    [Fact]
    public void UnionLayout_DerivesTheGeometryFromTheHeader()
    {
        QwenImageControlNetConfig config = QwenImageControlNetConfig.FromDescriptors(Header(controlInDim: 64, depth: 5));

        Assert.Equal(3072, config.HiddenSize);
        Assert.Equal(64, config.InDim);
        Assert.Equal(5, config.Depth);
        Assert.Equal(128, config.HeadDim);
        Assert.Equal(24, config.NumHeads);
        Assert.Equal(3584, config.ContextDim);
        Assert.Equal(12288, config.MlpDim);
    }

    [Fact]
    public void InpaintingLayout_IsRefusedNamingTheExtraChannels()
    {
        NotSupportedException ex = Assert.Throws<NotSupportedException>(() =>
            QwenImageControlNetConfig.FromDescriptors(Header(controlInDim: 68, depth: 6)));

        Assert.Contains("68 control channels", ex.Message);
        Assert.Contains("4 more", ex.Message);
        Assert.Contains("inpainting", ex.Message);
    }

    [Fact]
    public void RealUnionHeader_DerivesItsGeometry()
    {
        QwenImageControlNetConfig config = QwenImageControlNetConfig.FromDescriptors(ReadHeader("union.header.json"));

        Assert.Equal(3072, config.HiddenSize);
        Assert.Equal(64, config.InDim);
        Assert.Equal(5, config.Depth);
    }

    [Fact]
    public void RealInpaintingHeader_IsRefusedNamingTheExtraChannels()
    {
        NotSupportedException ex = Assert.Throws<NotSupportedException>(() =>
            QwenImageControlNetConfig.FromDescriptors(ReadHeader("inpainting.header.json")));

        Assert.Contains("68 control channels", ex.Message);
    }

    private static Dictionary<string, SafeTensorDescriptor> Header(int controlInDim, int depth)
    {
        Dictionary<string, SafeTensorDescriptor> d = new(StringComparer.Ordinal);
        void Add(string name, params long[] dims) =>
            d[name] = new SafeTensorDescriptor { Name = name, DType = DType.BF16, Shape = new TensorShape(dims), DataOffset = 0, ByteLength = 0 };

        Add("img_in.weight", 3072, 64);
        Add("controlnet_x_embedder.weight", 3072, controlInDim);
        Add("txt_in.weight", 3072, 3584);
        for (int i = 0; i < depth; i++)
        {
            Add($"transformer_blocks.{i}.attn.norm_q.weight", 128);
            Add($"transformer_blocks.{i}.img_mlp.net.0.proj.weight", 12288, 3072);
        }
        return d;
    }

    /// <summary>The tensor table of a released checkpoint's safetensors header, read from its committed JSON.</summary>
    private static Dictionary<string, SafeTensorDescriptor> ReadHeader(string fileName)
    {
        string path = Path.Combine(AppContext.BaseDirectory, "Fixtures", "QwenImageControlNet", fileName);
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));
        Dictionary<string, SafeTensorDescriptor> d = new(StringComparer.Ordinal);
        foreach (JsonProperty entry in document.RootElement.EnumerateObject())
        {
            if (entry.Name == "__metadata__")
            {
                continue;
            }
            Assert.True(SafeTensorDTypes.TryParse(entry.Value.GetProperty("dtype").GetString()!, out DType dtype));
            long[] dims = entry.Value.GetProperty("shape").EnumerateArray().Select(dim => dim.GetInt64()).ToArray();
            d[entry.Name] = new SafeTensorDescriptor { Name = entry.Name, DType = dtype, Shape = new TensorShape(dims), DataOffset = 0, ByteLength = 0 };
        }
        return d;
    }
}
