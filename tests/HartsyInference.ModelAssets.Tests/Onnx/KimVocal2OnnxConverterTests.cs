using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Onnx;
using Xunit;

namespace HartsyInference.ModelAssets.Tests.Onnx;

public sealed unsafe class KimVocal2OnnxConverterTests
{
    private static Dictionary<string, Tensor> Load(TinyKimGraph graph, out KimVocal2Geometry geometry, out List<Tensor> owners)
    {
        OnnxModel model = OnnxModel.Parse(graph.Build());
        Dictionary<string, Tensor> inits = [];
        owners = [];
        foreach (OnnxTensor t in model.Initializers)
        {
            Tensor tensor = new(new TensorShape(t.Dims), DType.F32);
            float[] values = graph.Values[t.Name];
            fixed (float* p = values) Buffer.MemoryCopy(p, (void*)tensor.DataPointer, values.Length * 4L, values.Length * 4L);
            inits[t.Name] = tensor;
            owners.Add(tensor);
        }
        return KimVocal2OnnxConverter.Convert(model, inits, out geometry);
    }

    [Fact]
    public void NodeAttributesAreParsed()
    {
        TinyKimGraph graph = new(1);
        OnnxModel model = OnnxModel.Parse(graph.Build());
        OnnxNode conv = model.Nodes.First(n => n.OpType == "Conv" && n.IntAttributes["strides"][0] == 2);
        Assert.Equal([2L, 2L], conv.IntAttributes["kernel_shape"]);
        Assert.Equal([0L, 0L, 0L, 0L], conv.IntAttributes["pads"]);
        Assert.Equal([1L], conv.IntAttributes["group"]);
        OnnxNode bn = model.Nodes.First(n => n.OpType == "BatchNormalization");
        Assert.Equal(TinyKimGraph.BnEpsilon, bn.FloatAttributes["epsilon"]);
        Assert.Empty(model.Nodes.First(n => n.OpType == "Relu").IntAttributes);
    }

    [Fact]
    public void WalkBindsEveryLayerAndReadsTheGeometry()
    {
        TinyKimGraph graph = new(2);
        Dictionary<string, Tensor> bound = Load(graph, out KimVocal2Geometry geometry, out List<Tensor> owners);
        try
        {
            Assert.Equal(TinyKimGraph.Widths, geometry.Widths);
            Assert.Equal(TinyKimGraph.DimF, geometry.DimF);
            Assert.Equal(TinyKimGraph.InChannels, geometry.InChannels);
            Assert.Equal(2, geometry.Levels);
            string[] expectedBlocks = ["encoding_blocks.0", "encoding_blocks.1", "bottleneck_block", "decoding_blocks.0", "decoding_blocks.1"];
            foreach (string block in expectedBlocks)
            {
                for (int j = 0; j < 3; j++) Assert.Contains($"{block}.tfc.{j}.weight", bound.Keys);
                foreach (string k in new[] { "tdf.0.weight", "tdf.1.scale", "tdf.1.shift", "tdf.2.weight", "tdf.3.scale", "tdf.3.shift" })
                    Assert.Contains($"{block}.{k}", bound.Keys);
            }
            // 2 first/final convs x2 + 5 blocks x (3 tfc convs x2 + 2 matmul + 4 bn vectors) + 2 downs x2 + 2 ups x2.
            Assert.Equal(4 + 5 * (6 + 2 + 4) + 4 + 4, bound.Count);
            Assert.Equal(new TensorShape(2, 4, 1, 1), bound["first_conv.weight"].Shape);
            Assert.Equal(new TensorShape(3, 2, 2, 2), bound["encoding_blocks.0.down.weight"].Shape);
            Assert.Equal(new TensorShape(3, 2, 2, 2), bound["decoding_blocks.1.up.weight"].Shape);
        }
        finally
        {
            foreach (Tensor t in bound.Values) t.Dispose();
            foreach (Tensor t in owners) t.Dispose();
        }
    }

    [Fact]
    public void MatMulWeightIsTransposedToOutByIn()
    {
        TinyKimGraph graph = new(3);
        Dictionary<string, Tensor> bound = Load(graph, out _, out List<Tensor> owners);
        try
        {
            float[] file = graph.Values[graph.FirstMatMulWeight];          // [in=8, out=4]
            Tensor linear = bound["encoding_blocks.0.tdf.0.weight"];        // [out=4, in=8]
            Assert.Equal(new TensorShape(4, 8), linear.Shape);
            float* p = (float*)linear.DataPointer;
            for (int o = 0; o < 4; o++)
                for (int i = 0; i < 8; i++) Assert.Equal(file[i * 4 + o], p[o * 8 + i]);
        }
        finally
        {
            foreach (Tensor t in bound.Values) t.Dispose();
            foreach (Tensor t in owners) t.Dispose();
        }
    }

    [Fact]
    public void BatchNormBecomesAffineUsingTheNodesEpsilon()
    {
        TinyKimGraph graph = new(4);
        Dictionary<string, Tensor> bound = Load(graph, out _, out List<Tensor> owners);
        try
        {
            string p = graph.FirstBnPrefix;
            float[] gamma = graph.Values[p + ".weight"], beta = graph.Values[p + ".bias"];
            float[] mean = graph.Values[p + ".running_mean"], variance = graph.Values[p + ".running_var"];
            float* scale = (float*)bound["encoding_blocks.0.tdf.1.scale"].DataPointer;
            float* shift = (float*)bound["encoding_blocks.0.tdf.1.shift"].DataPointer;
            for (int c = 0; c < gamma.Length; c++)
            {
                float s = gamma[c] / MathF.Sqrt(variance[c] + TinyKimGraph.BnEpsilon);
                Assert.Equal(s, scale[c], 6);
                Assert.Equal(beta[c] - mean[c] * s, shift[c], 6);
            }
        }
        finally
        {
            foreach (Tensor t in bound.Values) t.Dispose();
            foreach (Tensor t in owners) t.Dispose();
        }
    }

    [Fact]
    public void ConvTransposeBatchNormFoldsAlongTheOutputChannelAxis()
    {
        TinyKimGraph graph = new(5);
        Dictionary<string, Tensor> bound = Load(graph, out _, out List<Tensor> owners);
        try
        {
            string bn = graph.FirstUpPrefix + ".1";
            float[] gamma = graph.Values[bn + ".weight"], beta = graph.Values[bn + ".bias"];
            float[] mean = graph.Values[bn + ".running_mean"], variance = graph.Values[bn + ".running_var"];
            float[] w = graph.Values[graph.FirstUpWeight];                  // [cin=4, cout=3, 2, 2]
            float[] b = graph.Values[graph.FirstUpPrefix + ".0.bias"];
            Tensor fw = bound["decoding_blocks.0.up.weight"], fb = bound["decoding_blocks.0.up.bias"];
            Assert.Equal(new TensorShape(4, 3, 2, 2), fw.Shape);
            float* pw = (float*)fw.DataPointer, pb = (float*)fb.DataPointer;
            for (int co = 0; co < 3; co++)
            {
                float s = gamma[co] / MathF.Sqrt(variance[co] + TinyKimGraph.BnEpsilon);
                Assert.Equal(b[co] * s + beta[co] - mean[co] * s, pb[co], 5);
                for (int ci = 0; ci < 4; ci++)
                    for (int k = 0; k < 4; k++)
                        Assert.Equal(w[(ci * 3 + co) * 4 + k] * s, pw[(ci * 3 + co) * 4 + k], 6);
            }
        }
        finally
        {
            foreach (Tensor t in bound.Values) t.Dispose();
            foreach (Tensor t in owners) t.Dispose();
        }
    }

    [Fact]
    public void MiswiredSkipFailsTheLoad()
    {
        TinyKimGraph graph = new(6, swapSkip: true);
        InvalidDataException ex = Assert.Throws<InvalidDataException>(() => Load(graph, out _, out _));
        Assert.Contains("multiplicative skip", ex.Message);
    }
}
