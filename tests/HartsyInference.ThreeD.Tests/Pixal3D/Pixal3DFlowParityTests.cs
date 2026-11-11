using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.TrellisV2;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests.Pixal3D;

/// <summary>The real Pixal3D flows against upstream's own torch code: a forward over the dumps of <c>tests/python-reference/dump_pixal3d_structure_flow.py</c>
/// and <c>dump_pixal3d_slat_flows.py</c> (float32 on the CPU, the real checkpoints, random but fixed inputs), compared with this engine's CPU backend. Integration
/// lane; skipped, not passed, unless <c>PIXAL3D_FLOW_DUMPS</c> names a folder holding the dump folders <c>out_ss</c> and <c>out_slat</c> and the flows are installed.
/// The structure flow takes about 6 minutes on the CPU, each sparse flow about 1 minute.</summary>
[Trait("Category", "Integration")]
public sealed class Pixal3DFlowParityTests
{
    private const double MaxAbsTolerance = 2e-4, RelL2Tolerance = 1e-4;
    private const string StructureFlow = "ss_flow_img_dit_1_3B_64_bf16";
    private const string ShapeFlow1024 = "slat_flow_img2shape_dit_1_3B_1024_bf16";
    private const string TextureFlow1024 = "slat_flow_imgshape2tex_dit_1_3B_1024_bf16";

    private readonly ITestOutputHelper _output;

    public Pixal3DFlowParityTests(ITestOutputHelper output) => _output = output;

    private static string? Dumps => Environment.GetEnvironmentVariable("PIXAL3D_FLOW_DUMPS");

    [Pixal3DRealWeightsFact(StructureFlow + ".safetensors", StructureFlow + ".json")]
    public void StructureFlow_MatchesUpstreamTorchOnTheCpu()
    {
        string? dumps = Dumps;
        if (dumps is null || !Directory.Exists(Path.Combine(dumps, "out_ss")))
            return;
        string dir = Path.Combine(dumps, "out_ss");
        using CpuBackend backend = new();
        using Trellis2FlowWeights weights = Trellis2FlowWeights.Load(Path.Combine(Pixal3DRealWeightsFactAttribute.FindFolder()!, StructureFlow + ".safetensors"));
        using Trellis2SparseStructureFlow flow = new(weights);
        using Tensor x = Load(dir, "x", 1, 8, 16, 16, 16), global = Load(dir, "global", 1, 5, 1024), proj = Load(dir, "proj", 1, 4096, 1024);

        using Tensor y = flow.Forward(backend, x, Floats(dir, "t")[0], global, proj);

        AssertClose(y.AsReadOnlySpan<float>(), Floats(dir, "out"), "structure flow");
    }

    [Pixal3DRealWeightsFact(ShapeFlow1024 + ".safetensors", ShapeFlow1024 + ".json")]
    public void ShapeFlow1024_MatchesUpstreamTorchOnTheCpu()
    {
        if (!TryOpen("out_slat", out string dir, out int[] coords))
            return;
        int n = coords.Length / 4;
        using CpuBackend backend = new();
        using Trellis2FlowWeights weights = Trellis2FlowWeights.Load(Path.Combine(Pixal3DRealWeightsFactAttribute.FindFolder()!, ShapeFlow1024 + ".safetensors"));
        using Trellis2SlatFlow flow = new(weights);
        using Tensor x = Load(dir, "shape_x", 1, n, 32), global = Load(dir, "global", 1, 5, 1024), proj = Load(dir, "proj", 1, n, 2048);

        using Tensor y = flow.Forward(backend, x, coords, Floats(dir, "t")[0], global, proj);

        AssertClose(y.AsReadOnlySpan<float>(), Floats(dir, "shape_out"), "1024 shape flow");
    }

    [Pixal3DRealWeightsFact(TextureFlow1024 + ".safetensors", TextureFlow1024 + ".json")]
    public void TextureFlow1024_MatchesUpstreamTorchOnTheCpu()
    {
        if (!TryOpen("out_slat", out string dir, out int[] coords))
            return;
        int n = coords.Length / 4;
        using CpuBackend backend = new();
        using Trellis2FlowWeights weights = Trellis2FlowWeights.Load(Path.Combine(Pixal3DRealWeightsFactAttribute.FindFolder()!, TextureFlow1024 + ".safetensors"));
        using Trellis2TextureFlow flow = new(weights);
        using Tensor x = Load(dir, "texture_x", 1, n, 32), concat = Load(dir, "texture_concat", 1, n, 32);
        using Tensor global = Load(dir, "global", 1, 5, 1024), proj = Load(dir, "proj", 1, n, 2048);

        using Tensor y = flow.Forward(backend, x, concat, coords, Floats(dir, "t")[0], global, proj);

        AssertClose(y.AsReadOnlySpan<float>(), Floats(dir, "texture_out"), "1024 texture flow");
    }

    private static bool TryOpen(string folder, out string dir, out int[] coords)
    {
        dir = Dumps is null ? string.Empty : Path.Combine(Dumps, folder);
        coords = [];
        if (!Directory.Exists(dir))
            return false;
        byte[] raw = File.ReadAllBytes(Path.Combine(dir, "coords.bin"));
        coords = new int[raw.Length / 4];
        Buffer.BlockCopy(raw, 0, coords, 0, raw.Length);
        return true;
    }

    private static float[] Floats(string dir, string name)
    {
        byte[] raw = File.ReadAllBytes(Path.Combine(dir, name + ".bin"));
        float[] values = new float[raw.Length / 4];
        Buffer.BlockCopy(raw, 0, values, 0, raw.Length);
        return values;
    }

    private static Tensor Load(string dir, string name, params long[] shape)
    {
        Tensor tensor = new(new TensorShape(shape), DType.F32);
        Floats(dir, name).CopyTo(tensor.AsSpan<float>());
        return tensor;
    }

    private void AssertClose(ReadOnlySpan<float> actual, float[] expected, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        double maxAbs = 0, errSq = 0, refSq = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double d = actual[i] - expected[i];
            maxAbs = Math.Max(maxAbs, Math.Abs(d));
            errSq += d * d;
            refSq += expected[i] * (double)expected[i];
        }

        double rel = Math.Sqrt(errSq / refSq);
        _output.WriteLine($"{what}: max|diff| {maxAbs:E3}, relative L2 {rel:E3}");
        Assert.True(maxAbs < MaxAbsTolerance, $"{what}: max|diff| {maxAbs:E3}");
        Assert.True(rel < RelL2Tolerance, $"{what}: relative L2 {rel:E3}");
    }
}
