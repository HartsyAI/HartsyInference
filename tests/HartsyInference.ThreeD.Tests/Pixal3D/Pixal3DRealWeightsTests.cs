using System.Runtime.InteropServices;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ThreeD.Models.TrellisV2;
using Microsoft.Win32.SafeHandles;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests.Pixal3D;

/// <summary>Header-only checks of the downloaded Pixal3D flows against the key map: no tensor data is read. Integration lane; skipped, not
/// passed, when the files are absent. They show names, shapes and dtypes match; they are not parity.</summary>
[Trait("Category", "Integration")]
public sealed class Pixal3DRealWeightsTests
{
    private const string RopePhases = "rope_phases";
    private const string StructureFlow = "ss_flow_img_dit_1_3B_64_bf16";
    private const string ShapeFlow512 = "slat_flow_img2shape_dit_1_3B_512_bf16";
    private const string ShapeFlow1024 = "slat_flow_img2shape_dit_1_3B_1024_bf16";

    private readonly ITestOutputHelper _output;

    public Pixal3DRealWeightsTests(ITestOutputHelper output) => _output = output;

    [Pixal3DRealWeightsFact(
        StructureFlow + ".safetensors", StructureFlow + ".json", ShapeFlow512 + ".safetensors", ShapeFlow512 + ".json",
        ShapeFlow1024 + ".safetensors", ShapeFlow1024 + ".json")]
    public void FlowHeaders_MatchTheProjectionKeyMapExactly()
    {
        string folder = Pixal3DRealWeightsFactAttribute.FindFolder()!;
        foreach (string name in new[] { StructureFlow, ShapeFlow512, ShapeFlow1024 })
        {
            string path = Path.Combine(folder, name + ".safetensors");
            Trellis2FlowConfig config = Trellis2FlowConfig.FromJson(File.ReadAllText(Path.ChangeExtension(path, ".json")));
            SafeTensorHeader header = SafeTensorHeaderReader.Read(path);
            IReadOnlyList<ShapeVaeKeyMap.TensorSpec> specs = Trellis2FlowKeyMap.Specs(config);

            string[] expected = [.. specs.Select(s => s.CheckpointKey)];
            string[] actual = [.. header.Tensors.Keys.Where(k => k != RopePhases)];
            Assert.Equal(expected.Order(StringComparer.Ordinal), actual.Order(StringComparer.Ordinal));
            foreach (ShapeVaeKeyMap.TensorSpec spec in specs)
            {
                SafeTensorDescriptor tensor = header.Tensors[spec.CheckpointKey];
                Assert.Equal(DType.F32, tensor.DType);
                Assert.Equal(new TensorShape(spec.Shape), tensor.Shape);
            }

            Assert.Equal(config.Dense, header.Tensors.TryGetValue(RopePhases, out SafeTensorDescriptor? phases));
            if (phases is not null)
                Assert.Equal(DType.C64, phases.DType);
            _output.WriteLine($"{name}: {specs.Count} tensors match ({(config.Dense ? "dense, plus rope_phases C64" : "sparse")}).");
        }
    }

    [Pixal3DRealWeightsFact(StructureFlow + ".safetensors", StructureFlow + ".json")]
    public void StoredRopePhases_MatchTheRebuiltDenseRopeTable()
    {
        // The loader leaves the saved complex buffer unread because Trellis2RopeTable rebuilds it; this proves the two agree.
        string path = Path.Combine(Pixal3DRealWeightsFactAttribute.FindFolder()!, StructureFlow + ".safetensors");
        Trellis2FlowConfig config = Trellis2FlowConfig.FromJson(File.ReadAllText(Path.ChangeExtension(path, ".json")));
        SafeTensorDescriptor phases = SafeTensorHeaderReader.Read(path).Tensors[RopePhases];
        int tokens = config.DenseTokens, pairs = config.HeadDim / 2;
        Assert.Equal(new TensorShape(tokens, pairs), phases.Shape);
        float[] stored = new float[tokens * pairs * 2];
        using (SafeFileHandle handle = File.OpenHandle(path))
        {
            Assert.Equal(phases.ByteLength, RandomAccess.Read(handle, MemoryMarshal.AsBytes(stored.AsSpan()), phases.DataOffset));
        }

        using Trellis2RopeTable table = Trellis2RopeTable.Build(config.HeadDim, Trellis2RopeTable.DenseGridPositions(config.Resolution), tokens);
        ReadOnlySpan<float> cos = table.Cos.AsReadOnlySpan<float>(), sin = table.Sin.AsReadOnlySpan<float>();
        int rotated = (int)table.Cos.Shape[2];
        double worst = 0;
        for (int t = 0; t < tokens; t++)
        {
            for (int p = 0; p < pairs; p++)
            {
                // Upstream pads the 3f rotated pairs with zero phases, so the extra pairs hold exp(0) = 1 + 0i.
                float expectedRe = p < rotated ? cos[t * rotated + p] : 1f, expectedIm = p < rotated ? sin[t * rotated + p] : 0f;
                worst = Math.Max(worst, Math.Max(Math.Abs(stored[(t * pairs + p) * 2] - expectedRe), Math.Abs(stored[(t * pairs + p) * 2 + 1] - expectedIm)));
            }
        }

        _output.WriteLine($"rope_phases vs rebuilt table over {tokens} tokens x {pairs} pairs: worst absolute difference {worst:E2}.");
        Assert.True(worst < 1e-5, $"The saved RoPE phases differ from the rebuilt table by {worst:E2}.");
    }
}
