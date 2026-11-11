using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ThreeD.Models.TrellisV2;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests.Pixal3D;

/// <summary>Header-only checks of the downloaded Pixal3D flows against the key map: no tensor data is read. Integration lane; skipped, not
/// passed, when the files are absent. They show names, shapes and dtypes match; they are not parity.</summary>
[Trait("Category", "Integration")]
public sealed class Pixal3DRealWeightsTests
{
    private const string RopePhases = "rope_phases";

    private readonly ITestOutputHelper _output;

    public Pixal3DRealWeightsTests(ITestOutputHelper output) => _output = output;

    [Pixal3DRealWeightsFact]
    public void FlowHeaders_MatchTheProjectionKeyMapExactly()
    {
        string folder = Pixal3DRealWeightsFactAttribute.FindFolder()!;
        foreach (string name in new[]
        {
            "ss_flow_img_dit_1_3B_64_bf16", "slat_flow_img2shape_dit_1_3B_512_bf16", "slat_flow_img2shape_dit_1_3B_1024_bf16",
        })
        {
            string path = Path.Combine(folder, name + ".safetensors");
            if (!File.Exists(path))
                continue;
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
}
