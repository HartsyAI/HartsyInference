using System.Text;
using System.Text.Json;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Pipelines;
using HartsyInference.ThreeD.Tests.MeshPostProcess;

namespace HartsyInference.ThreeD.Tests.Part;

/// <summary>The checkpoint-free logic of <see cref="XPartPipeline"/>: the checkpoint check, the option check and the GLB of the parts.</summary>
public sealed class XPartPipelineTests
{
    [Fact]
    public void Constructor_NamesTheMissingCheckpointAndGenerateRejectsBadOptions()
    {
        string folder = Directory.CreateTempSubdirectory("xpart").FullName;
        try
        {
            using HartsyInference.Core.Backends.Cpu.CpuBackend backend = new();
            FileNotFoundException missing = Assert.Throws<FileNotFoundException>(() => new XPartPipeline(backend, folder));
            Assert.Contains("p3sam", missing.Message);

            foreach (string file in XPartPipeline.Files)
            {
                Directory.CreateDirectory(Path.Combine(folder, Path.GetDirectoryName(file)!));
                File.WriteAllBytes(Path.Combine(folder, file), [0]);
            }

            XPartPipeline pipeline = new(backend, folder);
            Mesh mesh = SyntheticMeshes.Icosphere(1);
            Assert.Throws<ArgumentOutOfRangeException>(() => pipeline.Generate(mesh, new XPartOptions { Steps = 0 }));
            Assert.Throws<ArgumentOutOfRangeException>(() => pipeline.Generate(mesh, new XPartOptions { BoxExtent = 0 }));
            Assert.Throws<ArgumentOutOfRangeException>(() => pipeline.Generate(mesh, new XPartOptions { QueryCount = 5000, PointCount = 4096 }));
            Assert.Throws<HartsyInference.Core.Exceptions.HartsyInferenceException>(() => pipeline.Generate(new Mesh { Vertices = [], Indices = [] }));
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    [Fact]
    public void Combine_WritesOneColouredNodePerPartWithTheirOwnFaces()
    {
        Mesh a = SyntheticMeshes.Icosphere(1), b = SyntheticMeshes.Icosphere(0);
        float[][] boxes = [[-1, -1, -1, 1, 1, 1], [-2, -2, -2, 2, 2, 2]];

        byte[] glb = XPartPipeline.Combine([a, b], boxes);

        Assert.Equal("glTF", Encoding.ASCII.GetString(glb, 0, 4));
        int jsonLength = BitConverter.ToInt32(glb, 12);
        using JsonDocument json = JsonDocument.Parse(Encoding.UTF8.GetString(glb, 20, jsonLength));
        JsonElement nodes = json.RootElement.GetProperty("nodes");
        Assert.Equal(2, nodes.GetArrayLength());
        Assert.Equal(a.TriangleCount, nodes[0].GetProperty("extras").GetProperty("faceCount").GetInt32());
        Assert.Equal(b.TriangleCount, nodes[1].GetProperty("extras").GetProperty("faceCount").GetInt32());
    }
}
