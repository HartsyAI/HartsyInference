using System.Buffers.Binary;
using System.Text;
using HartsyInference.ModelAssets.Checkpoints;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>The header snapshot variant resolution reads. A probe that misses a key, a shape or a metadata entry
/// resolves the wrong variant with no error, so each field it promises is pinned here.</summary>
public sealed class CheckpointProbeTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("probe-").FullName;

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    [Fact]
    public void Read_HeaderOnlySafeTensors_ReturnsKeysShapesMetadataAndName()
    {
        string path = WriteHeader("qwen_image_edit_fp8.safetensors", """{"modelspec.architecture":"qwen-image-edit"}""",
            ("transformer_blocks.0.attn.to_q.weight", "[3072,3072]"), ("__index_timestep_zero__", "[]"));
        CheckpointProbe probe = CheckpointProbe.Read(path);
        Assert.Contains("transformer_blocks.0.attn.to_q.weight", probe.Keys);
        Assert.Contains("__index_timestep_zero__", probe.Keys);
        Assert.DoesNotContain("__metadata__", probe.Keys);
        Assert.Equal(3072, probe.Shapes["transformer_blocks.0.attn.to_q.weight"][1]);
        Assert.Equal("qwen-image-edit", probe.GetMetadata("modelspec.architecture"));
        Assert.Equal(["qwen_image_edit_fp8"], probe.FileNames);
    }

    /// <summary>A capability query must never throw on a missing or unreadable file; it answers "no evidence".</summary>
    [Fact]
    public void Read_MissingOrGarbage_ReturnsEmptyEvidence()
    {
        Assert.Same(CheckpointProbe.Empty, CheckpointProbe.Read(Path.Combine(_dir, "missing.safetensors")));
        Assert.Same(CheckpointProbe.Empty, CheckpointProbe.Read(null));
        string garbage = Path.Combine(_dir, "garbage.safetensors");
        File.WriteAllText(garbage, "not a checkpoint");
        CheckpointProbe probe = CheckpointProbe.Read(garbage);
        Assert.Empty(probe.Keys);
        Assert.Equal(["garbage"], probe.FileNames);
    }

    private string WriteHeader(string fileName, string? metadataJson, params (string Key, string Shape)[] tensors)
    {
        StringBuilder header = new StringBuilder("{");
        if (metadataJson is not null)
        {
            header.Append("\"__metadata__\":").Append(metadataJson).Append(',');
        }
        header.Append(string.Join(",", tensors.Select(t => $"\"{t.Key}\":{{\"dtype\":\"F32\",\"shape\":{t.Shape},\"data_offsets\":[0,0]}}")));
        header.Append('}');
        byte[] json = Encoding.UTF8.GetBytes(header.ToString());
        string path = Path.Combine(_dir, fileName);
        using FileStream stream = File.Create(path);
        Span<byte> length = stackalloc byte[8];
        BinaryPrimitives.WriteInt64LittleEndian(length, json.Length);
        stream.Write(length);
        stream.Write(json);
        return path;
    }
}
