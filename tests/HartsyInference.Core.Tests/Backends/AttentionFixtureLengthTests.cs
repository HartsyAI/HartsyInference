using System.Security.Cryptography;
using System.Text.Json;
using HartsyInference.GpuBenchmarks.Attention;
using Xunit;

namespace HartsyInference.Core.Tests.Backends;

/// <summary>Shared fixtures must not turn per-sequence visibility into full-capacity attention.</summary>
public sealed class AttentionFixtureLengthTests
{
    [Theory]
    [InlineData(0, 53, 73)]
    [InlineData(72, 72, 72)]
    public void RejectsLengthsThatTheScalarAdapterCannotHonor(int first, int second, int third)
    {
        WithFixture([first, second, third], path => Assert.Throws<NotSupportedException>(() => AttentionFixture.Load(path)));
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void AcceptsScalarOrMatchingUniformLengths(bool includeArray)
    {
        WithFixture(includeArray ? [73, 73, 73] : null, path =>
        {
            using AttentionFixture fixture = AttentionFixture.Load(path);
            Assert.Equal(73, fixture.Call.ValidKvLength);
            _ = fixture.Oracle();
        });
    }

    [Theory]
    [InlineData(73, 73)]
    [InlineData(73, -1, 73)]
    [InlineData(73, 74, 73)]
    public void RejectsMalformedLengthTables(params int[] lengths)
    {
        WithFixture(lengths, path => Assert.Throws<InvalidDataException>(() => AttentionFixture.Load(path)));
    }

    private static void WithFixture(int[]? lengths, Action<string> check)
    {
        string directory = Path.Combine(Path.GetTempPath(), "hartsy-attention-lengths-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            string path = Path.Combine(directory, "fixture.json");
            using (FileStream file = File.Create(path))
            using (Utf8JsonWriter writer = new(file))
            {
                writer.WriteStartObject();
                writer.WriteNumber("schema", 1);
                writer.WriteString("case_id", "length-validation");
                writer.WriteStartObject("tensors");
                foreach (string name in new[] { "q", "k", "v" })
                {
                    int tokens = name == "q" ? 1 : 73;
                    byte[] bytes = new byte[3 * tokens * sizeof(float)];
                    File.WriteAllBytes(Path.Combine(directory, name + ".bin"), bytes);
                    writer.WriteStartObject(name);
                    writer.WriteString("file", name + ".bin");
                    writer.WriteString("sha256", Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant());
                    writer.WriteString("dtype", "f32");
                    WriteArray(writer, "shape", [3, 1, tokens, 1]);
                    WriteArray(writer, "storage_shape", [3, 1, tokens, 1]);
                    WriteArray(writer, "strides", [tokens, tokens, 1, 1]);
                    writer.WriteNumber("offset", 0);
                    writer.WriteEndObject();
                }
                writer.WriteEndObject();
                writer.WriteStartObject("call");
                writer.WriteNumber("kv_length", 73);
                if (lengths is not null) WriteArray(writer, "kv_lengths", lengths);
                writer.WriteString("route", "Dense");
                writer.WriteNumber("scale", 1);
                writer.WriteBoolean("causal", false);
                writer.WriteNumber("query_offset", 0);
                writer.WriteBoolean("legacy_asserted_bounded", false);
                writer.WriteEndObject();
                writer.WriteStartObject("acceptance");
                writer.WriteNumber("rel_l2", 2e-5);
                writer.WriteBoolean("reviewed", false);
                writer.WriteEndObject();
                writer.WriteEndObject();
            }
            check(path);
        }
        finally { Directory.Delete(directory, true); }
    }

    private static void WriteArray(Utf8JsonWriter writer, string name, int[] values)
    {
        writer.WriteStartArray(name);
        foreach (int value in values) writer.WriteNumberValue(value);
        writer.WriteEndArray();
    }
}
