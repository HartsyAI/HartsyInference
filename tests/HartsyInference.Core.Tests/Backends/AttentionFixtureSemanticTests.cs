using System.Text.Json.Nodes;
using HartsyInference.GpuBenchmarks.Attention;
using Xunit;

namespace HartsyInference.Core.Tests.Backends;

/// <summary>Specialized operations cannot be certified by executing a different dense operation.</summary>
public sealed class AttentionFixtureSemanticTests
{
    [Theory]
    [InlineData("tree-merge")]
    [InlineData("step-merge")]
    [InlineData("sol")]
    [InlineData("sage")]
    [InlineData("prequantized")]
    [InlineData("decode")]
    [InlineData("decode-gqa")]
    [InlineData("unknown")]
    public void RejectsUnsupportedApisBeforeLoadingOperands(string api)
    {
        Reject(new JsonObject { ["api"] = api }, new JsonObject());
    }

    [Theory]
    [InlineData("sol")]
    [InlineData("tree_mask")]
    [InlineData("unknown_option")]
    public void RejectsUnrepresentedOptionsEvenWithDenseApi(string option)
    {
        Reject(new JsonObject { ["api"] = "dense", [option] = new JsonObject() }, new JsonObject());
    }

    [Theory]
    [InlineData("prefix_output")]
    [InlineData("prefix_lse")]
    [InlineData("step_k")]
    [InlineData("step_v")]
    public void RejectsUnrepresentedOperandsEvenWithoutApi(string operand)
    {
        Reject(new JsonObject(), new JsonObject { [operand] = new JsonObject() });
    }

    [Theory]
    [InlineData(0)]
    [InlineData(2)]
    [InlineData(16)]
    public void RejectsUnimplementedPreparedReuse(int count)
    {
        Reject(new JsonObject { ["reuse_count"] = count }, new JsonObject());
    }

    private static void Reject(JsonObject call, JsonObject tensors)
    {
        string path = Path.Combine(Path.GetTempPath(), "hartsy-attention-semantics-" + Guid.NewGuid().ToString("N") + ".json");
        try
        {
            JsonObject manifest = new() { ["schema"] = 1, ["case_id"] = "semantic-rejection", ["call"] = call, ["tensors"] = tensors };
            File.WriteAllText(path, manifest.ToJsonString());
            Assert.Throws<NotSupportedException>(() => AttentionFixture.Load(path));
        }
        finally { File.Delete(path); }
    }
}
