using System.Security.Cryptography;
using System.Text.Json;
using HartsyInference.Tests.Common;
using Xunit;

namespace HartsyInference.LLM.Tests.DeepSeekV41.Engram;

/// <summary>The tracked constant files are the ones the manifest describes.</summary>
public sealed class EngramConstantsFileTests
{
    [Fact]
    public void CommittedFilesHashToTheManifestAndTheManifestPinsTheRevision()
    {
        string dir = Path.Combine(RepoRoot.Path, "src", "HartsyInference.LLM", "DeepSeekV41", "Engram", "Constants");
        JsonElement manifest = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "manifest.json"))).RootElement;
        Assert.Equal("dba1be0a40aa45a94ad051997016db3960a90277", manifest.GetProperty("checkpoint_revision").GetString());
        int files = 0;
        foreach (JsonProperty file in manifest.GetProperty("files").EnumerateObject())
        {
            string actual = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(Path.Combine(dir, file.Name)))).ToLowerInvariant();
            Assert.Equal(file.Value.GetProperty("sha256").GetString(), actual);
            files++;
        }
        Assert.Equal(4, files);
    }
}
