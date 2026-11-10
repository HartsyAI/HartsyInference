using System.Text.Json;
using HartsyInference.LLM.Models.MossVl;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.Tests;

/// <summary>A lone MOSS-VL GGUF stands in for the checkpoint's JSON files: the config built from its metadata, and its embedded
/// tokenizer, must agree with <c>config.json</c> and <c>tokenizer.json</c>. Reads headers only.</summary>
[Trait("Category", "Integration")]
public sealed class MossVlGgufConfigTests
{
    private const string CheckpointVariable = "HARTSY_MOSS_VL_INDEX";
    private const string GgufVariable = "HARTSY_MOSS_VL_GGUF";

    [RealWeightsFact(CheckpointVariable, GgufVariable)]
    public void ConfigFromGgufMatchesConfigJson()
    {
        string checkpoint = Path.GetDirectoryName(Environment.GetEnvironmentVariable(CheckpointVariable)!)!;
        MossVlConfig fromJson;
        using (JsonDocument doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(checkpoint, "config.json"))))
            fromJson = MossVlConfig.FromJson(doc.RootElement);

        ILlmTokenizer tokenizer;
        using (FileStream stream = File.OpenRead(Path.Combine(checkpoint, "tokenizer.json")))
            tokenizer = HfTokenizerJson.LoadByteLevelBpe(stream);
        using GgufLoader gguf = new();
        gguf.Load(Environment.GetEnvironmentVariable(GgufVariable)!);
        MossVlConfig fromGguf = MossVlConfig.FromGguf(gguf.Metadata, tokenizer);

        // Records compare their array and set members by reference, so the printed forms are compared and the sets separately.
        Assert.Equal(fromJson.Text.ToString(), fromGguf.Text.ToString());
        Assert.Equal(fromJson.Text.CrossAttnLayers.Order(), fromGguf.Text.CrossAttnLayers.Order());
        Assert.Equal(fromJson.Vision.ToString(), fromGguf.Vision.ToString());
        Assert.Equal(fromJson.Vision.DeepstackIndexes, fromGguf.Vision.DeepstackIndexes);
        Assert.Equal(fromJson.MropeSection, fromGguf.MropeSection);
        Assert.Equal(fromJson.MropeInterleaved, fromGguf.MropeInterleaved);
        Assert.Equal((fromJson.ImageTokenId, fromJson.VideoTokenId, fromJson.VisionStartTokenId, fromJson.VisionEndTokenId),
            (fromGguf.ImageTokenId, fromGguf.VideoTokenId, fromGguf.VisionStartTokenId, fromGguf.VisionEndTokenId));
    }
}
