using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.Tests;

/// <summary>The checkpoint's <c>tokenizer.json</c>, read by the shared byte-level BPE loader, encodes the MOSS-VL chat template to the
/// ids the HF <c>tokenizers</c> library produces. The expected ids were generated from the same file with that library.</summary>
[Trait("Category", "Integration")]
public sealed class MossVlTokenizerTests
{
    private const string CheckpointVariable = "HARTSY_MOSS_VL_INDEX";

    /// <summary>The template the generator builds around a question: the image placeholder is the single token 151655.</summary>
    private const string Prompt =
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|>\n<|im_start|>assistant\n";

    private static readonly int[] ExpectedIds =
        [151644, 872, 198, 151652, 151655, 151653, 74785, 279, 2168, 13, 151645, 198, 151644, 77091, 198];

    [RealWeightsFact(CheckpointVariable)]
    public void ChatTemplateEncodesLikeTheHfLibrary()
    {
        string tokenizerPath = Path.Combine(Path.GetDirectoryName(Environment.GetEnvironmentVariable(CheckpointVariable)!)!, "tokenizer.json");
        using FileStream stream = File.OpenRead(tokenizerPath);
        GgufTokenizer tokenizer = HfTokenizerJson.LoadByteLevelBpe(stream);

        int[] ids = tokenizer.Encode(Prompt, addSpecial: true);

        Assert.Equal(ExpectedIds, ids);
    }
}
