using HartsyInference.LLM.Generation;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.Looped;

/// <summary>The Dory GGUF's tokenizer (<c>tokenizer.ggml.pre = pixtral</c>, Mistral's Tekken split) against ids from the BF16 checkpoint's own
/// <c>tokenizer.json</c> through Hugging Face tokenizers 0.23.3 (pinned revision b366b87f, no special tokens). Skips when the GGUF is absent (set
/// DORY_GGUF_PATH to NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf).</summary>
public sealed class DoryTokenizerParityTests
{
    private readonly ITestOutputHelper _output;

    public DoryTokenizerParityTests(ITestOutputHelper output) => _output = output;

    public static TheoryData<string, int[]> Samples() => new()
    {
        { "Explain in three sentences why the sky is blue, then list two things that would change its color.",
            [2763, 44122, 1294, 3300, 35699, 5782, 1278, 21283, 1395, 10991, 1044, 2430, 2951, 2295, 5705, 1455, 2168, 4036, 2246, 4558, 1046] },
        { "You are Swarmie, the official in-app helper for SwarmUI — a Stable Diffusion / image generation interface.",
            [4568, 1584, 6767, 3802, 1386, 1044, 1278, 9481, 1294, 36924, 33106, 1394, 6767, 3802, 8442, 2251, 1261, 1757, 1638, 117606, 1987, 3937,
                12388, 8288, 1046] },
        { "Whenever the user asks how to do anything in SwarmUI — a feature, a setting, an API call, an extension, prompt syntax, model setup, "
            + "troubleshooting — you MUST consult the bundled documentation rather than guessing.",
            [3237, 60095, 1278, 3330, 25747, 2606, 1317, 1653, 7211, 1294, 6767, 3802, 8442, 2251, 1261, 7326, 1044, 1261, 9436, 1044, 1420, 10523,
                3690, 1044, 1420, 13088, 1044, 16925, 22692, 1044, 3244, 14919, 1044, 38273, 86920, 2251, 1636, 123390, 12627, 1278, 124252, 18189, 6153,
                2224, 67541, 1046] },
        { "Use the swarm_docs tool: 1. If you don't already know which doc covers the topic, call swarm_docs with action='list' once.\n\n2. Then call it again.",
            [15908, 1278, 4250, 3802, 3602, 11540, 9519, 1058, 1032, 1049, 1046, 3367, 1636, 2607, 2405, 5314, 2840, 1799, 8953, 23843, 1278, 17915,
                1044, 3690, 4250, 3802, 3602, 11540, 1454, 5263, 3760, 6059, 1039, 5785, 1338, 1050, 1046, 6830, 3690, 1494, 2790, 1046] },
        { "In 2026, GPT-4o and HTTPServer cost $1,234.56 (approx. 12%)!\r\n\tTabs   and    spaces\n\n\nend",
            [1785, 1032, 1050, 1048, 1050, 1054, 1044, 1461, 11130, 1045, 1052, 1111, 1321, 23733, 11473, 4362, 1659, 1049, 1044, 1050, 1051, 1052,
                1046, 1053, 1054, 1319, 32727, 1046, 1032, 1049, 1050, 6879, 1033, 1013, 1010, 1009, 71980, 1256, 1321, 1293, 18971, 4688, 1474] },
        { "CamelCaseWords like getHTTPResponseCode, iPhone and McDonald's; I'm sure they'll've done it.",
            [38487, 1299, 11139, 47832, 2479, 2012, 30499, 6566, 5603, 1044, 1623, 16742, 1321, 5303, 31609, 1681, 1059, 1362, 4525, 5257, 2127, 7534,
                6483, 5595, 1494, 1046] },
        { "Übergröße naïve café 日本語 😀 emoji // path/to/file.py\n",
            [79904, 1870, 1792, 9755, 98355, 35858, 30367, 15199, 119685, 1152, 1128, 1873, 26303, 1895, 3549, 35466, 55998, 8261, 1010] },
    };

    [Theory]
    [MemberData(nameof(Samples))]
    [Trait("Category", "Integration")]
    public void EncodesLikeTheCheckpointsOwnTokenizer(string text, int[] expected)
    {
        string? path = Environment.GetEnvironmentVariable("DORY_GGUF_PATH");
        if (string.IsNullOrEmpty(path) || !File.Exists(path))
        {
            _output.WriteLine("SKIPPED: set DORY_GGUF_PATH to the Dory Q5_K_M GGUF.");
            return;
        }
        using GgufLoader loader = new();
        loader.Load(path);
        Assert.Equal("pixtral", loader.Metadata.GetString("tokenizer.ggml.pre"));
        ILlmTokenizer tokenizer = GgufLanguageModel.BuildTokenizer(loader.Metadata);
        int[] ids = tokenizer.EncodeOrdinary(text);
        _output.WriteLine($"{ids.Length} ids (reference {expected.Length})");
        Assert.Equal(expected, ids);
    }
}
