using System.Text.Json;
using HartsyInference.Audio.Models.ControlFoley;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>A tiny random open_clip CLIP run through ControlFoley's patched text path and its normalised image path
/// (<c>tools/controlfoley/clip_reference.py</c>) must match the port, and the tokenizer must reproduce
/// <c>open_clip.get_tokenizer('ViT-H-14-378-quickgelu')</c>.</summary>
public sealed unsafe class ControlFoleyClipParityTests
{
    private static readonly string Dir = Path.Combine(AppContext.BaseDirectory, "Fixtures", "ControlFoleyClip");

    private static readonly ControlFoleyClipConfig Tiny = new()
    {
        EmbedDim = 16, TextWidth = 24, TextLayers = 3, TextHeads = 3, VocabSize = 512,
        VisionWidth = 32, VisionLayers = 3, VisionHeads = 4, PatchSize = 7, InputSize = 44,
    };

    private static float[] Read(Tensor t) => new ReadOnlySpan<float>((void*)t.DataPointer, (int)t.ElementCount).ToArray();

    private static float MaxAbs(float[] want, float[] got)
    {
        Assert.Equal(want.Length, got.Length);
        float worst = 0f;
        for (int i = 0; i < want.Length; i++)
        {
            worst = MathF.Max(worst, MathF.Abs(want[i] - got[i]));
        }

        return worst;
    }

    [Fact]
    public void TextAndImageTowers_MatchOfficialImplementation()
    {
        using SafeTensorsLoader loader = new();
        loader.Load(Path.Combine(Dir, "clip_tiny.safetensors"));
        Dictionary<string, Tensor> weights = new();
        foreach (string name in loader.Descriptors.Keys)
        {
            weights[name] = loader.GetTensor(name);
        }

        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(Dir, "clip_tiny.json")));
        int[][] tokens = doc.RootElement.GetProperty("tokens").EnumerateArray()
            .Select(row => row.EnumerateArray().Select(e => e.GetInt32()).ToArray()).ToArray();

        using IBackend backend = new CpuBackend();
        ControlFoleyClip clip = new(Tiny);
        clip.LoadWeights(weights);

        float[] text = clip.EncodeTokens(backend, tokens);
        float[] wantText = Read(weights["ref.text"]);
        Assert.True(MaxAbs(wantText, text) <= 1e-5f, $"text max |d| = {MaxAbs(wantText, text)}");

        float[] image = clip.EncodeImages(backend, Read(weights["in.frames"]));
        float[] wantImage = Read(weights["ref.image"]);
        Assert.True(MaxAbs(wantImage, image) <= 1e-5f, $"image max |d| = {MaxAbs(wantImage, image)}");
    }

    [Fact]
    public void Tokenizer_MatchesOpenClipSimpleTokenizer()
    {
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(Dir, "clip_tokens.json")));
        ControlFoleyClipTokenizer tokenizer = new();
        foreach (JsonElement item in doc.RootElement.EnumerateArray())
        {
            int[] want = item.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            string text = item.GetProperty("text").GetString()!;
            Assert.True(want.AsSpan().SequenceEqual(tokenizer.Encode(text)), $"token mismatch for '{text}'");
        }
    }
}
