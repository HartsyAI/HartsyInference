using System.Text.Json;
using HartsyInference.Audio.Models.ControlFoley;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Real-weight CLIP check; <c>HARTSY_CONTROLFOLEY_CLIP_DIR</c> must hold <c>dfn5b_fp16.safetensors</c>
/// (<c>tools/controlfoley/clip_convert_remote.py</c>) and <c>clip_real_reference.safetensors/.json</c>
/// (<c>clip_reference.py real</c>); the test does nothing when it is unset.</summary>
[Trait("Category", "Integration")]
public sealed unsafe class ControlFoleyClipRealWeightTests(ITestOutputHelper output)
{
    private static string? Dir => Environment.GetEnvironmentVariable("HARTSY_CONTROLFOLEY_CLIP_DIR");

    private static float[] Read(Tensor t) => new ReadOnlySpan<float>((void*)t.DataPointer, (int)t.ElementCount).ToArray();

    [Fact]
    public void TextAndImageTowers_MatchPythonOnRealWeights()
    {
        if (Dir is not { Length: > 0 } dir)
        {
            return;
        }

        using SafeTensorsLoader weightsLoader = new();
        weightsLoader.Load(Path.Combine(dir, "dfn5b_fp16.safetensors"));
        Dictionary<string, Tensor> weights = new();
        foreach (string name in weightsLoader.Descriptors.Keys)
        {
            weights[name] = weightsLoader.GetTensor(name);
        }

        using SafeTensorsLoader refLoader = new();
        refLoader.Load(Path.Combine(dir, "clip_real_reference.safetensors"));
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "clip_real_reference.json")));
        string[] prompts = doc.RootElement.GetProperty("prompts").EnumerateArray().Select(e => e.GetString()!).ToArray();

        using IBackend backend = new CpuBackend();
        ControlFoleyClip clip = new(ControlFoleyClipConfig.Dfn5bViTH14);
        clip.LoadWeights(weights);

        float[] text = clip.EncodeText(backend, prompts);
        float[] wantText = Read(refLoader.GetTensor("ref.text"));
        float textDiff = MaxAbs(wantText, text);
        output.WriteLine($"text [{prompts.Length},77,1024] max |d| = {textDiff:E3}");

        float[] image = clip.EncodeImages(backend, Read(refLoader.GetTensor("in.frames")));
        float imageDiff = MaxAbs(Read(refLoader.GetTensor("ref.image")), image);
        output.WriteLine($"image [2,1024] max |d| = {imageDiff:E3}");

        Assert.True(textDiff <= 1e-3f, $"text max |d| = {textDiff}");
        Assert.True(imageDiff <= 1e-3f, $"image max |d| = {imageDiff}");
    }

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
}
