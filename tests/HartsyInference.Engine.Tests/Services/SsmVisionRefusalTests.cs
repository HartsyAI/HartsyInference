using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.Gguf;
using Xunit;

namespace HartsyInference.Engine.Tests.Services;

/// <summary>An SSM backbone (Qwen3.5-VL) takes only a Qwen3-VL projector: its image grid comes from the Qwen3-VL tower. Any other sidecar is refused at
/// load time, so the model stays text-only instead of answering with a wrong grid. No weights are read.</summary>
public sealed class SsmVisionRefusalTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "hartsy-ssm-vision-" + Guid.NewGuid().ToString("N"));

    public SsmVisionRefusalTests() => Directory.CreateDirectory(_root);

    public void Dispose() => Directory.Delete(_root, recursive: true);

    private static void WriteGguf(string path, Action<GgufWriter> metadata)
    {
        using GgufWriter writer = new(path);
        metadata(writer);
        writer.AddTensor("t", new TensorShape(2, 2), DType.F32, () => new Tensor(new TensorShape(2, 2), DType.F32));
        writer.Flush();
    }

    [Fact]
    public void SsmBackboneWithANonQwen3VlProjector_StaysTextOnly()
    {
        string text = Path.Combine(_root, "sensenova-loop-9b.gguf");
        string mmproj = Path.Combine(_root, "mmproj-sensenova-loop-9b.gguf");
        WriteGguf(text, w => w.SetMetadata("general.architecture", "sensenova_loop"));
        WriteGguf(mmproj, w =>
        {
            w.SetMetadata("general.architecture", "clip");
            w.SetMetadata("clip.projector_type", "siglip");
        });
        // Precondition: the sidecar is found, so the refusal below is what keeps the slot text-only, not a missing file.
        Assert.Equal(mmproj, TextService.FindMmproj(text));

        List<string> warnings = new();
        Logs.SetLogger((level, message) => { if (level == LogLevel.Warning) warnings.Add(message); });

        TextDeviceSlot slot = new();
        TextService.LoadVisionInto(slot, text, ssmBackbone: true);

        Assert.Null(slot.SpliceVision);
        Assert.Null(slot.MllamaVision);
        Assert.Null(slot.VisionPath);
        Assert.Contains(warnings, w => w.Contains("is not a Qwen3-VL projector") && w.Contains("stays text-only"));
    }

    [Fact]
    public void NonSsmBackboneWithTheSameProjector_IsNotRefusedAsAQwen3VlMismatch()
    {
        // Negative control: the same sidecar on a plain transformer model goes down the generic path, so the refusal message is the SSM branch's alone.
        string text = Path.Combine(_root, "plain.gguf");
        string mmproj = Path.Combine(_root, "mmproj-plain.gguf");
        WriteGguf(text, w => w.SetMetadata("general.architecture", "llama"));
        WriteGguf(mmproj, w =>
        {
            w.SetMetadata("general.architecture", "clip");
            w.SetMetadata("clip.projector_type", "siglip");
        });
        List<string> warnings = new();
        Logs.SetLogger((level, message) => { if (level == LogLevel.Warning) warnings.Add(message); });

        TextDeviceSlot slot = new();
        TextService.LoadVisionInto(slot, text);

        Assert.DoesNotContain(warnings, w => w.Contains("is not a Qwen3-VL projector"));
    }
}
