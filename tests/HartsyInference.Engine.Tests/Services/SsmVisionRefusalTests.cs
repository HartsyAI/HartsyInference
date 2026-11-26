using System.Collections.Concurrent;
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
    private readonly ConcurrentQueue<string> _warnings = new();

    public SsmVisionRefusalTests()
    {
        Directory.CreateDirectory(_root);
        // Tests in other classes may run in parallel and log through the same sink, so this queue is concurrent and only the refusal text is asserted.
        Logs.SetLogger((level, message) => { if (level == LogLevel.Warning) _warnings.Enqueue(message); });
    }

    public void Dispose()
    {
        // Restore the default sink (stderr), so later tests log where they did before.
        Logs.SetLogger((_, message) => Console.Error.WriteLine(message));
        Directory.Delete(_root, recursive: true);
    }

    private static void WriteGguf(string path, Action<GgufWriter> metadata)
    {
        using GgufWriter writer = new(path);
        metadata(writer);
        writer.AddTensor("t", new TensorShape(2, 2), DType.F32, () => new Tensor(new TensorShape(2, 2), DType.F32));
        writer.Flush();
    }

    private (string Text, string Mmproj) WriteModelPair(string textName, string mmprojName, string architecture)
    {
        string text = Path.Combine(_root, textName);
        string mmproj = Path.Combine(_root, mmprojName);
        WriteGguf(text, w => w.SetMetadata("general.architecture", architecture));
        WriteGguf(mmproj, w =>
        {
            w.SetMetadata("general.architecture", "clip");
            w.SetMetadata("clip.projector_type", "siglip");
        });
        // The sidecar must be found, so a refusal below is the cause of a text-only slot, not a missing file.
        Assert.Equal(mmproj, TextService.FindMmproj(text));
        return (text, mmproj);
    }

    [Fact]
    public void SsmBackboneWithANonQwen3VlProjector_StaysTextOnly()
    {
        (string text, _) = WriteModelPair("sensenova-loop-9b.gguf", "mmproj-sensenova-loop-9b.gguf", "sensenova_loop");

        TextDeviceSlot slot = new();
        TextService.LoadVisionInto(slot, text, ssmBackbone: true);

        Assert.Null(slot.SpliceVision);
        Assert.Null(slot.MllamaVision);
        Assert.Null(slot.VisionPath);
        Assert.Contains(_warnings, w => w.Contains("is not a Qwen3-VL projector") && w.Contains("stays text-only"));
    }

    [Fact]
    public void NonSsmBackboneWithTheSameProjector_IsNotRefusedAsAQwen3VlMismatch()
    {
        // Negative control: the same sidecar on a plain transformer model goes down the generic path, so the refusal message is the SSM branch's alone.
        (string text, _) = WriteModelPair("plain.gguf", "mmproj-plain.gguf", "llama");

        TextDeviceSlot slot = new();
        TextService.LoadVisionInto(slot, text);

        Assert.DoesNotContain(_warnings, w => w.Contains("is not a Qwen3-VL projector"));
    }
}
