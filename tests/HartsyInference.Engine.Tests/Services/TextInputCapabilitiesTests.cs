using System.Runtime.CompilerServices;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Placement;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.Gguf;

namespace HartsyInference.Engine.Tests.Services;

/// <summary>What a host can learn about a local text model's inputs from its path alone, and how a MOSS-VL slot is reported as loaded.
/// No weights are read.</summary>
public sealed class TextInputCapabilitiesTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "hartsy-caps-" + Guid.NewGuid().ToString("N"));

    public TextInputCapabilitiesTests() => Directory.CreateDirectory(_root);

    public void Dispose() => Directory.Delete(_root, recursive: true);

    private string WriteGguf(string name, string architecture)
    {
        string path = Path.Combine(_root, name);
        using GgufWriter writer = new(path);
        writer.SetMetadata("general.architecture", architecture);
        writer.AddTensor("t", new TensorShape(2, 2), DType.F32, () => new Tensor(new TensorShape(2, 2), DType.F32));
        writer.Flush();
        return path;
    }

    private string WriteFolder(string name, string modelType)
    {
        string dir = Path.Combine(_root, name);
        Directory.CreateDirectory(dir);
        File.WriteAllText(Path.Combine(dir, "config.json"), $$"""{"model_type":"{{modelType}}"}""");
        File.WriteAllBytes(Path.Combine(dir, "model-00001-of-00001.safetensors"), [0]);
        return dir;
    }

    [Fact]
    public void AMossVlGgufTakesImagesVideoAndLiveSessionsWithNoSidecar()
    {
        TextInputCapabilities caps = TextService.InputCapabilities(WriteGguf("moss.gguf", "moss_vl"));
        Assert.Equal(new TextInputCapabilities(Images: true, Video: true, RealtimeSession: true), caps);
    }

    [Fact]
    public void AMossVlCheckpointFolderTakesImagesVideoAndLiveSessions()
    {
        Assert.Equal(new TextInputCapabilities(true, true, true), TextService.InputCapabilities(WriteFolder("moss-nf4", "moss_vl")));
    }

    [Fact]
    public void AnotherFolderOrAPlainGgufTakesNothing()
    {
        Assert.Equal(TextInputCapabilities.None, TextService.InputCapabilities(WriteFolder("qwen", "qwen3")));
        Assert.Equal(TextInputCapabilities.None, TextService.InputCapabilities(WriteGguf("plain.gguf", "llama")));
    }

    [Fact]
    public void AGgufWithASidecarMmprojTakesImagesOnly()
    {
        string text = WriteGguf("vision.gguf", "qwen3vl");
        WriteGguf("vision-mmproj.gguf", "clip");
        Assert.Equal(new TextInputCapabilities(Images: true, Video: false, RealtimeSession: false), TextService.InputCapabilities(text));
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    public void ABlankPathTakesNothing(string? path) => Assert.Equal(TextInputCapabilities.None, TextService.InputCapabilities(path));

    [Fact]
    public void AMissingOrUnreadablePathTakesNothingAndNeverThrows()
    {
        Assert.Equal(TextInputCapabilities.None, TextService.InputCapabilities(Path.Combine(_root, "absent.gguf")));
        string corrupt = Path.Combine(_root, "corrupt.gguf");
        File.WriteAllText(corrupt, "not a gguf file");
        Assert.Equal(TextInputCapabilities.None, TextService.InputCapabilities(corrupt));
    }

    [Fact]
    public void ALoadedMossVlSlotIsReportedWithASingleDevicePlacementAndOtherPlansAreKept()
    {
        TextDeviceSlot moss = new() { LoadedPath = "/models/moss", MossVl = (MossVlRuntime)RuntimeHelpers.GetUninitializedObject(typeof(MossVlRuntime)) };
        TextPlacement planned = new(TextPlacementMode.Split, Feasible: true, ["cuda:0", "cuda:1"], 0, 10, 20, "planned");
        TextDeviceSlot planner = new() { LoadedPath = "/models/llama.gguf", PlannedPlacement = planned };
        TextDeviceSlot idle = new() { LoadedPath = "/models/other.gguf" };
        TextDeviceSlot empty = new();

        IReadOnlyList<LoadedModelPlacement> placed = TextService.PlacementsOf(
        [
            new("vulkan:0", moss), new("cuda:0+cuda:1", planner), new("cpu", idle), new("cuda:1", empty),
        ]);

        Assert.Equal(2, placed.Count);
        LoadedModelPlacement mossPlacement = placed.Single(p => p.ModelPath == "/models/moss");
        Assert.Equal("vulkan:0", mossPlacement.DeviceKey);
        Assert.Equal(["vulkan:0"], mossPlacement.Placement.Devices);
        Assert.True(mossPlacement.Placement.Feasible);
        Assert.Same(planned, placed.Single(p => p.ModelPath == "/models/llama.gguf").Placement);
    }
}
