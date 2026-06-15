using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.ModelHandler.PyTorch;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Interactive.Tests;

/// <summary>Env-gated validation of the real Hunyuan-GameCraft checkpoint load chain: the <c>.pt</c> pickle reader
/// + the component converter. Skips cleanly unless <c>HUNYUAN_GAMECRAFT_PATH</c> points at a directory containing
/// the <c>mp_rank_00_model_states.pt</c> (or the file itself). This is the first real-weights checkpoint of the
/// 🔧→✅ pass; full numeric validation uses the python dump/diff harness on a GPU.</summary>
public sealed class GameCraftRealCheckpointTests
{
    private readonly ITestOutputHelper _output;
    public GameCraftRealCheckpointTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void LoadsAndRoutesRealCheckpoint()
    {
        string? path = Environment.GetEnvironmentVariable("HUNYUAN_GAMECRAFT_PATH");
        if (string.IsNullOrEmpty(path))
        {
            _output.WriteLine("SKIPPED: HUNYUAN_GAMECRAFT_PATH not set.");
            return;
        }

        string pt = Directory.Exists(path)
            ? Directory.GetFiles(path, "*.pt", SearchOption.AllDirectories).FirstOrDefault()
              ?? throw new FileNotFoundException($"No .pt under {path}")
            : path;

        using PytorchPickleLoader loader = new();
        loader.Load(pt);
        Dictionary<string, Tensor> all = loader.GetAllTensors();
        _output.WriteLine($"Loaded {all.Count} tensors from {pt}");
        Assert.NotEmpty(all);

        HunyuanGameCraftCheckpointConverter.ConvertedWeights cw = HunyuanGameCraftCheckpointConverter.Convert(all);
        _output.WriteLine($"Routed: Dit={cw.Dit.Count} CameraNet={cw.CameraNet.Count} Vae={cw.Vae.Count} Llava={cw.Llava.Count} Clip={cw.Clip.Count}");

        Assert.NotEmpty(cw.Dit);
        Assert.NotEmpty(cw.CameraNet);
        Assert.Contains(cw.Dit.Keys, k => k.StartsWith("double_blocks.", StringComparison.Ordinal));
        Assert.Contains(cw.CameraNet.Keys, k => k.StartsWith("camera_in.", StringComparison.Ordinal));
    }
}
