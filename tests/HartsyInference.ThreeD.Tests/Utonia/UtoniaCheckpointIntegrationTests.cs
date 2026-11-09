using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ThreeD.Models.Utonia;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Utonia;

/// <summary>
/// Real-weight key-map check. Needs Models/Utonia/utonia.pth (Pointcept/Utonia, CC-BY-NC-4.0, not downloaded by this
/// suite). Returns without asserting when the file is absent, so the Integration lane stays green offline.
/// </summary>
[Trait("Category", "Integration")]
public sealed class UtoniaCheckpointIntegrationTests
{
    [Fact]
    public void Checkpoint_EveryStateDictKeyIsMappedOrAKnownComputedBuffer()
    {
        string? path = FindCheckpoint("utonia.pth");
        if (path is null)
        {
            return;
        }

        using PytorchPickleLoader loader = new();
        loader.Load(path, recursiveFlatten: true);

        List<string> unmapped = new();
        foreach (string key in loader.Descriptors.Keys)
        {
            string stateKey = key.StartsWith("state_dict.", StringComparison.Ordinal) ? key["state_dict.".Length..] : key;
            if (UtoniaCheckpointKeys.ToCSharpName(stateKey) is null && !UtoniaCheckpointKeys.IsComputedBuffer(stateKey))
            {
                unmapped.Add(key);
            }
        }

        Assert.Empty(unmapped);
    }

    private static string? FindCheckpoint(string fileName)
    {
        DirectoryInfo? dir = new(AppContext.BaseDirectory);
        while (dir is not null)
        {
            string candidate = Path.Combine(dir.FullName, "Models", "Utonia", fileName);
            if (File.Exists(candidate))
            {
                return candidate;
            }

            dir = dir.Parent;
        }

        return null;
    }
}
