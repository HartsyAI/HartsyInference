using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.OpParity;

/// <summary>Fixture files on disk: <c>{root}/{family}/{case}/{backend}/</c> holding <c>inputs.safetensors</c>, <c>outputs.safetensors</c> and <c>manifest.json</c>.</summary>
public static class FixtureStore
{
    public static string CaseDir(string root, string family, string caseName, string backendKind) => Path.Combine(root, family, caseName, backendKind);

    public static string ManifestPath(string caseDir) => Path.Combine(caseDir, "manifest.json");

    /// <summary>Writes the named tensors as one safetensors file and returns its hash.</summary>
    public static string Save(string path, IReadOnlyDictionary<string, Tensor> tensors, IReadOnlyDictionary<string, string> metadata)
    {
        SafeTensorsWriter.Save(path, tensors, metadata);
        return FixtureManifest.HashFile(path);
    }

    /// <summary>Reads every tensor of a safetensors file into owned host copies.</summary>
    public static Dictionary<string, Tensor> Load(string path)
    {
        using SafeTensorsLoader loader = new();
        loader.Load(path);
        Dictionary<string, Tensor> owned = new(StringComparer.Ordinal);
        foreach ((string name, Tensor view) in loader.GetAllTensors())
        {
            owned[name] = view.To(view.Device);
            view.Dispose();
        }
        return owned;
    }
}
