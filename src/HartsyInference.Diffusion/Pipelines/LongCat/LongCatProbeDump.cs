using System.Globalization;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Pipelines.LongCat;

/// <summary>Writes the LongCat DiT's probe tensors of one denoise call into the directory of <c>diagnostics.longcatProbeDump</c>, as float32
/// <c>.npy</c> named <c>call{n}_probe_{name}.npy</c>. A name the DiT probes more than once in the same call (conditional and unconditional passes,
/// audio variants) gets <c>_k1</c>, <c>_k2</c>, ... in call order. Diagnostics only: it reads the tensors back to the host and changes no compute.</summary>
internal sealed class LongCatProbeDump
{
    private readonly string _directory;
    private readonly int _call;
    private readonly HashSet<int>? _stageBlocks;
    private readonly Dictionary<string, int> _seen = new(StringComparer.Ordinal);

    /// <param name="directory">The dump directory, created when missing.</param>
    /// <param name="call">The call number in the file names.</param>
    /// <param name="stageBlocks">Block indices whose every stage is written; null writes only block outputs and the non-block probes.</param>
    public LongCatProbeDump(string directory, int call, HashSet<int>? stageBlocks)
    {
        _directory = directory;
        _call = call;
        _stageBlocks = stageBlocks;
        Directory.CreateDirectory(directory);
    }

    /// <summary>Parses <c>diagnostics.longcatProbeBlocks</c> (comma-separated indices); null or blank gives null (no stage dumps).</summary>
    public static HashSet<int>? ParseBlocks(string? text)
    {
        if (string.IsNullOrWhiteSpace(text)) return null;
        HashSet<int> blocks = new();
        foreach (string part in text.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
            blocks.Add(int.Parse(part, CultureInfo.InvariantCulture));
        return blocks;
    }

    /// <summary>The probe callback to install on the DiT for this call.</summary>
    public Action<string, Tensor> Record => Write;

    private void Write(string name, Tensor tensor)
    {
        if (!Wanted(name)) return;
        int ordinal = _seen.GetValueOrDefault(name);
        _seen[name] = ordinal + 1;
        string suffix = ordinal == 0 ? "" : $"_k{ordinal}";
        int[] shape = new int[tensor.Shape.Rank];
        for (int i = 0; i < shape.Length; i++) shape[i] = (int)tensor.Shape[i];
        float[] values = tensor.AsReadOnlySpan<float>().ToArray();
        File.WriteAllBytes(Path.Combine(_directory, $"call{_call:D2}_probe_{name}{suffix}.npy"), NpyFile.Float32(values, shape));
    }

    /// <summary>Whether a probe is written: non-block probes always; a block probe when it is that block's output or its block is named.</summary>
    private bool Wanted(string name)
    {
        if (!name.StartsWith("block", StringComparison.Ordinal)) return true;
        int dot = name.IndexOf('.', StringComparison.Ordinal);
        if (dot < 0 || !int.TryParse(name.AsSpan(5, dot - 5), NumberStyles.None, CultureInfo.InvariantCulture, out int block)) return false;
        return name.EndsWith(".out", StringComparison.Ordinal) || (_stageBlocks is not null && _stageBlocks.Contains(block));
    }
}
