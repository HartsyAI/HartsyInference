using HartsyInference.Core.Configuration;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Pipelines.LongCat;

/// <summary>Diagnostics-only dump and replacement of the LongCat avatar's conditioning tensors (text embeddings, audio tokens, the first-frame
/// latent, the speech) in the reference's float32 .npy layouts. Default off: with the knobs unset nothing is read or written.</summary>
internal static class LongCatConditioningDiagnostics
{
    /// <summary>Writes one tensor as float32 .npy as <c>call{n}_{name}.npy</c> in the dump directory; a null or blank directory writes nothing.</summary>
    public static void Dump(string? directory, int call, string name, ReadOnlySpan<float> values, ReadOnlySpan<int> shape)
    {
        if (string.IsNullOrWhiteSpace(directory)) return;
        Directory.CreateDirectory(directory);
        File.WriteAllBytes(Path.Combine(directory, $"call{call:D2}_{name}.npy"), NpyFile.Float32(values, shape));
    }

    /// <summary>The knob's file as host floats, checked against <paramref name="shape"/>; <paramref name="engine"/> when the knob is unset.</summary>
    public static float[] Replace(Knob<string?> knob, float[] engine, int[] shape)
    {
        string? path = knob.Value;
        if (string.IsNullOrWhiteSpace(path)) return engine;
        long[] dims = new long[shape.Length];
        for (int i = 0; i < shape.Length; i++) dims[i] = shape[i];
        using Tensor loaded = SavedNoise.LoadFloat32Npy(path, new TensorShape(dims));
        Logs.Warning($"[LongCatConditioning] {knob.Id} replaces the engine's tensor with {path} (shape {string.Join(" x ", shape)}).");
        return loaded.AsReadOnlySpan<float>().ToArray();
    }
}
