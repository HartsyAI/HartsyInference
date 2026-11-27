using System.Globalization;
using System.Text;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Numerics;

namespace HartsyInference.Diffusion.Pipelines.LongCat;

/// <summary>Diagnostics-only dump and substitution of a LongCat avatar generation's step-0 conditioning, under the reference's float32 .npy names. A
/// substitute file is read by element count (its shape may differ from the engine's), so a reference layout with the same elements is accepted.
/// Default off: with the knobs unset nothing is read or written.</summary>
internal static class LongCatConditioningDiagnostics
{
    public const string TextEmbeddings = "text_embeddings_step00";
    public const string TextMask = "text_mask_step00";
    public const string AudioEmbedding = "audio_embedding_step00";
    public const string InputLatent = "input_latent_step00";
    public const string Speech = "speech_16k_mono_step00";

    /// <summary>Writes <paramref name="values"/> as float32 .npy <paramref name="name"/>.npy in <paramref name="directory"/>; a null or blank directory writes nothing.</summary>
    public static void Dump(string? directory, string name, ReadOnlySpan<float> values, ReadOnlySpan<int> shape)
    {
        if (string.IsNullOrWhiteSpace(directory)) return;
        Directory.CreateDirectory(directory);
        File.WriteAllBytes(Path.Combine(directory, name + ".npy"), NpyFile.Float32(values, shape));
    }

    /// <summary>Reads <paramref name="name"/>.npy from <paramref name="directory"/> as host floats. Returns false when the directory is unset or the file is
    /// absent. A present file must hold <paramref name="count"/> float32 values (any count when it is negative), or it is refused.</summary>
    public static bool TryRead(string? directory, string name, int count, out float[] values)
    {
        values = [];
        if (string.IsNullOrWhiteSpace(directory)) return false;
        string file = Path.Combine(directory, name + ".npy");
        if (!File.Exists(file)) return false;
        byte[] bytes = File.ReadAllBytes(file);
        if (bytes.Length < 10 || !bytes.AsSpan(0, NpyFile.Magic.Length).SequenceEqual(NpyFile.Magic))
            throw new InvalidDataException($"{file} is not an .npy file.");
        int headerStart = bytes[6] == 1 ? 10 : 12;
        int headerLength = bytes[6] == 1 ? BitConverter.ToUInt16(bytes, 8) : (int)BitConverter.ToUInt32(bytes, 8);
        string header = Encoding.ASCII.GetString(bytes, headerStart, headerLength);
        if (!header.Contains("'<f4'", StringComparison.Ordinal))
            throw new InvalidDataException($"{file} must be little-endian float32 (descr '<f4').");
        int dataStart = headerStart + headerLength;
        int elements = (bytes.Length - dataStart) / sizeof(float);
        if (count >= 0 && elements != count)
            throw new InvalidDataException($"{file} holds {elements} values; the engine's tensor holds {count}.");
        values = new float[elements];
        Buffer.BlockCopy(bytes, dataStart, values, 0, elements * sizeof(float));
        Logs.Warning($"[LongCatConditioning] {name} replaces the engine's tensor with {file} ({elements} values).");
        return true;
    }

    /// <summary>The first <paramref name="cond"/> frames of each channel of a host latent <c>[channels, frames, hw]</c>, as <c>[channels, cond, hw]</c>.</summary>
    public static float[] CondFrames(float[] latent, int channels, int frames, int hw, int cond)
    {
        float[] result = new float[channels * cond * hw];
        for (int c = 0; c < channels; c++)
            Array.Copy(latent, (long)c * frames * hw, result, (long)c * cond * hw, (long)cond * hw);
        return result;
    }
}
