using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Text;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Utilities;

/// <summary>Starting noise that a diagnostics setting can replace with a saved tensor. Equal seeds do not give equal noise in the engine and in a
/// Python reference, so a comparison starts both from one float32 <c>.npy</c>. Each use of a file logs a warning, and a shape, dtype or order
/// mismatch throws rather than falling back to the seeded draw.</summary>
internal static class SavedNoise
{
    /// <summary>Noise of <paramref name="shape"/>: the file <paramref name="knob"/> names when it is set, else the seeded draw.</summary>
    internal static Tensor Create(Knob<string?> knob, TensorShape shape, int seed, string logTag)
    {
        string? path = knob.Value;
        if (string.IsNullOrWhiteSpace(path))
            return SeedGenerator.CreateNoise(shape, seed);
        Logs.Warning($"[{logTag}] {knob.Id} replaces the seeded noise with {path} (shape {shape}).");
        return LoadFloat32Npy(path, shape);
    }

    /// <summary>Reads a little-endian float32 C-order <c>.npy</c> (format 1.0 or 2.0) whose shape must equal <paramref name="shape"/>.</summary>
    internal static Tensor LoadFloat32Npy(string path, TensorShape shape)
    {
        byte[] file = File.ReadAllBytes(path);
        if (file.Length < 10 || !file.AsSpan(0, NpyFile.Magic.Length).SequenceEqual(NpyFile.Magic))
            throw new InvalidDataException($"{path} is not an .npy file.");
        int headerLength = file[6] == 1
            ? BinaryPrimitives.ReadUInt16LittleEndian(file.AsSpan(8, 2))
            : checked((int)BinaryPrimitives.ReadUInt32LittleEndian(file.AsSpan(8, 4)));
        int headerStart = file[6] == 1 ? 10 : 12;
        int dataStart = headerStart + headerLength;
        if (dataStart > file.Length)
            throw new InvalidDataException($"{path} header runs past the end of the file.");
        string header = Encoding.ASCII.GetString(file, headerStart, headerLength);
        string compact = header.Replace(" ", string.Empty);
        if (!compact.Contains("'descr':'<f4'", StringComparison.Ordinal) && !compact.Contains("'descr':'|f4'", StringComparison.Ordinal))
            throw new InvalidDataException($"{path} must be float32 little-endian: {header}");
        if (compact.Contains("'fortran_order':True", StringComparison.Ordinal))
            throw new InvalidDataException($"{path} must be C-ordered.");

        string wanted = "(" + string.Join(",", Enumerable.Range(0, shape.Rank).Select(i => shape[i].ToString())) + (shape.Rank == 1 ? ",)" : ")");
        if (!compact.Contains("'shape':" + wanted, StringComparison.Ordinal))
            throw new InvalidDataException($"{path} shape does not match {shape}: {header}");

        long count = shape.ElementCount;
        long dataBytes = file.Length - dataStart;
        if (dataBytes != count * sizeof(float))
            throw new InvalidDataException($"{path} holds {dataBytes} data bytes; {shape} needs {count * sizeof(float)}.");

        Tensor tensor = new(shape, DType.F32);
        file.AsSpan(dataStart, (int)(count * sizeof(float))).CopyTo(MemoryMarshal.AsBytes(tensor.AsSpan<float>()));
        return tensor;
    }
}
