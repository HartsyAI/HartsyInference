using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Text;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>The starting noise of a Fibo 2.0 generation. Normally the seeded generator draws it. For the diffusers parity check the
/// <see cref="EngineKnobs.Fibo2NoiseNpy"/> setting (<c>diagnostics.fibo2NoiseNpy</c>) names a float32 <c>.npy</c> of the same shape, so the
/// engine and the diffusers reference start from the same tensor. It is a diagnostic: each use logs a warning, and a shape, dtype or
/// order mismatch throws rather than falling back to the seeded draw.</summary>
internal static class Fibo2NoiseSource
{
    /// <summary>Noise of <paramref name="shape"/>: the saved file when the setting names one, else the seeded draw.</summary>
    internal static Tensor Create(TensorShape shape, int seed)
    {
        string? path = EngineKnobs.Fibo2NoiseNpy.Value;
        if (string.IsNullOrWhiteSpace(path))
            return SeedGenerator.CreateNoise(shape, seed);
        Logs.Warning($"[Fibo2] diagnostics.fibo2NoiseNpy replaces the seeded noise with {path} (shape {shape}).");
        return LoadFloat32Npy(path, shape);
    }

    /// <summary>Reads a little-endian float32 C-order <c>.npy</c> (format 1.0 or 2.0) whose shape must equal <paramref name="shape"/>.</summary>
    internal static Tensor LoadFloat32Npy(string path, TensorShape shape)
    {
        byte[] file = File.ReadAllBytes(path);
        if (file.Length < 10 || file[0] != 0x93 || file[1] != (byte)'N' || file[2] != (byte)'U' || file[3] != (byte)'M' || file[4] != (byte)'P' || file[5] != (byte)'Y')
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
