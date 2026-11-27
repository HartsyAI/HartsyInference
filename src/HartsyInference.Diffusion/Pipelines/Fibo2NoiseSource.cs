using System.Buffers.Binary;
using System.Text;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>The starting noise of a Fibo 2.0 generation. Normally the seeded generator draws it. For the diffusers parity check the
/// environment variable <c>HARTSY_FIBO2_NOISE_NPY</c> names a float32 <c>.npy</c> file of the same shape, so the engine and the diffusers
/// reference start from the same tensor. That is a diagnostic hook: it is not a setting, and a shape or dtype mismatch is an error, never
/// a silent fallback.</summary>
internal static class Fibo2NoiseSource
{
    /// <summary>The environment variable that names the saved noise file.</summary>
    internal const string EnvVar = "HARTSY_FIBO2_NOISE_NPY";

    /// <summary>Noise of <paramref name="shape"/>: the saved file when <see cref="EnvVar"/> is set, else the seeded draw.</summary>
    internal static Tensor Create(TensorShape shape, int seed)
    {
        string? path = Environment.GetEnvironmentVariable(EnvVar);
        if (string.IsNullOrWhiteSpace(path))
            return SeedGenerator.CreateNoise(shape, seed);
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
        int dataStart = (file[6] == 1 ? 10 : 12) + headerLength;
        string header = Encoding.ASCII.GetString(file, file[6] == 1 ? 10 : 12, headerLength);
        if (!header.Contains("'<f4'", StringComparison.Ordinal) && !header.Contains("'|f4'", StringComparison.Ordinal))
            throw new InvalidDataException($"{path} must be float32 little-endian: {header}");
        if (header.Contains("'fortran_order': True", StringComparison.Ordinal))
            throw new InvalidDataException($"{path} must be C-ordered.");

        string wanted = "(" + string.Join(", ", Enumerable.Range(0, shape.Rank).Select(i => shape[i].ToString())) + (shape.Rank == 1 ? ",)" : ")");
        string wantedNoSpace = wanted.Replace(" ", string.Empty);
        if (!header.Replace(" ", string.Empty).Contains("'shape':" + wantedNoSpace, StringComparison.Ordinal))
            throw new InvalidDataException($"{path} shape does not match {shape}: {header}");

        long count = shape.ElementCount;
        if (file.Length - dataStart != count * sizeof(float))
            throw new InvalidDataException($"{path} holds {file.Length - dataStart} data bytes; {shape} needs {count * sizeof(float)}.");

        Tensor tensor = new(shape, DType.F32);
        file.AsSpan(dataStart, (int)(count * sizeof(float))).CopyTo(System.Runtime.InteropServices.MemoryMarshal.AsBytes(tensor.AsSpan<float>()));
        return tensor;
    }
}
