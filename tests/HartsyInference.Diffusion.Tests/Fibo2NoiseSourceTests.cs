using System.Text;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Pipelines;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The saved-noise reader behind diagnostics.fibo2NoiseNpy: a round trip, and each refusal (shape, dtype, order, length).</summary>
public sealed class Fibo2NoiseSourceTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "fibo2-noise-" + Guid.NewGuid().ToString("N"));

    public Fibo2NoiseSourceTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        if (Directory.Exists(_dir))
            Directory.Delete(_dir, recursive: true);
    }

    [Fact]
    public void RoundTrip_ReturnsTheSameFloatsInTheSameOrder()
    {
        float[] values = [0.5f, -1.25f, 3f, 1e-7f, -0f, 42f];
        string path = Write("ok.npy", "'<f4'", "False", "(2, 3)", Bytes(values));

        Tensor tensor = Fibo2NoiseSource.LoadFloat32Npy(path, new TensorShape(2, 3));
        try
        {
            Assert.Equal(DType.F32, tensor.DType);
            Assert.Equal(values, tensor.AsReadOnlySpan<float>().ToArray());
        }
        finally
        {
            tensor.Dispose();
        }
    }

    [Fact]
    public void WrongShape_IsRefused()
    {
        string path = Write("shape.npy", "'<f4'", "False", "(3, 2)", Bytes(new float[6]));

        Assert.Throws<InvalidDataException>(() => Fibo2NoiseSource.LoadFloat32Npy(path, new TensorShape(2, 3)));
    }

    [Fact]
    public void WrongDtype_IsRefused()
    {
        string path = Write("dtype.npy", "'<f8'", "False", "(2, 3)", new byte[6 * 8]);

        Assert.Throws<InvalidDataException>(() => Fibo2NoiseSource.LoadFloat32Npy(path, new TensorShape(2, 3)));
    }

    [Fact]
    public void FortranOrder_IsRefused()
    {
        string path = Write("order.npy", "'<f4'", "True", "(2, 3)", Bytes(new float[6]));

        Assert.Throws<InvalidDataException>(() => Fibo2NoiseSource.LoadFloat32Npy(path, new TensorShape(2, 3)));
    }

    [Fact]
    public void ShortData_IsRefused()
    {
        string path = Write("short.npy", "'<f4'", "False", "(2, 3)", Bytes(new float[5]));

        Assert.Throws<InvalidDataException>(() => Fibo2NoiseSource.LoadFloat32Npy(path, new TensorShape(2, 3)));
    }

    [Fact]
    public void NotAnNpyFile_IsRefused()
    {
        string path = Path.Combine(_dir, "junk.npy");
        File.WriteAllBytes(path, new byte[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });

        Assert.Throws<InvalidDataException>(() => Fibo2NoiseSource.LoadFloat32Npy(path, new TensorShape(2, 3)));
    }

    /// <summary>Writes a version 1.0 .npy whose header is padded to a 64-byte boundary, as numpy writes it.</summary>
    private string Write(string name, string descr, string fortran, string shape, byte[] data)
    {
        string header = $"{{'descr': {descr}, 'fortran_order': {fortran}, 'shape': {shape}, }}";
        int unpadded = 10 + header.Length + 1;
        int padding = (64 - unpadded % 64) % 64;
        string padded = header + new string(' ', padding) + "\n";

        using MemoryStream stream = new();
        stream.Write(new byte[] { 0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y', 1, 0 });
        stream.Write(BitConverter.GetBytes((ushort)padded.Length));
        stream.Write(Encoding.ASCII.GetBytes(padded));
        stream.Write(data);

        string path = Path.Combine(_dir, name);
        File.WriteAllBytes(path, stream.ToArray());
        return path;
    }

    private static byte[] Bytes(float[] values)
    {
        byte[] bytes = new byte[values.Length * sizeof(float)];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return bytes;
    }
}
