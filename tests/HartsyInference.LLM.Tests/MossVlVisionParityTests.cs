using System.Text;
using System.Text.RegularExpressions;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.TextEncoders;

namespace HartsyInference.LLM.Tests;

/// <summary>Vision tower against the numpy transcription in <c>tests/python-reference/dump_moss_vl_vision.py</c>, on the
/// real checkpoint. <c>HARTSY_MOSS_VL_INDEX</c> is the checkpoint's <c>model.safetensors.index.json</c>;
/// <c>HARTSY_MOSS_VL_REF</c> is the dump's <c>merged.npy</c> (<c>patches.npy</c> sits beside it).</summary>
[Trait("Category", "Integration")]
public sealed class MossVlVisionParityTests
{
    private const string CheckpointVariable = "HARTSY_MOSS_VL_INDEX";
    private const string ReferenceVariable = "HARTSY_MOSS_VL_REF";

    /// <summary>Largest allowed <c>max|diff| / max|ref|</c>. The tower stores BF16 weights and the reference runs in F32.</summary>
    private const double RelativeTolerance = 1e-2;

    [RealWeightsFact(CheckpointVariable, ReferenceVariable)]
    public void MergedTokensMatchTheReference()
    {
        string checkpoint = Path.GetDirectoryName(Environment.GetEnvironmentVariable(CheckpointVariable)!)!;
        string mergedPath = Environment.GetEnvironmentVariable(ReferenceVariable)!;
        string referenceDir = Path.GetDirectoryName(mergedPath)!;

        (long[] patchShape, float[] patchData) = ReadNpy(Path.Combine(referenceDir, "patches.npy"));
        (long[] refShape, float[] refData) = ReadNpy(mergedPath);
        Assert.Equal(2, patchShape.Length);
        Assert.Equal(2, refShape.Length);

        using MossVlVisionTower tower = MossVlVisionTower.Open(checkpoint);
        using IBackend backend = new CpuBackend();
        using Tensor patches = new(new TensorShape(patchShape[0], patchShape[1]), DType.F32);
        patchData.AsSpan().CopyTo(patches.AsSpan<float>());

        // Grid from the dump's meta.json: 1 x 16 x 24 patches.
        Qwen3VlVisionEncoder.VisionOutput output = tower.Encoder.Forward(backend, patches, 1, 16, 24);
        try
        {
            Assert.Equal(refShape[0], output.MergedTokens.Shape[0]);
            Assert.Equal(refShape[1], output.MergedTokens.Shape[1]);

            ReadOnlySpan<float> actual = output.MergedTokens.AsSpan<float>();
            double maxDiff = 0, maxRef = 0;
            for (int i = 0; i < refData.Length; i++)
            {
                maxDiff = Math.Max(maxDiff, Math.Abs(actual[i] - refData[i]));
                maxRef = Math.Max(maxRef, Math.Abs(refData[i]));
            }
            double relative = maxDiff / maxRef;
            Console.WriteLine($"MOSS-VL vision parity: max|diff| {maxDiff:G6}, max|ref| {maxRef:G6}, relative {relative:G6}.");
            Assert.True(relative < RelativeTolerance,
                $"MOSS-VL vision parity: max|diff| {maxDiff:G6}, max|ref| {maxRef:G6}, relative {relative:G6} >= {RelativeTolerance}.");
        }
        finally
        {
            output.MergedTokens.Dispose();
        }
    }

    /// <summary>Reads a little-endian float32 <c>.npy</c> (version 1 or 2 header, C order).</summary>
    private static (long[] shape, float[] data) ReadNpy(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        int headerLength = bytes[8] | (bytes[9] << 8);
        int dataOffset = 10 + headerLength;
        string header = Encoding.ASCII.GetString(bytes, 10, headerLength);
        Assert.Contains("'<f4'", header);
        Assert.Contains("'fortran_order': False", header);

        Match shapeMatch = Regex.Match(header, @"'shape':\s*\(([^)]*)\)");
        long[] shape = shapeMatch.Groups[1].Value
            .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(long.Parse).ToArray();

        float[] data = new float[(bytes.Length - dataOffset) / sizeof(float)];
        Buffer.BlockCopy(bytes, dataOffset, data, 0, data.Length * sizeof(float));
        return (shape, data);
    }
}
