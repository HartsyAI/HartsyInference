using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Vulkan;

namespace HartsyInference.LLM.Tests;

/// <summary>Vision tower against the numpy transcription in <c>tests/python-reference/dump_moss_vl_vision.py</c>, on the
/// real checkpoint. <c>HARTSY_MOSS_VL_INDEX</c> is the checkpoint's <c>model.safetensors.index.json</c>;
/// <c>HARTSY_MOSS_VL_REF</c> is the dump's <c>merged.npy</c> (<c>patches.npy</c> sits beside it); <c>HARTSY_MOSS_VL_GGUF</c> is the
/// MOSS-VL GGUF whose own vision blocks (F16, padded qkv) are the second source.</summary>
[Trait("Category", "Integration")]
public sealed class MossVlVisionParityTests
{
    private const string CheckpointVariable = "HARTSY_MOSS_VL_INDEX";
    private const string ReferenceVariable = "HARTSY_MOSS_VL_REF";
    private const string GgufVariable = "HARTSY_MOSS_VL_GGUF";

    /// <summary>Largest allowed <c>max|diff| / max|ref|</c>. The tower stores BF16 (or F16) weights and the reference runs in F32.</summary>
    private const double RelativeTolerance = 1e-2;

    [RealWeightsFact(CheckpointVariable, ReferenceVariable)]
    public void MergedTokensMatchTheReference()
    {
        using IBackend backend = new CpuBackend();
        using MossVlVisionTower tower = MossVlVisionTower.Open(Checkpoint());
        AssertMatchesReference(tower, backend, "safetensors CPU");
    }

    /// <summary>A GGUF-only user's tower: the vision blocks inside the MOSS-VL GGUF, with no safetensors checkpoint.</summary>
    [RealWeightsFact(GgufVariable, ReferenceVariable)]
    public void GgufTowerMatchesTheReference()
    {
        using IBackend backend = new CpuBackend();
        using MossVlVisionTower tower = MossVlVisionTower.OpenGguf(Environment.GetEnvironmentVariable(GgufVariable)!);
        AssertMatchesReference(tower, backend, "GGUF CPU");
    }

    [Trait("Category", "GpuIntegration")]
    [RealWeightsFact(CheckpointVariable, ReferenceVariable)]
    public void Bf16TowerOnVulkanMatchesTheReference()
    {
        using IBackend backend = new VulkanBackend();
        using MossVlVisionTower tower = MossVlVisionTower.Open(Checkpoint(), backend);
        AssertMatchesReference(tower, backend, "safetensors Vulkan");
    }

    [Trait("Category", "GpuIntegration")]
    [RealWeightsFact(GgufVariable, ReferenceVariable)]
    public void GgufTowerOnVulkanMatchesTheReference()
    {
        using IBackend backend = new VulkanBackend();
        using MossVlVisionTower tower = MossVlVisionTower.OpenGguf(Environment.GetEnvironmentVariable(GgufVariable)!, backend);
        AssertMatchesReference(tower, backend, "GGUF Vulkan");
    }

    private static string Checkpoint() => Path.GetDirectoryName(Environment.GetEnvironmentVariable(CheckpointVariable)!)!;

    private static void AssertMatchesReference(MossVlVisionTower tower, IBackend backend, string label)
    {
        string mergedPath = Environment.GetEnvironmentVariable(ReferenceVariable)!;
        string referenceDir = Path.GetDirectoryName(mergedPath)!;
        (long[] patchShape, float[] patchData) = NpyReader.ReadFloat32(Path.Combine(referenceDir, "patches.npy"));
        (long[] refShape, float[] refData) = NpyReader.ReadFloat32(mergedPath);
        Assert.Equal(2, patchShape.Length);
        Assert.Equal(2, refShape.Length);

        using Tensor patches = new(new TensorShape(patchShape[0], patchShape[1]), DType.F32);
        patchData.AsSpan().CopyTo(patches.AsSpan<float>());

        // Grid from the dump's meta.json: 1 x 16 x 24 patches.
        Qwen3VlVisionEncoder.VisionOutput output = tower.Encoder.Forward(backend, patches, 1, 16, 24);
        try
        {
            Assert.Equal(refShape[0], output.MergedTokens.Shape[0]);
            Assert.Equal(refShape[1], output.MergedTokens.Shape[1]);
            ReadOnlySpan<float> actual = output.MergedTokens.AsSpan<float>();   // brings a device result back to the host
            double maxDiff = 0, maxRef = 0;
            for (int i = 0; i < refData.Length; i++)
            {
                maxDiff = Math.Max(maxDiff, Math.Abs(actual[i] - refData[i]));
                maxRef = Math.Max(maxRef, Math.Abs(refData[i]));
            }
            double relative = maxDiff / maxRef;
            Console.WriteLine($"MOSS-VL vision parity ({label}): max|diff| {maxDiff:G6}, max|ref| {maxRef:G6}, relative {relative:G6}.");
            Assert.True(relative < RelativeTolerance,
                $"MOSS-VL vision parity ({label}): max|diff| {maxDiff:G6}, max|ref| {maxRef:G6}, relative {relative:G6} >= {RelativeTolerance}.");
        }
        finally
        {
            output.MergedTokens.Dispose();
        }
    }
}
