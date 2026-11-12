using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ThreeD.Models.Hunyuan3D.Omni;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests.Omni;

/// <summary>The Omni control encoder: the control-to-rows steps without weights, and the encoder against upstream's layers on the CPU with the real <c>cond_encoder/pytorch_model.bin</c>
/// (set <c>HARTSY_OMNI_COND</c> and <c>HARTSY_OMNI_CONTROL_REF</c>, the fixture of <c>tests/python-reference/dump_omni_control.py</c>).</summary>
public sealed class OmniControlEncoderTests(ITestOutputHelper output)
{
    [Fact]
    public void Rows_RepeatThreeValuesToSixAndKeepPoseBonesAsGiven()
    {
        Assert.Equal([1f, 2f, 3f, 1f, 2f, 3f, 4f, 5f, 6f, 4f, 5f, 6f], OmniControlEncoder.PointRows([1, 2, 3, 4, 5, 6]));
        Assert.Equal([0.5f, 0.8f, 0.3f, 0.5f, 0.8f, 0.3f], OmniControlEncoder.BoundingBoxRow(0.5f, 0.8f, 0.3f));
        Assert.Equal([1f, 2f, 3f, 4f, 5f, 6f], OmniControlEncoder.PoseRows([1, 2, 3, 4, 5, 6]));
        Assert.Throws<ArgumentException>(() => OmniControlEncoder.PointRows([1, 2]));
        Assert.Throws<ArgumentException>(() => OmniControlEncoder.PoseRows([1, 2, 3]));
        Assert.Throws<ArgumentException>(() => OmniControlEncoder.PointRows([]));
    }

    [Fact]
    public void VoxelRows_AreTheOccupiedVoxelCentresInAscendingGridOrder()
    {
        // Voxel size 0.125: x in [-1, -0.875) is cell 0 with centre -0.9375; the origin falls in cell 8 with centre 0.0625.
        float[] rows = OmniControlEncoder.VoxelRows([0.01f, 0.01f, 0.01f, -0.99f, -0.99f, -0.99f, -0.98f, -0.97f, -0.96f, 0.99f, -0.99f, -0.99f]);

        Assert.Equal(3 * 6, rows.Length); // three distinct voxels: (0,0,0) twice, (8,8,8), (15,0,0)
        Assert.Equal(-0.9375f, rows[0], 5);                 // cell (0,0,0) first: linear index 0
        Assert.Equal(0.9375f, rows[6], 5);                  // (15, 0, 0): linear index 15, before (8, 8, 8) = 8 + 128 + 2048
        Assert.Equal(-0.9375f, rows[7], 5);
        Assert.Equal(0.0625f, rows[12], 5);                 // (8, 8, 8)
        for (int i = 0; i < rows.Length; i += 6)
        {
            Assert.Equal(rows[i], rows[i + 3]);
            Assert.Equal(rows[i + 1], rows[i + 4]);
            Assert.Equal(rows[i + 2], rows[i + 5]);
        }

        // Points outside [-1, 1] clamp into the border cells.
        float[] clamped = OmniControlEncoder.VoxelRows([5f, -5f, 0f]);
        Assert.Equal(0.9375f, clamped[0], 5);
        Assert.Equal(-0.9375f, clamped[1], 5);
        Assert.Throws<ArgumentException>(() => OmniControlEncoder.VoxelRows([1, 2]));
    }

    private static unsafe float[] Values(Tensor t)
    {
        float[] values = new float[t.ElementCount];
        new ReadOnlySpan<float>((float*)t.DataPointer, values.Length).CopyTo(values);
        return values;
    }

    [RealWeightsFact("HARTSY_OMNI_COND", "HARTSY_OMNI_CONTROL_REF")]
    [Trait("Category", "Integration")]
    public void Encoder_MatchesUpstreamForEveryControlKind()
    {
        using PytorchPickleLoader pickle = new();
        pickle.Load(Environment.GetEnvironmentVariable("HARTSY_OMNI_COND")!);
        using SafeTensorsLoader refLoader = new();
        refLoader.Load(Environment.GetEnvironmentVariable("HARTSY_OMNI_CONTROL_REF")!);
        Dictionary<string, Tensor> fixture = refLoader.GetAllTensors();
        using CpuBackend backend = new();
        using OmniControlEncoder encoder = new(backend, pickle.GetAllTensors());

        foreach ((string name, OmniControlKind kind, Func<float[], float[]> rows) in new (string, OmniControlKind, Func<float[], float[]>)[]
        {
            ("pose", OmniControlKind.Pose, v => OmniControlEncoder.PoseRows(v)),
            ("bbox", OmniControlKind.BoundingBox, v => OmniControlEncoder.BoundingBoxRow(v[0], v[1], v[2])),
            ("voxel", OmniControlKind.Voxel, v => OmniControlEncoder.VoxelRows(v)),
            ("point", OmniControlKind.Point, v => OmniControlEncoder.PointRows(v)),
        })
        {
            float[] input = Values(fixture[$"{name}.in"]), expected = Values(fixture[$"{name}.out"]);
            using Tensor tokens = encoder.Encode(kind, rows(input));
            float[] actual = Values(tokens);
            Assert.Equal(expected.Length, actual.Length);
            double diff = 0, reference = 0, max = 0;
            for (int i = 0; i < expected.Length; i++)
            {
                double d = (double)actual[i] - expected[i];
                diff += d * d;
                reference += (double)expected[i] * expected[i];
                max = Math.Max(max, Math.Abs(d));
            }

            double rel = Math.Sqrt(diff / reference);
            output.WriteLine($"{name,-6} tokens {tokens.Shape[1],5}  max|diff|={max:E3}  relRMS={rel:E3}");
            Assert.True(rel < 1e-5, $"{name}: relRMS {rel:E3}");
        }
    }
}
