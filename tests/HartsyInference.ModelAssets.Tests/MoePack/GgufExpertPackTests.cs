using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.MoePack;
using Xunit;

namespace HartsyInference.ModelAssets.Tests.MoePack;

/// <summary>Packing a GGUF checkpoint's stacked expert tensors, read per expert, and verifying the pack against the same file.</summary>
public sealed class GgufExpertPackTests : IDisposable
{
    private const int Experts = 4;
    private const int Hidden = 64;
    private const int Intermediate = 32;
    private readonly string _root = Path.Combine(Path.GetTempPath(), "hartsy-gguf-pack-" + Guid.NewGuid().ToString("N"));

    public GgufExpertPackTests() => Directory.CreateDirectory(_root);

    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    private static float[] Values(int count, Random rng)
    {
        float[] values = new float[count];
        for (int i = 0; i < values.Length; i++) values[i] = (float)(rng.NextDouble() - 0.5);
        return values;
    }

    /// <summary>Writes a GGUF with separate or fused expert tensors for two layers (blk 0 and blk 2). Both forms hold the same values:
    /// the fused tensor is each expert's gate rows followed by its up rows.</summary>
    private string WriteCheckpoint(string name, bool fused, int experts = Experts, int intermediate = Intermediate, int seed = 1)
    {
        string path = Path.Combine(_root, name);
        Random rng = new(seed);
        List<Tensor> owned = [];
        try
        {
            using GgufWriter writer = new(path);
            writer.SetMetadata("general.architecture", "moetest");
            foreach (int layer in new[] { 0, 2 })
            {
                float[] gate = Values(experts * intermediate * Hidden, rng);
                float[] up = Values(experts * intermediate * Hidden, rng);
                float[] down = Values(experts * Hidden * intermediate, rng);
                if (fused)
                {
                    float[] gateUp = new float[2 * gate.Length];
                    for (int e = 0; e < experts; e++)
                    {
                        int matrix = intermediate * Hidden;
                        gate.AsSpan(e * matrix, matrix).CopyTo(gateUp.AsSpan(e * 2 * matrix, matrix));
                        up.AsSpan(e * matrix, matrix).CopyTo(gateUp.AsSpan(e * 2 * matrix + matrix, matrix));
                    }
                    owned.Add(FromValues(gateUp, [experts, 2 * intermediate, Hidden]));
                    writer.AddTensor($"blk.{layer}.ffn_gate_up_exps.weight", owned[^1]);
                }
                else
                {
                    owned.Add(FromValues(gate, [experts, intermediate, Hidden]));
                    owned.Add(FromValues(up, [experts, intermediate, Hidden]));
                    writer.AddTensor($"blk.{layer}.ffn_gate_exps.weight", owned[^2]);
                    writer.AddTensor($"blk.{layer}.ffn_up_exps.weight", owned[^1]);
                }
                owned.Add(FromValues(down, [experts, Hidden, intermediate]));
                writer.AddTensor($"blk.{layer}.ffn_down_exps.weight", owned[^1]);
            }
            writer.Flush();
        }
        finally
        {
            foreach (Tensor tensor in owned) tensor.Dispose();
        }
        return path;
    }

    private static Tensor FromValues(float[] values, long[] dims)
    {
        Tensor t = new(new TensorShape(dims), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }

    [Fact]
    public void Pack_RoundTripsEveryExpertWithinQuantizationError()
    {
        string gguf = WriteCheckpoint("separate.gguf", fused: false);
        string pack = Path.Combine(_root, "pack");

        GgufExpertPack.Write(gguf, pack, DType.Q8_0);
        ExpertPackVerification verification = GgufExpertPack.Verify(gguf, pack);

        Assert.Equal(2 * Experts, verification.Checked);
        Assert.Empty(verification.Failures);
        Assert.True(verification.RelativeRmse < 0.02, $"relative RMSE {verification.RelativeRmse}");
    }

    [Fact]
    public void FusedGateUp_ReadsTheSameExpertsAsSeparateTensors()
    {
        string separate = WriteCheckpoint("separate.gguf", fused: false);
        string fused = WriteCheckpoint("fused.gguf", fused: true);
        Assert.Equal(GetLayout(separate), GetLayout(fused));
    }

    /// <summary>Every expert's gate, up and down values, concatenated per expert, so the comparison sees the values themselves.</summary>
    private static List<float[]> GetLayout(string path)
    {
        using GgufExpertSource source = GgufExpertSource.Open(path);
        return source.Keys().Select(key =>
        {
            (float[] gate, float[] up, float[] down) = source.Read(key.Layer, key.Expert);
            return gate.Concat(up).Concat(down).ToArray();
        }).ToList();
    }

    [Fact]
    public void Verify_RefusesAPackBuiltForAnotherCheckpointGeometry()
    {
        string small = WriteCheckpoint("small.gguf", fused: false, experts: Experts, intermediate: Intermediate);
        string other = WriteCheckpoint("other.gguf", fused: false, experts: Experts, intermediate: Intermediate * 2);
        string pack = Path.Combine(_root, "pack");
        GgufExpertPack.Write(small, pack, DType.Q8_0);

        Assert.Throws<InvalidDataException>(() => GgufExpertPack.Verify(other, pack));
    }

    [Fact]
    public void ExplicitFingerprint_IsRefusedUnlessItMatches()
    {
        string gguf = WriteCheckpoint("explicit.gguf", fused: false);
        string pack = Path.Combine(_root, "pack-explicit");
        const string topology = "sparse-v1:0123456789abcdef";
        GgufExpertPack.Write(gguf, pack, DType.Q8_0, topology);

        Assert.Throws<InvalidDataException>(() => ExpertPackReader.Open(pack, "sparse-v1:fedcba9876543210"));
        Assert.Throws<InvalidDataException>(() => GgufExpertPack.Verify(gguf, pack, "sparse-v1:fedcba9876543210"));
        Assert.Throws<InvalidDataException>(() => GgufExpertPack.Verify(gguf, pack));

        using (ExpertPackReader reader = ExpertPackReader.Open(pack, topology))
            Assert.Equal(topology, reader.TopologyFingerprint);
        ExpertPackVerification verification = GgufExpertPack.Verify(gguf, pack, topology);
        Assert.Equal(2 * Experts, verification.Checked);
        Assert.Empty(verification.Failures);
    }
}
