using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ThreeD.Models.Dora;
using HartsyInference.ThreeD.Models.HairDeck;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Parity;

/// <summary>Real-weight parity of the Dora-VAE 1.1 encoder against <c>tools/parity/dump_dora_encoder.py</c> (CPU, fp32).
/// The reference runs the official encoder on a seeded cloud; the test runs the same cloud through the C# encoder with the
/// same released weights and compares the 512 condition tokens with a peak-relative tolerance.</summary>
[Trait("Category", "Integration")]
public sealed class DoraEncoderParityTests
{
    private const double TokenTolerance = 1e-4;
    private const string KeyPrefix = "state_dict.shape_model.";

    [DoraRealWeightsFact]
    public void ConditionTokens_MatchReference()
    {
        string ckpt = Environment.GetEnvironmentVariable("DORA_CKPT")!;
        string reference = Environment.GetEnvironmentVariable("DORA_REFERENCE")!;
        HairDeckReferenceDump dump = HairDeckReferenceDump.Open(reference);
        int tokensPerBranch = dump.Shape("dora_tokens")[0] / 2;

        using PytorchPickleLoader loader = new();
        loader.Load(ckpt, recursiveFlatten: true);
        IReadOnlyDictionary<string, Tensor> all = loader.GetAllTensors();
        Dictionary<string, Tensor> shape = all
            .Where(kv => kv.Key.StartsWith(KeyPrefix, StringComparison.Ordinal))
            .ToDictionary(kv => kv.Key[KeyPrefix.Length..], kv => kv.Value, StringComparer.Ordinal);
        using DoraShapeEncoder encoder = new(HartsyInference.ModelAssets.CheckpointConverters.DoraShapeEncoderCheckpointConverter.Convert(shape));

        float[] points = dump.Floats("points");
        using IBackend backend = new CpuBackend();
        using Tensor tokens = encoder.Encode(backend, points, tokensPerBranch);

        double[] got = HairDeckNumerics.ToArray(tokens).Select(v => (double)v).ToArray();
        double[] want = dump.Doubles("dora_tokens");
        Assert.Equal(want.Length, got.Length);
        double peak = want.Max(v => Math.Abs(v));
        double worst = 0;
        for (int i = 0; i < want.Length; i++)
        {
            worst = Math.Max(worst, Math.Abs(got[i] - want[i]));
        }
        Assert.True(worst / peak <= TokenTolerance, $"Dora tokens: max abs error {worst:E3} over peak {peak:E3} (relative {worst / peak:E3}).");
    }
}
