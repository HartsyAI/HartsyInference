using System.Runtime.InteropServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Core.Tensors.Quant;
using HartsyInference.Cuda;
using HartsyInference.ModelAssets.Quant;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>M1: routed experts of official shard 3 (layers.0, experts 0/191/383, 64-row windows) uploaded through <see cref="CudaExpertCache"/> and
/// dequantized on the device, bit-compared with the official convert.py fixtures. Needs the real shard or a sparse replica holding those windows.</summary>
[Collection("CudaSerial")]
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
[Trait("Category", "GpuIntegration")]
public sealed unsafe class CudaExpertM1FixtureTests
{
    private const string ShardName = "model-00003-of-00048.safetensors";
    private static readonly int[] Experts = [0, 191, 383];
    private static readonly string[] Projections = ["w1", "w2", "w3"];

    private readonly ITestOutputHelper _output;

    public CudaExpertM1FixtureTests(ITestOutputHelper output) => _output = output;

    private static string ShardPath => Path.Combine(
        Environment.GetEnvironmentVariable("HARTSY_DSV41_FLASH_DIR")
        ?? Path.Combine(TestPaths.ModelsDir, "DeepSeek-V4.1-Flash"), ShardName);

    private static string FixtureDir => Environment.GetEnvironmentVariable("HARTSY_DSV41_SHARD3_FIXTURES")
        ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "dsv41-ref", "shard3_fixtures");

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    [Fact]
    public void Layer0RoutedExperts_UploadAndDeviceDequant_MatchOfficialFixtures()
    {
        string manifest = Path.Combine(FixtureDir, "manifest.tsv");
        if (!RealWeightGate.Require(_output.WriteLine, ShardPath, manifest)) return;
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: no CUDA device"); return; }

        using ShardedSafeTensorSet set = ShardedSafeTensorSet.OpenFiles([ShardPath]);
        QuantBindingSet bindings = QuantCompanionBinder.Bind(set.Inventory, QuantFlavor.Official);
        Dictionary<string, (int Rows, int Cols)> windows = ReadWindows(manifest);

        ExpertMatrix Window(int expert, string projection)
        {
            string key = $"layers.0.ffn.experts.{expert}.{projection}";
            (int rows, _) = windows[key];
            QuantBinding binding = bindings.Bindings[key + ".weight"];
            QuantRecipe recipe = binding.ToRecipe(set.GetTensor).SliceRows(0, rows, key);
            return new ExpertMatrix(set.GetTensor(binding.WeightKey).SliceRows(0, rows), recipe);
        }

        ExpertBank bank = new(0, Experts.Length, key =>
            new ExpertWeights(key, Window(Experts[key.Expert], "w1"), Window(Experts[key.Expert], "w2"), Window(Experts[key.Expert], "w3")));
        using CudaBackend backend = new(0, PtxDir());
        _output.WriteLine($"device: {backend.Capabilities.Name}");
        using CudaExpertCache cache = new(backend, 16L << 20, [bank]);

        using ExpertLease lease = cache.Acquire([new ExpertKey(0, 0), new ExpertKey(0, 1), new ExpertKey(0, 2)]);
        int checkedMatrices = 0;
        for (int e = 0; e < Experts.Length; e++)
        {
            ExpertWeights weights = lease.Get(new ExpertKey(0, e));
            ExpertMatrix[] matrices = [weights.W1, weights.W2, weights.W3];
            for (int p = 0; p < 3; p++)
            {
                string name = $"layers.0.ffn.experts.{Experts[e]}.{Projections[p]}@0";
                float[] expected = MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(Path.Combine(FixtureDir, name + ".w.f32"))).ToArray();
                using QuantWorkspaceLease dense = backend.QuantWorkspace.Dequantize(matrices[p]);
                ushort[] want = expected.Select(CudaQuantWorkspaceTests.ToBf16).ToArray();
                CudaQuantWorkspaceTests.AssertBitExact(name, want, CudaQuantWorkspaceTests.ReadBack(backend, dense));
                checkedMatrices++;
            }
        }
        _output.WriteLine($"{checkedMatrices} expert matrices (3 experts x w1/w2/w3, 64-row windows) bit-exact after upload + device dequant");
        Assert.Equal(9, checkedMatrices);
    }

    private static Dictionary<string, (int Rows, int Cols)> ReadWindows(string manifest)
    {
        Dictionary<string, (int, int)> windows = new();
        foreach (string line in File.ReadLines(manifest).Skip(1))
        {
            string[] f = line.Split('\t');
            if (f[1] != "fp4" || f[2] != "0") continue;
            windows[f[0][..f[0].IndexOf('@')]] = (int.Parse(f[3]), int.Parse(f[4]));
        }
        return windows;
    }
}
