using System.Globalization;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ThreeD.Models.Idol;
using HartsyInference.ThreeD.Models.Idol.Budget;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests.Parity;

/// <summary>Budget3D's octree tables, salience and layouts against the reference (<c>dump_budget3d.py</c> on BudgetAny's code at <c>95bbeab</c>): the Sobol/octree bijection, the cube table and centres, the salience of
/// the saved latent, and for every budget the layout built from that salience and from seeded random scores. Indices are integers, so they match exactly. Needs the dump folder (<c>HARTSY_IDOL_REF</c>); a missing
/// input skips, and with <c>HARTSY_REQUIRE_REAL_WEIGHTS=1</c> it fails.</summary>
[Trait("Category", "Integration")]
public sealed class BudgetLayoutParityTests
{
    private const string ReferenceEnv = "HARTSY_IDOL_REF";
    private const string RequireEnv = "HARTSY_REQUIRE_REAL_WEIGHTS";

    private readonly ITestOutputHelper _output;

    public BudgetLayoutParityTests(ITestOutputHelper output) => _output = output;

    private bool TryLoad(out Dictionary<string, Tensor> octree, out Dictionary<string, Tensor> layouts, out Dictionary<string, Tensor> inputs, out List<IDisposable> owners)
    {
        octree = layouts = inputs = [];
        List<IDisposable> opened = [];
        owners = opened;
        string? refDir = Environment.GetEnvironmentVariable(ReferenceEnv);
        string[] files = [Path.Combine(refDir ?? "", "b3d", "octree.safetensors"), Path.Combine(refDir ?? "", "b3d", "layouts.safetensors"), Path.Combine(refDir ?? "", "den", "denoiser_inputs.safetensors")];
        string[] missing = files.Where(f => !File.Exists(f)).ToArray();
        if (missing.Length > 0)
        {
            string message = $"required real-weight asset(s) missing: {string.Join(", ", missing)}";
            if (Environment.GetEnvironmentVariable(RequireEnv) == "1")
                throw new InvalidOperationException($"{RequireEnv}=1 but {message}.");
            _output.WriteLine($"SKIPPED: {message}");
            return false;
        }

        Dictionary<string, Tensor> Open(string path)
        {
            SafeTensorsLoader loader = new();
            loader.Load(path);
            opened.Add(loader);
            return loader.GetAllTensors();
        }

        octree = Open(files[0]);
        layouts = Open(files[1]);
        inputs = Open(files[2]);
        return true;
    }

    [Fact]
    public void TheOctreeTablesMatchTheReference()
    {
        if (!TryLoad(out Dictionary<string, Tensor> octree, out _, out _, out List<IDisposable> owners))
            return;
        using (new Owners(owners))
        {
            (int[] toOctree, int[] fromOctree) = BudgetOctree.Permutation(IdolWeights.EmbeddedPositions());

            Assert.Equal(Ints(octree["perm_to_octree"]), toOctree);
            Assert.Equal(Ints(octree["inv_perm"]), fromOctree);
            Assert.Equal(Ints(octree["cube_to_cells"]), BudgetOctree.CubeToCells());
            Assert.Equal(octree["cell_ctr"].AsReadOnlySpan<float>().ToArray(), BudgetOctree.CellCenters());
            Assert.Equal(octree["cube_ctr"].AsReadOnlySpan<float>().ToArray(), BudgetOctree.CubeCenters());
            _output.WriteLine("perm_to_octree, inv_perm, cube_to_cells, cell and cube centres: identical");
        }
    }

    [Fact]
    public void TheSalienceOfTheSavedLatentMatchesTheReference()
    {
        if (!TryLoad(out Dictionary<string, Tensor> octree, out Dictionary<string, Tensor> layouts, out Dictionary<string, Tensor> inputs, out List<IDisposable> owners))
            return;
        using (new Owners(owners))
        {
            (float[] cube, float[] cell) = Scores(octree, inputs);

            Compare("cube salience", cube, layouts["score4"].AsReadOnlySpan<float>());
            Compare("cell salience", cell, layouts["score2"].AsReadOnlySpan<float>());
        }
    }

    [Theory]
    [InlineData("")]
    [InlineData("rand_")]
    public void EveryBudgetsLayoutMatchesTheReferenceExactly(string source)
    {
        if (!TryLoad(out Dictionary<string, Tensor> octree, out Dictionary<string, Tensor> layouts, out Dictionary<string, Tensor> inputs, out List<IDisposable> owners))
            return;
        using (new Owners(owners))
        {
            float[] cube, cell;
            if (source.Length == 0)
                (cube, cell) = Scores(octree, inputs);
            else
            {
                cube = layouts["rand_score4"].AsReadOnlySpan<float>().ToArray();
                cell = layouts["rand_score2"].AsReadOnlySpan<float>().ToArray();
            }

            foreach (double budget in new[] { 1.0, 0.5, 0.25, 0.1, 0.0625 })
            {
                string key = ("b" + budget.ToString("0.####", CultureInfo.InvariantCulture)).Replace(".", "p");
                BudgetLayout got = BudgetLayout.Build(cube, cell, BudgetLayout.SnapTokens(budget * 8192));
                long[] k = layouts[$"{source}{key}_k4k2"].AsReadOnlySpan<long>().ToArray();

                _output.WriteLine($"{(source.Length == 0 ? "salience" : "random")} budget {budget}: {got.Tokens} tokens, k4 {got.K4}, k2 {got.K2}");
                Assert.Equal((int)k[0], got.K4);
                Assert.Equal((int)k[1], got.K2);
                Assert.Equal(Ints(layouts[$"{source}{key}_idx4"]), got.Cubes);
                Assert.Equal(Ints(layouts[$"{source}{key}_idx2"]), got.Cells);
                Assert.Equal(Ints(layouts[$"{source}{key}_idx1"]), got.Pairs);
            }
        }
    }

    private static (float[] Cube, float[] Cell) Scores(Dictionary<string, Tensor> octree, Dictionary<string, Tensor> inputs)
    {
        int[] toOctree = Ints(octree["perm_to_octree"]);
        ReadOnlySpan<float> z = inputs["latent"].AsReadOnlySpan<float>();
        float[] inOctree = new float[8192 * 16];
        for (int k = 0; k < 8192; k++)
            z.Slice(toOctree[k] * 16, 16).CopyTo(inOctree.AsSpan(k * 16, 16));
        return BudgetOctree.DetailScores(inOctree, 16);
    }

    private void Compare(string name, float[] got, ReadOnlySpan<float> want)
    {
        Assert.Equal(want.Length, got.Length);
        double worst = 0, peak = 0;
        for (int i = 0; i < got.Length; i++)
        {
            worst = Math.Max(worst, Math.Abs(got[i] - want[i]));
            peak = Math.Max(peak, Math.Abs(want[i]));
        }

        _output.WriteLine(string.Create(CultureInfo.InvariantCulture, $"{name}: max abs {worst:E2} of a peak {peak:E2}"));
        Assert.True(worst <= 1e-5 * peak, $"{name} differs by {worst:E2}");
    }

    private static int[] Ints(Tensor t) => [.. t.AsReadOnlySpan<long>().ToArray().Select(v => (int)v)];

    private sealed class Owners(List<IDisposable> items) : IDisposable
    {
        public void Dispose()
        {
            foreach (IDisposable item in items)
                item.Dispose();
        }
    }
}
