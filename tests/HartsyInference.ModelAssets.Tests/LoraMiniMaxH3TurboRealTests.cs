using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.Lora;
using HartsyInference.Tests.Common;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>The real published MiniMax-H3 LoRA (larryvrh/MiniMax-H3-Turbo-Lora), against real weight shapes. The
/// synthetic tests next door pin the mapping rules; this pins that the actual file the world ships loads and merges,
/// because the mapping rules were written from a hand-built LoRA and the real one turned out not to match them.
/// <para>The merge is exercised one module at a time rather than end-to-end: merging all 259 targets into the
/// unpruned bf16 build would materialize ~60 GB of owned host tensors (the mmap'd originals are replaced), which does
/// not fit this box's 62 GB. That is a property of the build, not of these tests — see the RAM note in
/// <c>docs/Checklists/MODEL_STATUS_VIDEO.md</c>.</para></summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed unsafe class LoraMiniMaxH3TurboRealTests
{
    private readonly ITestOutputHelper _output;
    public LoraMiniMaxH3TurboRealTests(ITestOutputHelper output) => _output = output;

    /// <summary>The whole file: it must detect as the bare-root format and expose every one of its 259 modules.</summary>
    [Fact]
    public void RealTurboLora_LoadsAsBareDit_WithEveryModuleMapped()
    {
        if (!RealWeightGate.Require(_output.WriteLine, TestPaths.MiniMaxH3.TurboLora)) return;

        using LoraFile file = LoraFile.Load(TestPaths.MiniMaxH3.TurboLora);
        Assert.Equal(LoraFormat.DiffusersBareDit, file.Format);
        Assert.Equal(259, file.Layers.Count);
        Assert.All(file.Layers, l => Assert.Equal(LoraTarget.Transformer, l.Target));

        // Every target must be a canonical checkpoint key: the roots are used verbatim, so a stray prefix or a
        // missing `.weight` would silently match nothing and merge zero.
        Assert.All(file.Layers, l => Assert.EndsWith(".weight", l.TargetKey, StringComparison.Ordinal));
        Assert.Contains(file.Layers, l => l.TargetKey == "blocks.0.attn.qkv_proj.weight");
        Assert.Contains(file.Layers, l => l.TargetKey == "blocks.49.mlp.fc2.weight");
        Assert.Contains(file.Layers, l => l.TargetKey == "final_layer.adaln_proj.linear.weight");
        Assert.Contains(file.Layers, l => l.TargetKey == "token_refiner.blocks.0.attn.qkv_proj.weight");
        _output.WriteLine($"{file.Layers.Count} modules, format {file.Format}.");
    }

    /// <summary>The 51 adaln modules a pruned checkpoint skips are the ones a step-distillation LoRA leans on, so
    /// "they would merge if the shapes lined up" cannot be left as an assumption. Merges the real adaln tensors into
    /// an UNPRUNED-shaped weight and checks the delta actually landed and is finite.</summary>
    [Fact]
    public void RealTurboLora_AdalnMergesIntoUnprunedShapedWeight()
    {
        if (!RealWeightGate.Require(_output.WriteLine, TestPaths.MiniMaxH3.TurboLora)) return;

        const string key = "blocks.0.adaln_proj.linear.weight";
        const int rows = 96768, cols = 2688;   // the UNPRUNED projection; pruned builds store [96768, 8]
        using LoraFile file = LoraFile.Load(TestPaths.MiniMaxH3.TurboLora);
        LoraLayer adaln = file.Layers.Single(l => l.TargetKey == key);
        Assert.Equal(rows, (int)adaln.LoraUp.Shape[0]);
        Assert.Equal(cols, (int)adaln.LoraDown.Shape[1]);

        Tensor baseW = new Tensor(new TensorShape(rows, cols), DType.F32);
        new Span<float>((void*)baseW.DataPointer, rows * cols).Clear();
        Dictionary<string, Tensor> weights = new Dictionary<string, Tensor> { [key] = baseW };

        IBackend backend = new CpuBackend();
        using LoraStack stack = new LoraStack();
        stack.AddFromPath(TestPaths.MiniMaxH3.TurboLora, strength: 1.0f);
        // Only this one key is present, so every other module reports "not present" and this returns exactly 1.
        Assert.Equal(1, stack.ApplyTo(weights, LoraTarget.Transformer, backend));

        float* merged = (float*)weights[key].DataPointer;
        long n = (long)rows * cols;
        double sumAbs = 0;
        float maxAbs = 0;
        for (long i = 0; i < n; i++)
        {
            float v = merged[i];
            Assert.True(float.IsFinite(v), $"non-finite at {i}");
            float a = Math.Abs(v);
            sumAbs += a;
            if (a > maxAbs) { maxAbs = a; }
        }
        // The base was all zeros, so whatever is here IS the LoRA delta. A merge that silently did nothing would
        // leave this at exactly zero — the failure this test exists to catch.
        Assert.True(maxAbs > 0, "the adaln delta did not land: the merged weight is still all zeros");
        _output.WriteLine($"adaln delta into [{rows}, {cols}]: max|d|={maxAbs:G6}, mean|d|={sumAbs / n:G6}.");
        baseW.Dispose();
    }
}
