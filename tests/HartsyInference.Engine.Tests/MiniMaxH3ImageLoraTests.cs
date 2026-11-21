using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Features;
using HartsyInference.ModelAssets.Lora;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Engine.Tests;

/// <summary>MiniMax-H3-Image LoRA merge on a tiny DiT geometry. An ai-toolkit style file must land on every module it names
/// and change the weight, and a delta sized for another build must be reported as a shape skip beside the ones that apply.</summary>
public sealed class MiniMaxH3ImageLoraTests : IDisposable
{
    private const int Hidden = 8;
    private const int Rank = 2;

    private readonly string _folder = Path.Combine(Path.GetTempPath(), "h3-image-lora-" + Guid.NewGuid().ToString("N"));

    public MiniMaxH3ImageLoraTests() => Directory.CreateDirectory(_folder);

    public void Dispose()
    {
        if (Directory.Exists(_folder))
        {
            Directory.Delete(_folder, recursive: true);
        }
    }

    [Fact]
    public void AiToolkitLoraLandsOnEveryNamedModuleAndChangesIt()
    {
        string lora = WriteLora("skeletor.safetensors",
            ("blocks.0.attn.qkv_proj", Hidden * 3),
            ("blocks.0.attn.out_proj", Hidden),
            ("blocks.0.mlp.fc1", Hidden * 2));
        using IBackend backend = new CpuBackend();
        using Tensor qkv = Base(Hidden * 3, Hidden);
        using Tensor outProj = Base(Hidden, Hidden);
        using Tensor fc1 = Base(Hidden * 2, Hidden);
        float[] qkvBefore = qkv.AsSpan<float>().ToArray();
        float[] outBefore = outProj.AsSpan<float>().ToArray();
        float[] fc1Before = fc1.AsSpan<float>().ToArray();
        Dictionary<string, Tensor> weights = new(StringComparer.Ordinal)
        {
            ["blocks.0.attn.qkv_proj.weight"] = qkv,
            ["blocks.0.attn.out_proj.weight"] = outProj,
            ["blocks.0.mlp.fc1.weight"] = fc1,
        };

        using LoraStack? stack = RecipeLoraMerge.Apply([Spec(lora)], backend,
            new LoraMergeTargets { Transformer = weights }, "MiniMaxH3ImageLoraTests");

        Assert.NotNull(stack);
        Assert.Equal(3, stack.Report.Applied);
        Assert.Equal(0, stack.Report.SkippedMissing);
        Assert.Equal(0, stack.Report.SkippedShape);
        Assert.NotEqual(qkvBefore, weights["blocks.0.attn.qkv_proj.weight"].AsSpan<float>().ToArray());
        Assert.NotEqual(outBefore, weights["blocks.0.attn.out_proj.weight"].AsSpan<float>().ToArray());
        Assert.NotEqual(fc1Before, weights["blocks.0.mlp.fc1.weight"].AsSpan<float>().ToArray());
    }

    [Fact]
    public void DeltaSizedForAnotherBuildIsReportedAsShapeSkipBesideAppliedOne()
    {
        // The out_proj delta has the wrong rows (another output-head layout); the qkv delta fits and still applies.
        string lora = WriteLora("mixed-build.safetensors",
            ("blocks.0.attn.qkv_proj", Hidden * 3),
            ("blocks.0.attn.out_proj", Hidden / 2));
        using IBackend backend = new CpuBackend();
        using Tensor qkv = Base(Hidden * 3, Hidden);
        using Tensor outProj = Base(Hidden, Hidden);
        Dictionary<string, Tensor> weights = new(StringComparer.Ordinal)
        {
            ["blocks.0.attn.qkv_proj.weight"] = qkv,
            ["blocks.0.attn.out_proj.weight"] = outProj,
        };

        using LoraStack? stack = RecipeLoraMerge.Apply([Spec(lora)], backend,
            new LoraMergeTargets { Transformer = weights }, "MiniMaxH3ImageLoraTests");

        Assert.NotNull(stack);
        Assert.Equal(1, stack.Report.Applied);
        Assert.Equal(1, stack.Report.SkippedShape);
    }

    [Fact]
    public void AiToolkitFileWithTheWrapperIsDetectedAsMiniMaxH3NotWan()
    {
        // The Skeletor shape: every block module under ai-toolkit's diffusion_model. wrapper, including the token-refiner stream.
        string lora = WriteLora("skeletor-shape.safetensors",
            ("blocks.0.attn.qkv_proj", Hidden * 3),
            ("blocks.0.attn.out_proj", Hidden),
            ("blocks.0.mlp.fc1", Hidden * 2),
            ("blocks.0.mlp.fc2", Hidden),
            ("token_refiner.blocks.0.attn.qkv_proj", Hidden * 3),
            ("token_refiner.blocks.0.mlp.fc2", Hidden));

        using LoraFile file = LoraFile.Load(lora);

        Assert.Equal(LoraFormat.AiToolkitMiniMaxH3, file.Format);
        Assert.Equal(6, file.Layers.Count);
        Assert.Contains(file.Layers, layer => layer.TargetKey == "blocks.0.attn.qkv_proj.weight");
        Assert.Contains(file.Layers, layer => layer.TargetKey == "token_refiner.blocks.0.mlp.fc2.weight");
    }

    private static Tensor Base(int rows, int cols)
    {
        Tensor tensor = new(new TensorShape(rows, cols), DType.F32);
        tensor.AsSpan<float>().Fill(0.5f);
        return tensor;
    }

    private static LoraResolver.LoraSpec Spec(string path) => new()
    {
        FilePath = path,
        ModelStrength = 1f,
        TencStrength = 1f,
    };

    private string WriteLora(string name, params (string Module, int OutRows)[] modules)
    {
        string path = Path.Combine(_folder, name);
        List<Tensor> tensors = [];
        List<SafeTensorsStreamWriter.Entry> entries = [];
        foreach ((string module, int outRows) in modules)
        {
            Tensor down = Factor(new TensorShape(Rank, Hidden));
            Tensor up = Factor(new TensorShape(outRows, Rank));
            tensors.Add(down);
            tensors.Add(up);
            entries.Add(new SafeTensorsStreamWriter.Entry($"diffusion_model.{module}.lora_A.weight", DType.F32, down.Shape, () => (down, false)));
            entries.Add(new SafeTensorsStreamWriter.Entry($"diffusion_model.{module}.lora_B.weight", DType.F32, up.Shape, () => (up, false)));
        }
        try
        {
            SafeTensorsStreamWriter.Write(path, entries);
        }
        finally
        {
            foreach (Tensor tensor in tensors)
            {
                tensor.Dispose();
            }
        }
        return path;
    }

    private static Tensor Factor(TensorShape shape)
    {
        Tensor tensor = new(shape, DType.F32);
        tensor.AsSpan<float>().Fill(0.1f);
        return tensor;
    }
}
