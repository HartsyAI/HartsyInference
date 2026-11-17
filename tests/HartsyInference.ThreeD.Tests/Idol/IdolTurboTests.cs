using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Lora;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ThreeD.Models.Idol;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Idol;

public sealed class IdolTurboTests : IDisposable
{
    private const int Rank = 2;

    private static readonly IBackend Backend = new CpuBackend();
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "idol-turbo-tests-" + Guid.NewGuid().ToString("N"));

    public IdolTurboTests() => Directory.CreateDirectory(_directory);

    public void Dispose()
    {
        try { Directory.Delete(_directory, recursive: true); }
        catch (IOException) { /* the temp folder is the OS's to clean */ }
    }

    [Theory]
    [InlineData("")]
    [InlineData("diffusion_model.")]
    public void AComfyIdolLoraFileIsRecognisedAndMapsOntoTheDenoisersWeights(string wrapper)
    {
        string path = WriteLora("a.safetensors", IdolModelTests.TinyConfig(), seed: 1, wrapper);

        using LoraFile file = LoraFile.Load(path);

        Assert.Equal(LoraFormat.Idol, file.Format);
        IdolConfig c = IdolModelTests.TinyConfig();
        Assert.Equal((2 * c.RefinerBlocks + c.Blocks) * 4, file.Layers.Count);
        Assert.All(file.Layers, layer => Assert.Equal(LoraTarget.Transformer, layer.Target));
        Assert.Contains(file.Layers, l => l.TargetKey == "blocks.0.attn.qkv.weight");
        Assert.Contains(file.Layers, l => l.TargetKey == "noise_refiner.0.mlp.mlp.2.weight");
        Assert.Contains(file.Layers, l => l.TargetKey == "context_refiner.0.attn.out.weight");
    }

    [Fact]
    public void AWanStyleFileIsNotTakenForIdol()
    {
        Dictionary<string, Tensor> tensors = new()
        {
            ["diffusion_model.blocks.0.self_attn.q.lora_A.weight"] = IdolModelTests.Random(new TensorShape(Rank, 16), 1, 0.1f),
            ["diffusion_model.blocks.0.self_attn.q.lora_B.weight"] = IdolModelTests.Random(new TensorShape(16, Rank), 2, 0.1f),
        };
        string path = Path.Combine(_directory, "wan.safetensors");
        SafeTensorsWriter.Save(path, tensors);
        foreach (Tensor t in tensors.Values) t.Dispose();

        using LoraFile file = LoraFile.Load(path);

        Assert.NotEqual(LoraFormat.Idol, file.Format);
    }

    [Fact]
    public void ThePhaseExpertsAreGatedAndEqualTheMergedWeightsOfTheirOwnPhase()
    {
        IdolConfig tiny = IdolModelTests.TinyConfig();
        Dictionary<string, Tensor> baseTensors = IdolModelTests.TinyTensors(tiny, seed: 3);
        string slowPath = WriteLora("slow.safetensors", tiny, seed: 10, "");
        string fastPath = WriteLora("fast.safetensors", tiny, seed: 20, "");
        Dictionary<string, Tensor> slowMerged = Merge(baseTensors, slowPath);
        Dictionary<string, Tensor> fastMerged = Merge(baseTensors, fastPath);
        try
        {
            using IdolWeights weights = IdolWeights.FromTensors(tiny, baseTensors);
            using IdolTurboAdapters turbo = IdolTurboAdapters.Attach(Backend, weights, slowPath, fastPath);
            using IdolDenoiser carried = new(turbo.Weights);
            using IdolDenoiser plain = new(weights);
            using IdolWeights slowWeights = IdolWeights.FromTensors(tiny, slowMerged);
            using IdolWeights fastWeights = IdolWeights.FromTensors(tiny, fastMerged);
            using IdolDenoiser slowModel = new(slowWeights);
            using IdolDenoiser fastModel = new(fastWeights);

            float[] unadapted = Velocity(plain, tiny, 800f, scope: null);
            Assert.Equal(unadapted, Velocity(carried, tiny, 800f, scope: null));   // no gate open: the base weights' own result

            float[] slow = Velocity(carried, tiny, 800f, turbo.Phase(800f));
            float[] fast = Velocity(carried, tiny, 200f, turbo.Phase(200f));
            Assert.NotEqual(unadapted, slow);
            AssertClose(Velocity(slowModel, tiny, 800f, scope: null), slow);
            AssertClose(Velocity(fastModel, tiny, 200f, scope: null), fast);

            // The same time through the other expert differs, so the switch is real; a pass at 500 (shifted time 0.5) is the slow one.
            Assert.NotEqual(slow, Velocity(fastModel, tiny, 800f, scope: null));
            AssertClose(Velocity(slowModel, tiny, 500f, scope: null), Velocity(carried, tiny, 500f, turbo.Phase(500f)));
        }
        finally
        {
            foreach (Tensor t in baseTensors.Values) t.Dispose();
            foreach (Tensor t in slowMerged.Values.Concat(fastMerged.Values).Where(t => !baseTensors.Values.Contains(t))) t.Dispose();
        }
    }

    [Fact]
    public void AFileThatDoesNotMatchTheDenoiserIsRefusedByName()
    {
        IdolConfig tiny = IdolModelTests.TinyConfig();
        IdolConfig deeper = new()
        {
            Tokens = tiny.Tokens, LatentChannels = tiny.LatentChannels, CameraChannels = tiny.CameraChannels, ModelChannels = tiny.ModelChannels, CondChannels = tiny.CondChannels,
            Cond2Channels = tiny.Cond2Channels, Blocks = tiny.Blocks + 1, RefinerBlocks = tiny.RefinerBlocks, Heads = tiny.Heads, HeadDim = tiny.HeadDim, MlpRatio = tiny.MlpRatio, CondTokens = tiny.CondTokens,
        };
        string slow = WriteLora("slow.safetensors", deeper, seed: 1, ""), fast = WriteLora("fast.safetensors", deeper, seed: 2, "");
        Dictionary<string, Tensor> baseTensors = IdolModelTests.TinyTensors(tiny, seed: 4);
        try
        {
            using IdolWeights weights = IdolWeights.FromTensors(tiny, baseTensors);
            Assert.Throws<HartsyInference.Core.Exceptions.HartsyInferenceException>(() => IdolTurboAdapters.Attach(Backend, weights, slow, fast));
        }
        finally
        {
            foreach (Tensor t in baseTensors.Values) t.Dispose();
        }
    }

    [Theory]
    [InlineData(1000f, true)]
    [InlineData(500f, true)]
    [InlineData(499f, false)]
    [InlineData(0f, false)]
    public void TheSlowExpertRunsAtAndAboveAShiftedTimeOfOneHalf(float tModel, bool slow)
    {
        IdolConfig tiny = IdolModelTests.TinyConfig();
        Dictionary<string, Tensor> baseTensors = IdolModelTests.TinyTensors(tiny, seed: 5);
        string slowPath = WriteLora("slow.safetensors", tiny, seed: 1, ""), fastPath = WriteLora("fast.safetensors", tiny, seed: 2, "");
        try
        {
            using IdolWeights weights = IdolWeights.FromTensors(tiny, baseTensors);
            using IdolTurboAdapters turbo = IdolTurboAdapters.Attach(Backend, weights, slowPath, fastPath);
            using IdolDenoiser carried = new(turbo.Weights);
            using IdolWeights oneExpert = IdolWeights.FromTensors(tiny, Merge(baseTensors, slow ? slowPath : fastPath));
            using IdolDenoiser expected = new(oneExpert);

            AssertClose(Velocity(expected, tiny, tModel, scope: null), Velocity(carried, tiny, tModel, turbo.Phase(tModel)));
        }
        finally
        {
            foreach (Tensor t in baseTensors.Values) t.Dispose();
        }
    }

    private static float[] Velocity(IdolDenoiser model, IdolConfig tiny, float tModel, IDisposable? scope)
    {
        using (scope)
        {
            using Tensor f1 = IdolModelTests.Random(new TensorShape(1, tiny.CondTokens, tiny.CondChannels), 3, 1f);
            using Tensor f2 = IdolModelTests.Random(new TensorShape(1, tiny.CondTokens, tiny.Cond2Channels), 4, 1f);
            using Tensor z = IdolModelTests.Random(new TensorShape(1, tiny.Tokens, tiny.LatentChannels), 5, 1f);
            using Tensor cam = IdolModelTests.Random(new TensorShape(1, 1, tiny.CameraChannels), 6, 1f);
            using Tensor context = model.EmbedContext(Backend, f1, f2);
            (Tensor v, Tensor c) = model.Forward(Backend, z, cam, tModel, context);
            using (v)
            using (c)
                return [.. v.AsReadOnlySpan<float>(), .. c.AsReadOnlySpan<float>()];
        }
    }

    private static void AssertClose(float[] want, float[] got)
    {
        Assert.Equal(want.Length, got.Length);
        double peak = 0, worst = 0;
        for (int i = 0; i < want.Length; i++)
        {
            peak = Math.Max(peak, Math.Abs(want[i]));
            worst = Math.Max(worst, Math.Abs(want[i] - got[i]));
        }

        Assert.True(worst <= 1e-4 * peak, $"max |difference| {worst:E2} against a peak of {peak:E2}");
    }

    /// <summary>A Comfy-format file with a rank-2 pair for every block module the denoiser of <paramref name="c"/> has.</summary>
    private string WriteLora(string name, IdolConfig c, int seed, string wrapper)
    {
        Dictionary<string, Tensor> tensors = new(StringComparer.Ordinal);
        int s = seed * 100;
        foreach ((string key, long[] shape) in IdolWeights.ExpectedShapes(c))
        {
            bool block = key.StartsWith("blocks.", StringComparison.Ordinal) || key.StartsWith("noise_refiner.", StringComparison.Ordinal) || key.StartsWith("context_refiner.", StringComparison.Ordinal);
            if (!block || !(key.EndsWith(".attn.qkv.weight", StringComparison.Ordinal) || key.EndsWith(".attn.out.weight", StringComparison.Ordinal)
                || key.EndsWith(".mlp.mlp.0.weight", StringComparison.Ordinal) || key.EndsWith(".mlp.mlp.2.weight", StringComparison.Ordinal)))
                continue;
            string module = wrapper + key[..^".weight".Length];
            tensors[module + ".lora_down.weight"] = IdolModelTests.Random(new TensorShape(Rank, shape[1]), s++, 0.3f);
            tensors[module + ".lora_up.weight"] = IdolModelTests.Random(new TensorShape(shape[0], Rank), s++, 0.3f);
        }

        string path = Path.Combine(_directory, name);
        SafeTensorsWriter.Save(path, tensors);
        foreach (Tensor t in tensors.Values) t.Dispose();
        return path;
    }

    /// <summary>The base tensors with the file's pairs merged into their weights (W + up·down), as new tensors for those weights and the same ones for the rest.</summary>
    private static Dictionary<string, Tensor> Merge(Dictionary<string, Tensor> baseTensors, string loraPath)
    {
        using SafeTensorsLoader loader = new();
        loader.Load(loraPath);
        Dictionary<string, Tensor> pairs = loader.GetAllTensors();
        Dictionary<string, Tensor> merged = new(baseTensors, StringComparer.Ordinal);
        foreach (string key in baseTensors.Keys.ToArray())
        {
            string module = key.EndsWith(".weight", StringComparison.Ordinal) ? key[..^".weight".Length] : key;
            if (!pairs.TryGetValue(module + ".lora_down.weight", out Tensor? down))
                continue;
            Tensor up = pairs[module + ".lora_up.weight"];
            Tensor weight = baseTensors[key];
            long outDim = weight.Shape[0], inDim = weight.Shape[1];
            Tensor sum = new(weight.Shape, DType.F32);
            ReadOnlySpan<float> w = weight.AsReadOnlySpan<float>(), d = down.AsReadOnlySpan<float>(), u = up.AsReadOnlySpan<float>();
            Span<float> o = sum.AsSpan<float>();
            for (long r = 0; r < outDim; r++)
            {
                for (long k = 0; k < inDim; k++)
                {
                    float acc = w[(int)(r * inDim + k)];
                    for (int j = 0; j < Rank; j++)
                        acc += u[(int)(r * Rank + j)] * d[(int)(j * inDim + k)];
                    o[(int)(r * inDim + k)] = acc;
                }
            }

            merged[key] = sum;
        }

        return merged;
    }
}
