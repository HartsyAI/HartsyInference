using HartsyInference.Audio.Models.Haru;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Checkpoints;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests.Haru;

public sealed class HaruPipelineTests(ITestOutputHelper output) : IDisposable
{
    private static readonly string Dir = Path.Combine(AppContext.BaseDirectory, "Fixtures", "Haru");
    private static readonly HaruTextCodec ByteCodec = new(
        text => [.. System.Text.Encoding.UTF8.GetBytes(text).Select(static b => (int)b)],
        ids => string.Concat(ids.Select(static i => ((char)(33 + i % 90)).ToString())));

    private readonly string _cache = Path.Combine(Path.GetTempPath(), $"haru-test-{Guid.NewGuid():N}");

    public void Dispose()
    {
        if (Directory.Exists(_cache))
        {
            Directory.Delete(_cache, recursive: true);
        }
    }

    private static HaruPipeline Open(CpuBackend backend, HaruPrecision precision, string cache, int capacity)
    {
        HaruConfig config = HaruConfig.FromJson(File.ReadAllText(Path.Combine(Dir, "config.json")));
        HfQuantCheckpoint checkpoint = HfQuantCheckpoint.OpenFiles([Path.Combine(Dir, "haru_tiny.safetensors")]);
        (IReadOnlyDictionary<string, Tensor> weights, IDisposable owner) = HaruWeights.Load(checkpoint, precision, Path.Combine(cache, $"{precision}.gguf"));
        return HaruPipeline.Build(backend, config, ByteCodec, weights, owner, capacity);
    }

    private static int[] Run(HaruPipeline pipeline, int frames)
    {
        using HaruDuplexSession session = pipeline.OpenSession(new HaruDuplexOptions { Temperature = 0f, Seed = 2 });
        List<int> tokens = [session.Start("You are Haru.", []).Token];
        float[] silence = new float[pipeline.FrameSamples];
        for (int i = 0; i < frames; i++)
        {
            tokens.Add(session.Step(silence).Token);
        }
        return [.. tokens];
    }

    [Fact]
    public void OnlyOneConversationIsOpenAtATimeAndTheNextOneStartsClean()
    {
        using CpuBackend backend = new();
        using HaruPipeline pipeline = Open(backend, HaruPrecision.Native, _cache, 128);
        int[] first;
        using (HaruDuplexSession open = pipeline.OpenSession(new HaruDuplexOptions { Temperature = 0f, Seed = 2 }))
        {
            Assert.Throws<InvalidOperationException>(() => pipeline.OpenSession(new HaruDuplexOptions()));
        }
        first = Run(pipeline, 3);
        Assert.Equal(first, Run(pipeline, 3));
        Assert.Equal(24000, pipeline.SampleRate);
        Assert.Equal(3200, pipeline.FrameSamples);
    }

    [Fact]
    public void APromptLongerThanTheCapacityIsRefused()
    {
        using CpuBackend backend = new();
        using HaruPipeline pipeline = Open(backend, HaruPrecision.Native, _cache, capacity: 8);
        using HaruDuplexSession session = pipeline.OpenSession(new HaruDuplexOptions { Temperature = 0f });
        Assert.Throws<ArgumentException>(() => session.Start(new string('x', 20), []));
    }

    [Fact]
    public void AConversationIsDoneWhenItReachesTheCapacity()
    {
        using CpuBackend backend = new();
        using HaruPipeline pipeline = Open(backend, HaruPrecision.Native, _cache, capacity: 36);
        using HaruDuplexSession session = pipeline.OpenSession(new HaruDuplexOptions { Temperature = 0f });
        HaruFrame frame = session.Start("hi", []);
        float[] silence = new float[pipeline.FrameSamples];
        int steps = 0;
        while (!frame.Done && steps < 100)
        {
            frame = session.Step(silence);
            steps++;
        }
        Assert.True(frame.Done);
        Assert.True(steps < 8);
    }

    [Fact]
    public void Q8_0Weights_ConvertTheLanguageModelAndKeepTheRest()
    {
        using CpuBackend backend = new();
        HaruConfig config = HaruConfig.FromJson(File.ReadAllText(Path.Combine(Dir, "config.json")));
        HfQuantCheckpoint checkpoint = HfQuantCheckpoint.OpenFiles([Path.Combine(Dir, "haru_tiny.safetensors")]);
        (IReadOnlyDictionary<string, Tensor> weights, IDisposable owner) = HaruWeights.Load(checkpoint, HaruPrecision.Q8_0, Path.Combine(_cache, "q8.gguf"));
        using (owner)
        {
            Assert.Equal(DType.Q8_0, weights["lm.layers.0.mlp.up_proj.weight"].DType);
            Assert.Equal(DType.Q8_0, weights["lm.layers.0.linear_attn.in_proj_qkv.weight"].DType);
            Assert.Equal(DType.Q8_0, weights["lm.layers.3.self_attn.q_proj.weight"].DType);
            Assert.Equal(DType.F32, weights["lm.embed_tokens.weight"].DType);
            Assert.Equal(DType.F32, weights["head.cond_proj.weight"].DType);
            Assert.Equal(DType.F32, weights["codec.encoder.stem.weight"].DType);
            Assert.Equal(DType.F32, weights["lm.layers.0.linear_attn.in_proj_a.weight"].DType);
        }
        output.WriteLine($"{config.Lm.NumLayers} layers converted");
    }

    [Theory]
    [InlineData(null, true, HaruPrecision.Native)]
    [InlineData("", true, HaruPrecision.Native)]
    [InlineData("bf16", true, HaruPrecision.Native)]
    [InlineData("Q8", true, HaruPrecision.Q8_0)]
    [InlineData("q4_k_m", true, HaruPrecision.Q4_K)]
    [InlineData("q3", false, HaruPrecision.Native)]
    public void PrecisionNames_Parse(string? name, bool ok, HaruPrecision expected)
    {
        Assert.Equal(ok, HaruWeights.TryParsePrecision(name, out HaruPrecision precision));
        Assert.Equal(expected, precision);
    }
}
