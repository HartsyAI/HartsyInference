using System.Text.Json;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Hy4Flash;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.Hy4Flash;

/// <summary>The whole text model against the Hugging Face classes of transformers PR 50903 at eb1d5db on the committed tiny-config dump (3 layers, float32, CPU): every stage the dump records.</summary>
public sealed class Hy4FlashModelHfDumpTests(ITestOutputHelper output)
{
    private static string DataDir
    {
        get
        {
            string? dir = AppContext.BaseDirectory;
            while (dir is not null && !Directory.Exists(Path.Combine(dir, "tests", "python-reference", "hy4_flash", "data"))) dir = Path.GetDirectoryName(dir);
            return dir is null ? throw new DirectoryNotFoundException("tests/python-reference/hy4_flash/data not found.") : Path.Combine(dir, "tests", "python-reference", "hy4_flash", "data");
        }
    }

    private sealed class SetSource(ShardedSafeTensorSet set) : IHy4FlashWeightSource
    {
        public Tensor Read(string name) => set.GetTensor(name);

        public bool Contains(string name) => set.Inventory.ContainsKey(name);
    }

    private static int[] Ids()
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Path.Combine(DataDir, "inputs_text.json")));
        return [.. document.RootElement.GetProperty("input_ids").EnumerateArray().Select(static e => e.GetInt32())];
    }

    private static float[] Npy(string name) => NpyReader.ReadFloat32(Path.Combine(DataDir, name + ".npy")).Data;

    private static float RelativeMax(float[] expected, float[] actual, int stride, out int worstRow)
    {
        float scale = 1e-6f;
        foreach (float v in expected) scale = MathF.Max(scale, MathF.Abs(v));
        float worst = 0f;
        worstRow = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            float d = MathF.Abs(expected[i] - actual[i]) / scale;
            if (d > worst)
            {
                worst = d;
                worstRow = i / stride;
            }
        }
        return worst;
    }

    [Fact]
    public void TextForward_MatchesEveryStageOfTheHfDump()
    {
        Hy4FlashConfig config = Hy4FlashConfig.Load(Path.Combine(DataDir, "config.json"));
        using ShardedSafeTensorSet set = ShardedSafeTensorSet.OpenFiles([Path.Combine(DataDir, "weights.safetensors")]);
        using Hy4FlashModel model = new(new CpuBackend(), config, new SetSource(set), 64);
        Dictionary<string, float[]> got = [];
        model.Trace = (name, data) => got[name] = data;
        int[] ids = Ids();
        using Hy4FlashSequenceState state = model.CreateState();
        model.Forward(ids, state, allRows: true);

        List<string> failures = [];
        int h = config.HiddenSize, hc = config.HcMult;
        foreach ((string stage, int stride) in new (string, int)[]
        {
            ("embed", h), ("layer_0_attn_hc_post", hc), ("layer_0_attn_hc_in", h), ("layer_0_attn_out", h), ("layer_0_ffn_hc_in", h), ("layer_0_ffn_out", h), ("layer_0_out", hc * h),
            ("engram_1_out", hc * h), ("layer_1_attn_out", h), ("layer_1_ffn_out", h), ("layer_1_out", hc * h), ("layer_2_attn_out", h), ("layer_2_ffn_out", h), ("layer_2_out", hc * h),
            ("hc_head_out", h), ("final_norm_out", h), ("logits", config.VocabSize),
        })
        {
            float diff = RelativeMax(Npy("text__" + stage), got[stage], stride, out int row);
            output.WriteLine($"{stage,-22} max relative difference {diff:E2} (worst row {row})");
            if (diff > 1e-5f) failures.Add($"{stage}: {diff:E2} at row {row}");
        }
        Assert.True(failures.Count == 0, string.Join("; ", failures));
    }

    [Fact]
    public void PrefillThenCachedDecodeSteps_MatchTheHfCachedRun()
    {
        Hy4FlashConfig config = Hy4FlashConfig.Load(Path.Combine(DataDir, "config.json"));
        using ShardedSafeTensorSet set = ShardedSafeTensorSet.OpenFiles([Path.Combine(DataDir, "weights.safetensors")]);
        using Hy4FlashModel model = new(new CpuBackend(), config, new SetSource(set), 64);
        int[] ids = Ids();
        using Hy4FlashSequenceState state = model.CreateState();
        float diff = RelativeMax(Npy("cached_prefill__logits"), model.Forward(ids.AsSpan(0, 37), state, allRows: true), config.VocabSize, out _);
        output.WriteLine($"prefill 37: {diff:E2}");
        Assert.True(diff < 1e-5f, $"prefill: {diff:E2}");
        for (int step = 0; step < 3; step++)
        {
            float d = RelativeMax(Npy($"cached_step{step}__logits"), model.Forward(ids.AsSpan(37 + step, 1), state), config.VocabSize, out _);
            output.WriteLine($"decode step {step}: {d:E2}");
            Assert.True(d < 1e-5f, $"decode step {step}: {d:E2}");
        }
        Assert.Equal(40, state.Length);
    }

    [Fact]
    public void VisionTower_MatchesTheHfTowerAndMergerOutput()
    {
        Hy4FlashConfig config = Hy4FlashConfig.Load(Path.Combine(DataDir, "config.json"));
        using ShardedSafeTensorSet set = ShardedSafeTensorSet.OpenFiles([Path.Combine(DataDir, "weights.safetensors")]);
        using Hy4FlashVisionTower tower = new(new CpuBackend(), config.Vision, new SetSource(set));
        float[] pixels = Npy("image__pixel_values");
        float[] actual = tower.Forward(pixels, 4, 4);
        float[] expected = Npy("image__vision_out");
        Assert.Equal(tower.TokenCount(4, 4) * config.Vision.OutHiddenSize, actual.Length);
        float diff = RelativeMax(expected, actual, config.Vision.OutHiddenSize, out int row);
        output.WriteLine($"vision tokens: {diff:E2} (worst token {row})");
        Assert.True(diff < 1e-5f, $"vision output: {diff:E2} at token {row}");
    }

    [Fact]
    public void ImageSequence_MatchesEveryStageOfTheHfDump()
    {
        Hy4FlashConfig config = Hy4FlashConfig.Load(Path.Combine(DataDir, "config.json"));
        using ShardedSafeTensorSet set = ShardedSafeTensorSet.OpenFiles([Path.Combine(DataDir, "weights.safetensors")]);
        SetSource source = new(set);
        CpuBackend backend = new();
        float[] features;
        using (Hy4FlashVisionTower tower = new(backend, config.Vision, source)) features = tower.Forward(Npy("image__pixel_values"), 4, 4);
        using Hy4FlashModel model = new(backend, config, source, 64);
        Dictionary<string, float[]> got = [];
        model.Trace = (name, data) => got[name] = data;
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Path.Combine(DataDir, "inputs_image.json")));
        int[] ids = [.. document.RootElement.GetProperty("input_ids").EnumerateArray().Select(static e => e.GetInt32())];
        using Hy4FlashSequenceState state = model.CreateState();
        model.Forward(ids, state, allRows: true, images: [new Hy4FlashImage(4, 4, features)]);

        List<string> failures = [];
        int h = config.HiddenSize, hc = config.HcMult;
        foreach ((string stage, int stride) in new (string, int)[]
        {
            ("embed", h), ("layer_0_attn_out", h), ("layer_0_out", hc * h), ("engram_1_out", hc * h), ("layer_1_attn_out", h), ("layer_1_ffn_out", h), ("layer_1_out", hc * h),
            ("layer_2_attn_out", h), ("layer_2_out", hc * h), ("hc_head_out", h), ("final_norm_out", h), ("logits", config.VocabSize),
        })
        {
            float diff = RelativeMax(Npy("image__" + stage), got[stage], stride, out int row);
            output.WriteLine($"image {stage,-18} max relative difference {diff:E2} (worst row {row})");
            if (diff > 1e-5f) failures.Add($"{stage}: {diff:E2} at row {row}");
        }
        Assert.True(failures.Count == 0, string.Join("; ", failures));
        Assert.Equal(11 + 10, state.NextPosition);
    }
}
