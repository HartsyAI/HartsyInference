using System.Text.Json;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Tests.Parity;

/// <summary>Reads a tests/python-reference/longcat_dit dump directory (raw F32 files plus meta.json and a <c>ckpt</c> folder) and
/// compares the C# forward
/// against it twice: cumulatively (the whole forward, every stage after all earlier C# stages) and isolated (each block fed the reference's own
/// inputs), so one early difference cannot hide or fake the stages after it.</summary>
internal static class LongCatReferenceDump
{
    public const double StageTolerance = 1e-5, RopeTolerance = 1e-6;
    private static readonly string[] BlockStages =
        ["modulation", "attn_in", "self_attn", "after_self", "cross_attn", "after_cross", "ffn", "out"];

    public static (List<string> Failures, List<string> Passed) Compare(string dir)
    {
        using JsonDocument meta = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "meta.json")));
        using CpuBackend backend = new();
        using LongCatCheckpoint checkpoint = LongCatCheckpoint.Open(Path.Combine(dir, "ckpt"));
        List<string> failures = [], passed = [];

        foreach (JsonProperty scenario in meta.RootElement.GetProperty("scenarios").EnumerateObject())
        {
            JsonElement s = scenario.Value;
            string name = scenario.Name;
            int frames = s.GetProperty("frames").GetInt32();
            int hw = s.GetProperty("h").GetInt32() / 2 * (s.GetProperty("w").GetInt32() / 2);
            LongCatConditioning conditioning = new(s.GetProperty("num_cond").GetInt32(), s.GetProperty("num_ref").GetInt32(),
                s.GetProperty("ref_index").GetInt32(), s.GetProperty("mask_range").GetInt32());
            float[] timesteps = s.GetProperty("timesteps").EnumerateArray().Select(static e => e.GetSingle()).ToArray();
            Stage check = new(dir, meta, name, conditioning.NumCondLatents * hw * checkpoint.Config.HiddenSize, failures, passed);

            using Tensor latent = Load(dir, meta, $"{name}.latent");
            using Tensor encoder = Load(dir, meta, $"{name}.encoder");
            Dictionary<string, float[]> got = [];
            using LongCatTransformer transformer = new(checkpoint.Config)
            {
                Probe = (stage, t) => got[stage] = t.AsReadOnlySpan<float>().ToArray(),
            };
            transformer.LoadWeights(checkpoint.Weights);
            using Tensor velocity = transformer.Forward(backend, latent, encoder, timesteps, in conditioning, s.GetProperty("valid_text").GetInt32());
            foreach (string stage in Stages(checkpoint.Config.Depth))
                check.Compare("cumulative", stage, got.GetValueOrDefault(stage), conditioning.NumCondLatents == 0);

            for (int i = 0; i < checkpoint.Config.Depth; i++)
                IsolatedBlock(backend, checkpoint, dir, meta, name, i, frames, hw, conditioning, check);
        }
        return (failures, passed);
    }

    public static IEnumerable<string> Stages(int depth)
    {
        foreach (string stage in new[] { "t_freq", "x_embedder", "t_embedder", "y_embedder", "rope_cos", "rope_sin" })
            yield return stage;
        for (int i = 0; i < depth; i++)
            foreach (string stage in BlockStages)
                yield return $"block{i}.{stage}";
        foreach (string stage in new[] { "block0.q", "block0.k", "block0.v", "final_layer", "velocity" })
            yield return stage;
    }

    /// <summary>One block run on the reference's saved inputs: the previous block's output (or the patch embedding), the caption, the timestep
    /// embedding and the RoPE tables.</summary>
    private static void IsolatedBlock(CpuBackend backend, LongCatCheckpoint checkpoint, string dir, JsonDocument meta, string name, int index,
        int frames, int hw, LongCatConditioning conditioning, Stage check)
    {
        LongCatBlock block = new(checkpoint.Config);
        List<Tensor> owned = [];
        block.LoadWeights(checkpoint.Weights, $"blocks.{index}", owned);
        using Tensor hidden = Load(dir, meta, index == 0 ? $"{name}.x_embedder" : $"{name}.block{index - 1}.out");
        using Tensor text = Load(dir, meta, $"{name}.y_embedder");
        using Tensor timeEmbed = Load(dir, meta, $"{name}.t_embedder");
        using Tensor cos = Load(dir, meta, $"{name}.rope_cos");
        using Tensor sin = Load(dir, meta, $"{name}.rope_sin");
        using Tensor rows = LongCatOps.FrameRowIndex(frames, hw);
        LongCatBlockInputs inputs = new(frames, hw, conditioning.NumCondLatents, cos, sin, conditioning.AttentionPlan(hw, frames), rows);
        Dictionary<string, float[]> got = [];

        using Tensor output = block.Forward(backend, hidden, text, timeEmbed, in inputs,
            (stage, t) => got[stage] = t.AsReadOnlySpan<float>().ToArray(), $"block{index}");

        foreach (string stage in BlockStages)
            check.Compare("isolated", $"block{index}.{stage}", got.GetValueOrDefault($"block{index}.{stage}"), false);
        if (index == 0)
            foreach (string stage in new[] { "block0.q", "block0.k", "block0.v" })
                check.Compare("isolated", stage, got.GetValueOrDefault(stage), conditioning.NumCondLatents == 0);
        foreach (Tensor copy in owned)
            copy.Dispose();
    }

    public static Tensor Load(string dir, JsonDocument meta, string name)
    {
        long[] shape = meta.RootElement.GetProperty("tensors").GetProperty(name).EnumerateArray().Select(static e => e.GetInt64()).ToArray();
        Tensor t = new(new TensorShape(shape), DType.F32);
        Read(dir, meta, name).CopyTo(t.AsSpan<float>());
        return t;
    }

    public static float[] Read(string dir, JsonDocument meta, string name)
    {
        if (!meta.RootElement.GetProperty("tensors").TryGetProperty(name, out _))
            throw new InvalidDataException($"The dump has no tensor '{name}'.");
        byte[] bytes = File.ReadAllBytes(Path.Combine(dir, name + ".f32"));
        float[] values = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        return values;
    }

    /// <summary>One scenario's comparisons: the reference file of a stage against what the C# forward reported for it.</summary>
    private sealed class Stage(string dir, JsonDocument meta, string scenario, int crossCondOffset, List<string> failures, List<string> passed)
    {
        public void Compare(string mode, string stage, float[]? actual, bool optional)
        {
            if (actual is null)
            {
                // The dense path attends through the layer's own forward and exposes no q/k/v.
                if (!optional || stage is not ("block0.q" or "block0.k" or "block0.v"))
                    failures.Add($"{mode} {scenario}/{stage}: the C# forward did not report this stage");
                return;
            }
            bool cross = stage.EndsWith(".cross_attn", StringComparison.Ordinal);
            float[] expected = Read(dir, meta, $"{scenario}.{(cross ? stage + "_full" : stage)}");
            if (cross)
                expected = expected[crossCondOffset..];   // the reference's update is zero on the conditioning rows
            double error = actual.Length == expected.Length ? LongCatTestData.RelRms(actual, expected) : double.PositiveInfinity;
            double limit = stage.StartsWith("rope_", StringComparison.Ordinal) ? RopeTolerance : StageTolerance;
            string line = $"{mode} {scenario}/{stage}: relRMS {error:E2} (limit {limit:E0}, {actual.Length} vs {expected.Length} values)";
            (error <= limit ? passed : failures).Add(line);
        }
    }
}
