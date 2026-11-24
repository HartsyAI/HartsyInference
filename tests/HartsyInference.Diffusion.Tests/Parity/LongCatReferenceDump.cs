using System.Text.Json;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;

namespace HartsyInference.Diffusion.Tests.Parity;

/// <summary>Reads a tests/python-reference/longcat_dit dump directory (raw F32 files plus meta.json and a <c>ckpt</c> folder) and compares every stage of the
/// C# forward against it.</summary>
internal static class LongCatReferenceDump
{
    public const double StageTolerance = 1e-5, RopeTolerance = 1e-6;

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
            int height = s.GetProperty("h").GetInt32(), width = s.GetProperty("w").GetInt32();
            int numCond = s.GetProperty("num_cond").GetInt32();
            LongCatConditioning conditioning = new(numCond, s.GetProperty("num_ref").GetInt32(), s.GetProperty("ref_index").GetInt32(),
                s.GetProperty("mask_range").GetInt32());
            float[] timesteps = s.GetProperty("timesteps").EnumerateArray().Select(static e => e.GetSingle()).ToArray();
            int hw = height / 2 * (width / 2);

            using Tensor latent = Load(dir, meta, $"{name}.latent");
            using Tensor encoder = Load(dir, meta, $"{name}.encoder");
            Dictionary<string, float[]> got = [];
            using LongCatTransformer transformer = new(checkpoint.Config) { Probe = (stage, t) => got[stage] = t.AsReadOnlySpan<float>().ToArray() };
            transformer.LoadWeights(checkpoint.Weights);
            using Tensor velocity = transformer.Forward(backend, latent, encoder, timesteps, in conditioning, s.GetProperty("valid_text").GetInt32());

            foreach (string stage in Stages(checkpoint.Config.Depth))
            {
                if (!got.TryGetValue(stage, out float[]? actual))
                {
                    if (stage is "block0.q" or "block0.k" or "block0.v" && numCond == 0)
                        continue;   // the dense path attends through the layer's own forward and exposes no q/k/v
                    failures.Add($"{name}/{stage}: the C# forward did not report this stage");
                    continue;
                }
                bool cross = stage.EndsWith(".cross_attn", StringComparison.Ordinal);
                float[] expected = Read(dir, meta, $"{name}.{(cross ? stage + "_full" : stage)}");
                if (cross)
                    expected = expected[(numCond * hw * checkpoint.Config.HiddenSize)..];
                double error = actual.Length == expected.Length ? LongCatTestData.RelRms(actual, expected) : double.PositiveInfinity;
                double limit = stage.StartsWith("rope_", StringComparison.Ordinal) ? RopeTolerance : StageTolerance;
                (error <= limit ? passed : failures).Add($"{name}/{stage}: relRMS {error:E2} (limit {limit:E0}, {actual.Length} vs {expected.Length} values)");
            }
        }
        return (failures, passed);
    }

    public static IEnumerable<string> Stages(int depth)
    {
        foreach (string stage in new[] { "x_embedder", "t_embedder", "y_embedder", "rope_cos", "rope_sin" })
            yield return stage;
        for (int i = 0; i < depth; i++)
            foreach (string stage in new[] { "modulation", "attn_in", "self_attn", "after_self", "cross_attn", "after_cross", "ffn", "out" })
                yield return $"block{i}.{stage}";
        foreach (string stage in new[] { "block0.q", "block0.k", "block0.v", "final_layer", "velocity" })
            yield return stage;
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
}
