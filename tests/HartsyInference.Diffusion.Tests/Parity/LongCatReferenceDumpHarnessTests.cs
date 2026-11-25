using System.Text.Json;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using Xunit;

namespace HartsyInference.Diffusion.Tests.Parity;

/// <summary>The reference-dump reader and comparer, exercised on a dump the C# forward writes itself in the dumper's file format.
/// It proves the plumbing
/// (file layout, shapes, the cross-attention row slice, scenario fields), not the model: the Python dump is what proves that.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class LongCatReferenceDumpHarnessTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "longcat-dump-" + Guid.NewGuid().ToString("N"));
    private readonly LongCatConfig _config = LongCatTestData.Tiny;

    public LongCatReferenceDumpHarnessTests() => Directory.CreateDirectory(Path.Combine(_dir, "ckpt"));

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    [Fact]
    public void ADumpOfTheSameForwardMatchesAndATamperedStageIsNamed()
    {
        WriteDump();

        (List<string> failures, List<string> passed) = LongCatReferenceDump.Compare(_dir);

        Assert.Empty(failures);
        Assert.Contains(passed, static line => line.StartsWith("cumulative avatar_ref_skip/block1.cross_attn:", StringComparison.Ordinal));
        Assert.Contains(passed, static line => line.StartsWith("isolated avatar_ref_skip/block1.cross_attn:", StringComparison.Ordinal));
        Assert.Contains(passed, static line => line.StartsWith("isolated avatar_ref_skip/block0.q:", StringComparison.Ordinal));
        Assert.Contains(passed, static line => line.StartsWith("cumulative t2v/t_freq:", StringComparison.Ordinal));
        Assert.DoesNotContain(passed, static line => line.Contains(" t2v/block0.q:", StringComparison.Ordinal));

        string ffn = Path.Combine(_dir, "t2v.block1.ffn.f32");
        byte[] bytes = File.ReadAllBytes(ffn);
        bytes[43] ^= 0x40;   // the top byte of the eleventh value: sign and exponent
        File.WriteAllBytes(ffn, bytes);
        (List<string> tampered, _) = LongCatReferenceDump.Compare(_dir);

        Assert.Equal(2, tampered.Count);   // the cumulative and the isolated comparison of that stage
        Assert.All(tampered, static line => Assert.Contains(" t2v/block1.ffn:", line));
    }

    private readonly record struct DumpCase(string Name, int Frames, int H, int W, int Cond, int Ref, int RefIndex, int Range);

    private void WriteDump()
    {
        using CpuBackend backend = new();
        Dictionary<string, Tensor> weights = LongCatTestData.Weights(_config, 13);
        LongCatTestData.WriteCheckpoint(Path.Combine(_dir, "ckpt"), weights);
        Dictionary<string, long[]> shapes = [];
        Dictionary<string, object> scenarios = [];
        DumpCase[] cases =
        [
            new("t2v", 5, 6, 4, 0, 0, 0, 0), new("avatar_ref_skip", 9, 4, 4, 3, 1, 4, 1),
        ];
        foreach (DumpCase c in cases)
        {
            LongCatConditioning conditioning = new(c.Cond, c.Ref, c.RefIndex, c.Range);
            float[] timesteps = conditioning.FrameTimesteps(640f, c.Frames);
            using Tensor latent = LongCatTestData.Latent(_config, c.Frames, c.H, c.W, 2);
            using Tensor encoder = LongCatTestData.Encoder(_config, 7, 3);
            Dictionary<string, float[]> stages = [];
            Dictionary<string, TensorShape> stageShapes = [];
            using LongCatTransformer transformer = new(_config)
            {
                Probe = (stage, t) =>
                {
                    stages[stage] = t.AsReadOnlySpan<float>().ToArray();
                    stageShapes[stage] = t.Shape;
                },
            };
            transformer.LoadWeights(weights);
            using Tensor velocity = transformer.Forward(backend, latent, encoder, timesteps, in conditioning, 5);

            int condRows = c.Cond * (c.H / 2) * (c.W / 2);
            foreach ((string stage, float[] values) in stages.ToArray())
            {
                if (!stage.EndsWith(".cross_attn", StringComparison.Ordinal))
                    continue;
                float[] full = new float[condRows * _config.HiddenSize + values.Length];
                values.CopyTo(full, condRows * _config.HiddenSize);
                stages[stage + "_full"] = full;
                stageShapes[stage + "_full"] = new TensorShape(full.Length / _config.HiddenSize, _config.HiddenSize);
            }
            stages["latent"] = latent.AsReadOnlySpan<float>().ToArray();
            stageShapes["latent"] = latent.Shape;
            stages["encoder"] = encoder.AsReadOnlySpan<float>().ToArray();
            stageShapes["encoder"] = encoder.Shape;
            foreach ((string stage, float[] values) in stages)
            {
                byte[] bytes = new byte[values.Length * 4];
                Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
                File.WriteAllBytes(Path.Combine(_dir, $"{c.Name}.{stage}.f32"), bytes);
                shapes[$"{c.Name}.{stage}"] = Enumerable.Range(0, stageShapes[stage].Rank).Select(i => stageShapes[stage][i]).ToArray();
            }
            scenarios[c.Name] = new Dictionary<string, object>
            {
                ["frames"] = c.Frames, ["h"] = c.H, ["w"] = c.W, ["num_cond"] = c.Cond, ["num_ref"] = c.Ref, ["ref_index"] = conditioning.RefImgIndex,
                ["mask_range"] = c.Range, ["timesteps"] = timesteps, ["valid_text"] = 5, ["text_len"] = 7,
            };
        }
        File.WriteAllText(Path.Combine(_dir, "meta.json"),
            JsonSerializer.Serialize(new Dictionary<string, object> { ["tensors"] = shapes, ["scenarios"] = scenarios }));
        LongCatTestData.DisposeAll(weights);
    }
}
