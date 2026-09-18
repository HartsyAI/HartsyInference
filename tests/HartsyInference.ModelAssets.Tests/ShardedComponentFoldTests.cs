using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Pins the two orderings a sharded component load has to get right, both of which fail silently rather than
/// loudly. Safetensors sharding makes no promise that a weight and its <c>.weight_scale</c> land in the same file, so
/// folding per shard splits a pair that belongs together; and a key map that renames <c>.weight</c> has no rule for
/// <c>.weight_scale</c>, so folding after it pairs nothing. Either way the fp8 weight keeps a scale of 1.0 and runs
/// hundreds of times too large — noise at the end of a generation, no error at load.</summary>
public sealed unsafe class ShardedComponentFoldTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("hartsy-shard-fold").FullName;

    public void Dispose()
    {
        try
        {
            Directory.Delete(_dir, recursive: true);
        }
        catch (IOException)
        {
            // A leftover temp directory is not worth failing a passing test over.
        }
    }

    private void WriteShard(string name, IReadOnlyDictionary<string, Tensor> tensors)
        => SafeTensorsWriter.Save(Path.Combine(_dir, name), tensors);

    [Fact]
    public void LoadShards_FoldsAScaleThatLivesInAnotherShard_BeforeTheKeyMapRenamesTheWeight()
    {
        using Tensor weight = new Tensor(new TensorShape(4, 8), DType.F8E4M3);
        using Tensor scale = new Tensor(new TensorShape(1), DType.F32);
        scale.AsSpan<float>()[0] = 0.0125f;
        WriteShard("model-00001-of-00002.safetensors", new Dictionary<string, Tensor> { ["blocks.0.attn.wq.weight"] = weight });
        WriteShard("model-00002-of-00002.safetensors", new Dictionary<string, Tensor> { ["blocks.0.attn.wq.weight_scale"] = scale });

        string[] shards =
        [
            Path.Combine(_dir, "model-00001-of-00002.safetensors"),
            Path.Combine(_dir, "model-00002-of-00002.safetensors"),
        ];
        // Krea 2's real map, which renames only `.weight`. It used to carry a companion-suffix pre-rename purely so
        // a later fold could still pair the two; that pre-rename is gone, and this is what replaced it.
        (Dictionary<string, Tensor> weights, Checkpoints.CheckpointSource source) = CheckpointConvertUtils.LoadShards(
            shards, 8, key => Krea2CheckpointConverter.RemapTransformerKey(CheckpointConvertUtils.StripTransformerPrefix(key)));
        using (source)
        {
            Tensor mapped = Assert.Contains("transformer_blocks.0.attn.to_q.weight", weights);
            Assert.Equal(0.0125f, mapped.Fp8ScaleFactor);
            // The companion must not survive under either its raw or its mapped name.
            Assert.DoesNotContain(weights, entry => entry.Key.EndsWith(".weight_scale", StringComparison.Ordinal));
        }
    }

    [Fact]
    public void LoadShards_DropsAKeyTheMapRejects()
    {
        using Tensor keep = new Tensor(new TensorShape(2, 4), DType.F32);
        using Tensor drop = new Tensor(new TensorShape(2, 4), DType.F32);
        WriteShard("single.safetensors", new Dictionary<string, Tensor> { ["keep.weight"] = keep, ["visual.drop.weight"] = drop });

        (Dictionary<string, Tensor> weights, Checkpoints.CheckpointSource source) = CheckpointConvertUtils.LoadShards(
            [Path.Combine(_dir, "single.safetensors")], 4,
            key => key.StartsWith("visual.", StringComparison.Ordinal) ? null : key);
        using (source)
        {
            Assert.Contains("keep.weight", weights);
            Assert.DoesNotContain("visual.drop.weight", weights);
        }
    }
}
