using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Pins folder-layout discovery to the container rather than the extension. A component published as
/// <c>.gguf</c> used to be invisible to a <c>*.safetensors</c> glob, so the loader threw before
/// <see cref="CheckpointSource"/> ever got to sniff it and the advertised GGUF path could not be reached without
/// renaming every component to a lie.</summary>
public sealed class FolderContainerDiscoveryTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("hartsy-container-discovery").FullName;

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

    private static void WriteSafeTensors(string path, string key)
    {
        using Tensor weight = new Tensor(new TensorShape(4, 8), DType.F32);
        SafeTensorsWriter.Save(path, new Dictionary<string, Tensor> { [key] = weight });
    }

    private static void WriteGguf(string path, string key)
    {
        using Tensor weight = new Tensor(new TensorShape(4, 8), DType.F32);
        using GgufWriter writer = new GgufWriter(path);
        writer.SetMetadata("general.architecture", "containerdiscoverytest");
        writer.AddTensor(key, weight);
        writer.Flush();
    }

    private string Component(string name)
    {
        string dir = Path.Combine(_dir, name);
        Directory.CreateDirectory(dir);
        return dir;
    }

    [Fact]
    public void DiscoverContainerFiles_FindsAGgufAndIgnoresTheConfigsBesideIt()
    {
        string dir = Component("dit_model");
        WriteGguf(Path.Combine(dir, "model-Q4_K_M.gguf"), "blocks.0.attn.to_q.weight");
        File.WriteAllText(Path.Combine(dir, "config.json"), """{ "num_layers": 2 }""");
        File.WriteAllText(Path.Combine(dir, "model.safetensors.index.json"), """{ "weight_map": {} }""");

        Assert.Equal([Path.Combine(dir, "model-Q4_K_M.gguf")], CheckpointConvertUtils.DiscoverContainerFiles(dir));
    }

    [Fact]
    public void DiscoverContainerFiles_RefusesAShardThatOnlyLooksLikeACheckpoint()
    {
        // Dropping the unreadable one instead would load the rest of the set as if it were whole.
        string dir = Component("half-pulled");
        WriteSafeTensors(Path.Combine(dir, "model-00001-of-00002.safetensors"), "blocks.0.attn.to_q.weight");
        File.WriteAllText(Path.Combine(dir, "model-00002-of-00002.safetensors"),
            "version https://git-lfs.github.com/spec/v1\noid sha256:0\nsize 4096\n");

        UnsupportedModelException error = Assert.Throws<UnsupportedModelException>(
            () => CheckpointConvertUtils.DiscoverContainerFiles(dir));
        Assert.Contains("model-00002-of-00002.safetensors", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void OpenShards_RefusesASetThatMixesContainers()
    {
        string dir = Component("mixed");
        WriteSafeTensors(Path.Combine(dir, "model.safetensors"), "blocks.0.attn.to_q.weight");
        WriteGguf(Path.Combine(dir, "model.gguf"), "blocks.0.attn.to_q.weight");

        string[] discovered = CheckpointConvertUtils.DiscoverContainerFiles(dir);
        Assert.Equal(2, discovered.Length);
        UnsupportedModelException error = Assert.Throws<UnsupportedModelException>(() => CheckpointSource.OpenShards(discovered));
        Assert.Contains("mixes containers", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void FLiteLoadFolder_LoadsAComponentPublishedAsGguf()
    {
        string root = Path.Combine(_dir, "F-Lite");
        Directory.CreateDirectory(Path.Combine(root, "dit_model"));
        Directory.CreateDirectory(Path.Combine(root, "text_encoder"));
        Directory.CreateDirectory(Path.Combine(root, "vae"));
        WriteGguf(Path.Combine(root, "dit_model", "diffusion_pytorch_model-Q8_0.gguf"), "blocks.0.attn.to_q.weight");
        WriteSafeTensors(Path.Combine(root, "text_encoder", "model.safetensors"), "encoder.block.0.layer.0.SelfAttention.q.weight");
        WriteSafeTensors(Path.Combine(root, "vae", "diffusion_pytorch_model.safetensors"), "encoder.conv_in.weight");

        (FLiteCheckpointConverter.ConvertedWeights converted, IDisposable sources) = FLiteCheckpointConverter.LoadFolder(root);
        using (sources)
        {
            Assert.Contains("blocks.0.attn.to_q.weight", converted.Transformer);
            Assert.Single(converted.T5);
            Assert.Single(converted.Vae);
        }
    }

    [Fact]
    public void LanceLoadVariant_LoadsAVariantPublishedAsGguf()
    {
        string variant = Component("Lance_3B");
        WriteGguf(Path.Combine(variant, "model-Q4_K_M.gguf"), "language_model.model.layers.0.self_attn.q_proj.weight");

        (LanceCheckpointConverter.ConvertedWeights converted, CheckpointSource source) = LanceCheckpointConverter.LoadVariant(variant);
        using (source)
        {
            Assert.Contains("layers.0.self_attn.q_proj.weight", converted.Transformer);
        }
    }
}
