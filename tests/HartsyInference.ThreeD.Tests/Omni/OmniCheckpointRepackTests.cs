using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ThreeD.Models.Hunyuan3D.Omni;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Omni;

/// <summary>An Omni folder whose pickles were repacked to safetensors beside them: the same tensors under the same names, read without a pickle.</summary>
public sealed class OmniCheckpointRepackTests : IDisposable
{
    private const string PipelineJson = """{"target":"hy3dshape.pipelines.Hunyuan3DOmniSiTFlowMatchingPipeline","params":{"box_v":1.01,"octree_resolution":384,"num_chunks":50000,"mc_level":0.0,"output_type":"mesh","num_inference_steps":50,"scaling_factor":1.0079887561230991}}""";
    private const string ModelJson = """{"target":"hy3dshape.models.denoisers.hunyuandit.HunYuanDiTPlain","params":{"input_size":4096,"in_channels":64,"hidden_size":2048,"context_dim":1024,"depth":21,"num_heads":16,"qk_norm":true,"text_len":1370,"with_decoupled_ca":false,"use_attention_pooling":false,"qk_norm_type":"rms","qkv_bias":false,"use_pos_emb":false,"num_moe_layers":6,"num_experts":8,"moe_top_k":2}}""";
    private const string VaeJson = """{"target":"hy3dshape.models.autoencoders.shape_vae_pl.ShapeVAEPL","params":{"module_cfg":{"target":"hy3dshape.models.autoencoders.shape_vae.ShapeVAE","params":{"num_latents":4096,"embed_dim":64,"point_feats":3,"num_freqs":8,"include_pi":false,"heads":16,"width":1024,"num_encoder_layers":8,"num_decoder_layers":16,"use_ln_post":true,"qkv_bias":false,"qk_norm":true}},"loss_cfg":{"target":"torch.nn.Identity"}},"scale_factor":1.0079887561230991}""";

    private readonly string _folder = Directory.CreateTempSubdirectory("omni-repack-").FullName;

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    private static Tensor Filled(TensorShape shape, float first)
    {
        Tensor tensor = new(shape, DType.F32);
        Span<float> data = tensor.AsSpan<float>();
        for (int i = 0; i < data.Length; i++)
            data[i] = first + i;
        return tensor;
    }

    private void Write(string relative, params (string Name, Tensor Tensor)[] tensors)
    {
        string path = Path.Combine(_folder, relative);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        SafeTensorsWriter.Save(path, tensors.ToDictionary(t => t.Name, t => t.Tensor));
        foreach ((_, Tensor tensor) in tensors)
            tensor.Dispose();
    }

    private void PlaceConfigs()
    {
        File.WriteAllText(Path.Combine(_folder, "config.json"), PipelineJson);
        Directory.CreateDirectory(Path.Combine(_folder, "model"));
        File.WriteAllText(Path.Combine(_folder, "model", "config.json"), ModelJson);
        Directory.CreateDirectory(Path.Combine(_folder, "vae"));
        File.WriteAllText(Path.Combine(_folder, "vae", "config.json"), VaeJson);
    }

    [Fact]
    public void Load_ReadsTheRepacksThatSitBesideTheMissingPicklesAndNarrowsMatricesWhenAsked()
    {
        PlaceConfigs();
        Write("cond_encoder/pytorch_model.safetensors", ("linear.0.weight", Filled(new TensorShape(2, 3), 1f)));
        Write("vae/pytorch_model.safetensors", ("sal.post_kl.weight", Filled(new TensorShape(2, 2), 10f)), ("sal.encoder.input_proj.weight", Filled(new TensorShape(2), 0f)));
        Write("model/pytorch_model.safetensors", ("blocks.0.attn.weight", Filled(new TensorShape(2, 2), 100f)), ("final_layer.bias", Filled(new TensorShape(3), 7f)));

        using OmniCheckpoint full = OmniCheckpoint.Load(_folder, halfWeights: false);
        using OmniCheckpoint half = OmniCheckpoint.Load(_folder, halfWeights: true);

        Assert.Equal([1f, 2f, 3f, 4f, 5f, 6f], full.Control["linear.0.weight"].AsSpan<float>().ToArray());
        // Only the decoder half of the VAE file is read.
        Assert.Equal(["sal.post_kl.weight"], full.Vae.Keys);
        Assert.Equal([100f, 101f, 102f, 103f], full.Dit["blocks.0.attn.weight"].AsSpan<float>().ToArray());
        Assert.Equal(DType.F16, half.Dit["blocks.0.attn.weight"].DType);
        Assert.Equal([(Half)100f, (Half)101f, (Half)102f, (Half)103f], half.Dit["blocks.0.attn.weight"].AsSpan<Half>().ToArray());
        Assert.Equal(DType.F32, half.Dit["final_layer.bias"].DType);
        Assert.Equal(DType.F32, half.Vae["sal.post_kl.weight"].DType);
    }

    [Fact]
    public void Load_PrefersARepackOverAPickleAndNeverOpensThePickle()
    {
        PlaceConfigs();
        foreach (string pickle in new[] { "cond_encoder", "vae", "model" })
        {
            Directory.CreateDirectory(Path.Combine(_folder, pickle));
            File.WriteAllText(Path.Combine(_folder, pickle, "pytorch_model.bin"), "not a torch file");
        }
        Write("cond_encoder/pytorch_model.safetensors", ("linear.0.weight", Filled(new TensorShape(1), 1f)));
        Write("vae/pytorch_model.safetensors", ("sal.post_kl.weight", Filled(new TensorShape(1), 2f)));
        Write("model/pytorch_model.safetensors", ("blocks.0.attn.weight", Filled(new TensorShape(1), 3f)));

        using OmniCheckpoint checkpoint = OmniCheckpoint.Load(_folder, halfWeights: false);

        Assert.Equal(3f, checkpoint.Dit["blocks.0.attn.weight"].AsSpan<float>()[0]);
    }

    [Fact]
    public void Load_RefusesAHandPlacedRepackWhoseStoredBytesDoNotMatchItsShape()
    {
        PlaceConfigs();
        Write("cond_encoder/pytorch_model.safetensors", ("linear.0.weight", Filled(new TensorShape(1), 1f)));
        Write("vae/pytorch_model.safetensors", ("sal.post_kl.weight", Filled(new TensorShape(1), 2f)));
        // The header says a 2 x 2 F32 tensor and stores 8 bytes.
        byte[] header = System.Text.Encoding.UTF8.GetBytes("""{"blocks.0.attn.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,8]}}""");
        string path = Path.Combine(_folder, "model", "pytorch_model.safetensors");
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        using (FileStream file = File.Create(path))
        {
            file.Write(BitConverter.GetBytes((long)header.Length));
            file.Write(header);
            file.Write(new byte[8]);
        }

        InvalidDataException error = Assert.Throws<InvalidDataException>(() => OmniCheckpoint.Load(_folder, halfWeights: false));

        Assert.Contains("blocks.0.attn.weight", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Load_NamesTheMissingFileWhenNeitherThePickleNorItsRepackIsThere()
    {
        PlaceConfigs();

        FileNotFoundException error = Assert.Throws<FileNotFoundException>(() => OmniCheckpoint.Load(_folder, halfWeights: false));

        Assert.Contains("cond_encoder/pytorch_model.bin", error.Message.Replace('\\', '/'), StringComparison.Ordinal);
    }
}
