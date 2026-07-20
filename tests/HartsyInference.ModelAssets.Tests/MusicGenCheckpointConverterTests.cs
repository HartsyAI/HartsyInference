using Xunit;
using HartsyInference.Audio.Models.Codecs.EnCodec;
using HartsyInference.Audio.Models.Music;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>MusicGen converter: decoder prefix probing (combined vs standalone HF layouts), the HF transformers →
/// Meta AudioCraft EnCodec rename (incl. shape-based ConvTranspose detection), and end-to-end loads into tiny
/// engine <see cref="MusicGenDecoder"/> / <see cref="EnCodec"/> instances with no KeyNotFound.</summary>
public unsafe class MusicGenCheckpointConverterTests
{
    [Theory]
    // Combined MusicgenForConditionalGeneration layout → engine names.
    [InlineData("decoder.model.decoder.embed_tokens.0.weight", "model.decoder.embed_tokens.0.weight")]
    [InlineData("decoder.model.decoder.layers.3.self_attn.q_proj.weight", "model.decoder.layers.3.self_attn.q_proj.weight")]
    [InlineData("decoder.model.decoder.layer_norm.bias", "model.decoder.layer_norm.bias")]
    [InlineData("decoder.lm_heads.1.weight", "lm_heads.1.weight")]
    [InlineData("enc_to_dec_proj.weight", "enc_to_dec_proj.weight")]
    // Standalone MusicgenForCausalLM layout passes through.
    [InlineData("model.decoder.embed_tokens.2.weight", "model.decoder.embed_tokens.2.weight")]
    [InlineData("lm_heads.0.weight", "lm_heads.0.weight")]
    public void MapDecoderKey_HandlesBothHfLayouts(string key, string expected)
    {
        Assert.Equal(expected, MusicGenCheckpointConverter.MapDecoderKey(key));
    }

    [Theory]
    [InlineData("text_encoder.encoder.block.0.layer.0.SelfAttention.q.weight")]
    [InlineData("audio_encoder.encoder.layers.0.conv.weight_g")]
    [InlineData("decoder.model.decoder.embed_positions.weights")]
    public void MapDecoderKey_DropsNonDecoderComponents(string key)
    {
        Assert.Null(MusicGenCheckpointConverter.MapDecoderKey(key));
    }

    [Theory]
    // HF transformers EncodecModel naming → Meta AudioCraft naming.
    [InlineData("encoder.layers.0.conv.weight_g", false, "encoder.model.0.conv.conv.weight_g")]
    [InlineData("encoder.layers.1.block.3.conv.bias", false, "encoder.model.1.block.3.conv.conv.bias")]
    [InlineData("encoder.layers.7.lstm.weight_ih_l0", false, "encoder.model.7.lstm.weight_ih_l0")]
    [InlineData("decoder.layers.3.conv.weight_v", true, "decoder.model.3.convtr.convtr.weight_v")]
    [InlineData("decoder.layers.9.conv.bias", false, "decoder.model.9.conv.conv.bias")]
    [InlineData("quantizer.layers.1.codebook.embed", false, "quantizer.vq.layers.1._codebook.embed")]
    // torch ≥2.1 parametrizations spelling normalizes to the classic pair names.
    [InlineData("encoder.layers.0.conv.parametrizations.weight.original0", false, "encoder.model.0.conv.conv.weight_g")]
    // Meta-named keys pass through untouched.
    [InlineData("decoder.model.3.convtr.convtr.weight_g", false, "decoder.model.3.convtr.convtr.weight_g")]
    public void MapEnCodecKey_MapsHfToMetaNaming(string key, bool transpose, string expected)
    {
        Assert.Equal(expected, MusicGenCheckpointConverter.MapEnCodecKey(key, transpose));
    }

    [Theory]
    [InlineData("quantizer.layers.0.codebook.inited")]
    [InlineData("quantizer.layers.0.codebook.cluster_size")]
    [InlineData("quantizer.layers.0.codebook.embed_avg")]
    public void MapEnCodecKey_DropsTrainingOnlyQuantizerBuffers(string key)
    {
        Assert.Null(MusicGenCheckpointConverter.MapEnCodecKey(key));
    }

    [Fact]
    public void LoadDecoder_CombinedCheckpoint_EngineLoadsWithoutKeyNotFound()
    {
        MusicGenConfig cfg = new() { Hidden = 8, NumLayers = 1, NumHeads = 2, TextDim = 4, NumCodebooks = 2, CodebookSize = 6, SpecialToken = 6 };
        string path = Path.Combine(Path.GetTempPath(), $"musicgen_{Guid.NewGuid():N}.safetensors");
        Dictionary<string, Tensor> synthetic = BuildCombinedDecoderCheckpoint(cfg);
        try
        {
            SafeTensorsWriter.Save(path, synthetic);
            (Dictionary<string, Tensor> weights, SafeTensorsLoader loader) = MusicGenCheckpointConverter.LoadDecoder(path);
            using (loader)
            {
                Assert.False(weights.ContainsKey("text_encoder.shared.weight"));
                Assert.DoesNotContain(weights.Keys, k => k.Contains("embed_positions"));
                using MusicGenDecoder decoder = new(cfg);
                decoder.LoadWeights(weights);
            }
        }
        finally
        {
            foreach (Tensor t in synthetic.Values) t.Dispose();
            File.Delete(path);
        }
    }

    // SyntheticSmoke: loads a synthetic HF-EnCodec checkpoint into the (not yet real-weight-validated)
    // MusicGen EnCodec. Currently fails on a ConvTranspose weight-norm dim mismatch (weightG got 4,
    // expects 8 = C_in) — tracked in PARITY_VERIFICATION.md until the converter/decoder is validated.
    [Trait("Category", "SyntheticSmoke")]
    [Fact]
    public void LoadEnCodec_HfTransformersCheckpoint_EngineLoadsWithoutKeyNotFound()
    {
        EnCodecConfig cfg = TinyEnCodecConfig();
        string path = Path.Combine(Path.GetTempPath(), $"encodec_{Guid.NewGuid():N}.safetensors");
        Dictionary<string, Tensor> synthetic = BuildHfEnCodecCheckpoint();
        try
        {
            SafeTensorsWriter.Save(path, synthetic);
            (Dictionary<string, Tensor> weights, SafeTensorsLoader loader) = MusicGenCheckpointConverter.LoadEnCodec(path);
            using (loader)
            {
                Assert.True(weights.ContainsKey("decoder.model.3.convtr.convtr.weight_g"));
                Assert.True(weights.ContainsKey("quantizer.vq.layers.0._codebook.embed"));
                Assert.False(weights.ContainsKey("quantizer.vq.layers.0._codebook.cluster_size"));
                EnCodec codec = new(cfg);
                codec.LoadWeights(weights);
            }
        }
        finally
        {
            foreach (Tensor t in synthetic.Values) t.Dispose();
            File.Delete(path);
        }
    }

    /// <summary>Tiny SEANet: NFilters=2, Ratios=[2,2], 1 residual layer, 1 LSTM layer, latent 3, 2×4-entry codebooks.</summary>
    private static EnCodecConfig TinyEnCodecConfig() => new()
    {
        NFilters = 2, Ratios = [2, 2], NResidualLayers = 1, LstmLayers = 1, LatentDim = 3,
        KernelSize = 3, LastKernelSize = 3, ResidualKernelSize = 3, Compress = 2,
        VqMaxCodebooks = 2, VqCodebookSize = 4, ActiveCodebooks = 2,
    };

    private static Dictionary<string, Tensor> BuildCombinedDecoderCheckpoint(MusicGenConfig cfg)
    {
        Dictionary<string, Tensor> w = new();
        for (int cb = 0; cb < cfg.NumCodebooks; cb++)
        {
            w[$"decoder.model.decoder.embed_tokens.{cb}.weight"] = T(cfg.CodebookSize + 1, cfg.Hidden);
            w[$"decoder.lm_heads.{cb}.weight"] = T(cfg.CodebookSize, cfg.Hidden);
        }
        w["enc_to_dec_proj.weight"] = T(cfg.Hidden, cfg.TextDim);
        w["enc_to_dec_proj.bias"] = T(cfg.Hidden);
        for (int i = 0; i < cfg.NumLayers; i++)
        {
            string p = $"decoder.model.decoder.layers.{i}";
            foreach (string attn in new[] { "self_attn", "encoder_attn" })
            {
                w[$"{p}.{attn}_layer_norm.weight"] = T(cfg.Hidden);
                w[$"{p}.{attn}_layer_norm.bias"] = T(cfg.Hidden);
                foreach (string proj in new[] { "q_proj", "k_proj", "v_proj", "out_proj" })
                    w[$"{p}.{attn}.{proj}.weight"] = T(cfg.Hidden, cfg.Hidden);
            }
            w[$"{p}.final_layer_norm.weight"] = T(cfg.Hidden);
            w[$"{p}.final_layer_norm.bias"] = T(cfg.Hidden);
            w[$"{p}.fc1.weight"] = T(cfg.FfnDim, cfg.Hidden);
            w[$"{p}.fc1.bias"] = T(cfg.FfnDim);
            w[$"{p}.fc2.weight"] = T(cfg.Hidden, cfg.FfnDim);
            w[$"{p}.fc2.bias"] = T(cfg.Hidden);
        }
        w["decoder.model.decoder.layer_norm.weight"] = T(cfg.Hidden);
        w["decoder.model.decoder.layer_norm.bias"] = T(cfg.Hidden);
        // Siblings the converter must drop.
        w["decoder.model.decoder.embed_positions.weights"] = T(16, cfg.Hidden);
        w["text_encoder.shared.weight"] = T(3, cfg.TextDim);
        w["audio_encoder.encoder.layers.0.conv.weight_g"] = T(2, 1, 1);
        return w;
    }

    /// <summary>HF transformers EncodecModel state dict for <see cref="TinyEnCodecConfig"/>: encoder seq indices
    /// 0=stem, 1/4=resnet blocks, 3/6=downsample convs, 7=LSTM, 9=final; decoder 0=initial, 1=LSTM, 3/6=transpose
    /// convs, 4/7=resnet blocks, 9=final. Transpose convs carry the dim=1 weight-norm shape [1, C_out, 1].</summary>
    private static Dictionary<string, Tensor> BuildHfEnCodecCheckpoint()
    {
        Dictionary<string, Tensor> w = new();
        AddHfConv(w, "encoder.layers.0", cOut: 2, cIn: 1, k: 3);
        AddHfResnet(w, "encoder.layers.1", dim: 2, k: 3);
        AddHfConv(w, "encoder.layers.3", cOut: 4, cIn: 2, k: 4);
        AddHfResnet(w, "encoder.layers.4", dim: 4, k: 3);
        AddHfConv(w, "encoder.layers.6", cOut: 8, cIn: 4, k: 4);
        AddHfLstm(w, "encoder.layers.7", dim: 8);
        AddHfConv(w, "encoder.layers.9", cOut: 3, cIn: 8, k: 3);

        AddHfConv(w, "decoder.layers.0", cOut: 8, cIn: 3, k: 3);
        AddHfLstm(w, "decoder.layers.1", dim: 8);
        AddHfTransposeConv(w, "decoder.layers.3", cIn: 8, cOut: 4, k: 4);
        AddHfResnet(w, "decoder.layers.4", dim: 4, k: 3);
        AddHfTransposeConv(w, "decoder.layers.6", cIn: 4, cOut: 2, k: 4);
        AddHfResnet(w, "decoder.layers.7", dim: 2, k: 3);
        AddHfConv(w, "decoder.layers.9", cOut: 1, cIn: 2, k: 3);

        for (int q = 0; q < 2; q++)
        {
            w[$"quantizer.layers.{q}.codebook.embed"] = T(4, 3);
            w[$"quantizer.layers.{q}.codebook.embed_avg"] = T(4, 3);
            w[$"quantizer.layers.{q}.codebook.cluster_size"] = T(4);
            w[$"quantizer.layers.{q}.codebook.inited"] = T(1);
        }
        return w;
    }

    private static void AddHfConv(Dictionary<string, Tensor> w, string prefix, int cOut, int cIn, int k)
    {
        w[$"{prefix}.conv.weight_g"] = T(cOut, 1, 1);
        w[$"{prefix}.conv.weight_v"] = T(cOut, cIn, k);
        w[$"{prefix}.conv.bias"] = T(cOut);
    }

    private static void AddHfTransposeConv(Dictionary<string, Tensor> w, string prefix, int cIn, int cOut, int k)
    {
        w[$"{prefix}.conv.weight_g"] = T(1, cOut, 1);
        w[$"{prefix}.conv.weight_v"] = T(cIn, cOut, k);
        w[$"{prefix}.conv.bias"] = T(cOut);
    }

    /// <summary>SEANet resnet block (compress=2): block.1 = dim→dim/2 k=residual kernel, block.3 = dim/2→dim k=1.</summary>
    private static void AddHfResnet(Dictionary<string, Tensor> w, string prefix, int dim, int k)
    {
        AddHfConv(w, $"{prefix}.block.1", cOut: dim / 2 < 1 ? 1 : dim / 2, cIn: dim, k: k);
        AddHfConv(w, $"{prefix}.block.3", cOut: dim, cIn: dim / 2 < 1 ? 1 : dim / 2, k: 1);
    }

    private static void AddHfLstm(Dictionary<string, Tensor> w, string prefix, int dim)
    {
        w[$"{prefix}.lstm.weight_ih_l0"] = T(4 * dim, dim);
        w[$"{prefix}.lstm.weight_hh_l0"] = T(4 * dim, dim);
        w[$"{prefix}.lstm.bias_ih_l0"] = T(4 * dim);
        w[$"{prefix}.lstm.bias_hh_l0"] = T(4 * dim);
    }

    private static Tensor T(params long[] dims)
    {
        Tensor t = new(new TensorShape(dims), DType.F32);
        new Span<float>((void*)t.DataPointer, (int)t.ElementCount).Fill(0.5f);
        return t;
    }
}
