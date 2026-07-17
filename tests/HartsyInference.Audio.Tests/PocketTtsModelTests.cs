using HartsyInference.Audio.Models.Codecs.Mimi;
using HartsyInference.Audio.Models.PocketTts;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Synthetic-weights smoke tests for the continuous-latent Pocket-TTS stack on tiny configs: the
/// per-frame flow head produces a finite latent, the continuous-Mimi path round-trips shapes, and the full model
/// generates finite audio. CPU backend + deterministic random weights — CI-safe, no checkpoint needed.</summary>
public sealed unsafe class PocketTtsModelTests
{
    private static uint _rng = 0x9E37u;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return ((_rng & 0xFFFF) / 65535f - 0.5f) * 0.2f; }
    private static Tensor Fill(Tensor t) { float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand(); return t; }
    private static Tensor R(params int[] dims)
    {
        Span<long> ld = stackalloc long[dims.Length];
        for (int i = 0; i < dims.Length; i++) ld[i] = dims[i];
        return Fill(new Tensor(new TensorShape(ld), DType.F32));
    }
    private static Tensor Ones(int n) { Tensor t = new(new TensorShape(n), DType.F32); float* p = (float*)t.DataPointer; for (int i = 0; i < n; i++) p[i] = 1f; return t; }

    // Tiny Mimi: 2 stages, stride product 4 (PCM hop = 4), latent dim 8.
    private static MimiConfig TinyMimi() => new()
    {
        SampleRate = 24_000,
        Channels = 1,
        EncoderDim = 4,
        EncoderRates = [2, 2],
        LatentDim = 8,
        TransformerDim = 8,
        TransformerLayers = 1,
        TransformerHeads = 2,
        TransformerFfnDim = 16,
        ResidualDilations = [1],
        AcousticCodebooks = 3,
        CodebookSize = 16,
        CodebookDim = 4,
    };

    private static PocketTtsConfig TinyConfig() => new()
    {
        SampleRate = 24_000,
        FrameRateHz = 12,
        DModel = 16,
        NumLayers = 2,
        LatentDim = 8,
        NumHeads = 4,
        NumKvHeads = 2,
        FfnDim = 32,
        VocabSize = 24,
        MaxSequenceLength = 256,
        LsdDecodeSteps = 3,
        FlowHeadDim = 16,
        LatentCfgScale = 1.0f,
        TimeEmbedDim = 16,
        MaxFrames = 8,
        Mimi = TinyMimi(),
    };

    // ── Flow head ──

    [Fact]
    public void FlowHead_ProducesFiniteLatent()
    {
        PocketTtsConfig cfg = TinyConfig();
        Dictionary<string, Tensor> w = FlowHeadWeights(cfg, "flow_lm.flow_head");
        using CpuBackend backend = new();

        PocketTtsFlowHead head = new(cfg);
        head.LoadWeights(w, "flow_lm.flow_head");

        using Tensor cond = R(1, cfg.DModel);
        using Tensor latent = head.SampleLatent(backend, cond, seed: 7);

        Assert.Equal(2, latent.Shape.Rank);
        Assert.Equal(cfg.LatentDim, (int)latent.Shape[1]);
        AssertFinite(latent, "flow-head latent");
        DisposeAll(w);
    }

    // ── Continuous-Mimi round trip ──

    [Fact]
    public void MimiContinuousLatent_RoundTripsShapes()
    {
        MimiConfig mimi = TinyMimi();
        Dictionary<string, Tensor> w = MimiWeights(mimi);
        using CpuBackend backend = new();

        MimiContinuousLatent codec = new(mimi);
        codec.LoadWeights(w);

        int hop = 2 * 2;                 // product of EncoderRates
        int tFrames = 6;
        int tPcm = tFrames * hop;
        using Tensor pcm = R(1, 1, tPcm);

        using Tensor latent = codec.EncodeToLatent(backend, pcm, 1, tPcm);
        Assert.Equal(mimi.LatentDim, (int)latent.Shape[1]);
        AssertFinite(latent, "encoded latent");

        int latentFrames = (int)latent.Shape[2];
        using Tensor recon = codec.DecodeFromLatent(backend, latent, 1, latentFrames);
        Assert.Equal(1, (int)recon.Shape[1]);
        Assert.Equal(latentFrames * hop, (int)recon.Shape[2]);
        AssertFinite(recon, "decoded pcm");
        DisposeAll(w);
    }

    // ── Full model ──

    [Fact]
    public void Model_GeneratesFiniteAudio()
    {
        PocketTtsConfig cfg = TinyConfig();
        Dictionary<string, Tensor> w = AllWeights(cfg);
        using CpuBackend backend = new();

        using PocketTtsModel model = new(cfg);
        model.LoadWeights(w);

        int[] text = [1, 4, 9, 2];
        float[] audio = model.Generate(backend, text, voiceState: null, seed: 3, numFrames: 5);

        Assert.NotEmpty(audio);
        foreach (float s in audio) Assert.True(float.IsFinite(s), "generated audio sample must be finite");
        DisposeAll(w);
    }

    // ── Weight synthesis ──

    private static Dictionary<string, Tensor> AllWeights(PocketTtsConfig cfg)
    {
        Dictionary<string, Tensor> w = new();
        // Transformer body (headless Qwen2): layers + final norm, plus the Pocket-TTS-specific heads.
        int h = cfg.DModel, qDim = cfg.NumHeads * (h / cfg.NumHeads), kvDim = cfg.NumKvHeads * (h / cfg.NumHeads);
        const string lm = "flow_lm.transformer";
        w[$"{lm}.norm.weight"] = Ones(h);
        for (int i = 0; i < cfg.NumLayers; i++)
        {
            string p = $"{lm}.layers.{i}";
            w[$"{p}.input_layernorm.weight"] = Ones(h);
            w[$"{p}.post_attention_layernorm.weight"] = Ones(h);
            w[$"{p}.self_attn.q_proj.weight"] = R(qDim, h);
            w[$"{p}.self_attn.k_proj.weight"] = R(kvDim, h);
            w[$"{p}.self_attn.v_proj.weight"] = R(kvDim, h);
            w[$"{p}.self_attn.o_proj.weight"] = R(h, qDim);
            w[$"{p}.mlp.gate_proj.weight"] = R(cfg.FfnDim, h);
            w[$"{p}.mlp.up_proj.weight"] = R(cfg.FfnDim, h);
            w[$"{p}.mlp.down_proj.weight"] = R(h, cfg.FfnDim);
        }
        w[$"{lm}.text_embed.weight"] = R(cfg.VocabSize, h);
        w[$"{lm}.latent_in.weight"] = R(h, cfg.LatentDim);
        w[$"{lm}.latent_in.bias"] = R(h);
        w[$"{lm}.audio_bos"] = R(h);

        foreach (KeyValuePair<string, Tensor> kv in FlowHeadWeights(cfg, "flow_lm.flow_head")) w[kv.Key] = kv.Value;
        foreach (KeyValuePair<string, Tensor> kv in MimiWeights(cfg.Mimi)) w[kv.Key] = kv.Value;
        return w;
    }

    private static Dictionary<string, Tensor> FlowHeadWeights(PocketTtsConfig cfg, string prefix)
    {
        int hid = cfg.FlowHeadDim > 0 ? cfg.FlowHeadDim : cfg.DModel;
        int timeDim = cfg.TimeEmbedDim > 0 ? cfg.TimeEmbedDim : hid;
        return new Dictionary<string, Tensor>
        {
            [$"{prefix}.time_in.weight"] = R(hid, timeDim),
            [$"{prefix}.time_in.bias"] = R(hid),
            [$"{prefix}.cond_in.weight"] = R(hid, cfg.DModel),
            [$"{prefix}.cond_in.bias"] = R(hid),
            [$"{prefix}.in.weight"] = R(hid, cfg.LatentDim),
            [$"{prefix}.in.bias"] = R(hid),
            [$"{prefix}.mlp.0.weight"] = R(hid, hid),
            [$"{prefix}.mlp.0.bias"] = R(hid),
            [$"{prefix}.out.weight"] = R(cfg.LatentDim, hid),
            [$"{prefix}.out.bias"] = R(cfg.LatentDim),
        };
    }

    // Builds the SEANet encoder/decoder + transformer-of-codecs keys the continuous-Mimi path consumes. Mirrors
    // the Sequential-index scheme in SeaNetEncoder/SeaNetDecoder (LstmLayers == 0, so no LSTM keys).
    private static Dictionary<string, Tensor> MimiWeights(MimiConfig c)
    {
        Dictionary<string, Tensor> w = new();
        int nFilters = c.EncoderDim;
        int stages = c.EncoderRates.Count;
        int residual = c.ResidualDilations.Count;
        const int compress = 2;
        const int k = 7, rk = 3;

        // ── Encoder ── stem at index 0.
        AddConv(w, "encoder.model.0.conv.conv", nFilters, c.Channels, k);
        int seq = 1;
        int dim = nFilters;
        int[] ratiosRev = new int[stages];
        for (int i = 0; i < stages; i++) ratiosRev[i] = c.EncoderRates[stages - 1 - i];
        for (int i = 0; i < stages; i++)
        {
            int dimIn = nFilters * (1 << i);
            for (int j = 0; j < residual; j++)
            {
                AddConv(w, $"encoder.model.{seq}.block.1.conv.conv", dimIn / compress, dimIn, rk);
                AddConv(w, $"encoder.model.{seq}.block.3.conv.conv", dimIn, dimIn / compress, 1);
                seq++;
            }
            seq++;     // ELU
            int ratio = ratiosRev[i];
            AddConv(w, $"encoder.model.{seq}.conv.conv", dimIn * 2, dimIn, ratio * 2);
            seq++;
            dim = dimIn * 2;
        }
        seq++;         // final ELU
        AddConv(w, $"encoder.model.{seq}.conv.conv", c.LatentDim, dim, k);

        // ── Decoder ── initial conv at index 0, no LSTM (index 1 unused), stages then final.
        int maxChannels = nFilters * (1 << stages);
        AddConv(w, "decoder.model.0.conv.conv", maxChannels, c.LatentDim, k);
        seq = 1;       // SeaNetDecoder (no LSTM) starts seqIdx at 1; first ELU→upsample lands at model.2.
        for (int i = 0; i < stages; i++)
        {
            int dimInD = nFilters * (1 << (stages - i));
            int dimOutD = dimInD / 2;
            seq++;     // ELU
            AddConvTranspose(w, $"decoder.model.{seq}.convtr.convtr", dimInD, dimOutD, c.EncoderRates[i] * 2);
            seq++;
            for (int j = 0; j < residual; j++)
            {
                AddConv(w, $"decoder.model.{seq}.block.1.conv.conv", dimOutD / compress, dimOutD, rk);
                AddConv(w, $"decoder.model.{seq}.block.3.conv.conv", dimOutD, dimOutD / compress, 1);
                seq++;
            }
        }
        seq++;         // final ELU
        AddConv(w, $"decoder.model.{seq}.conv.conv", c.Channels, nFilters, k);

        // ── Transformer-of-codecs (encoder + decoder side) ──
        AddMimiTransformer(w, "encoder_transformer", c);
        AddMimiTransformer(w, "decoder_transformer", c);
        return w;
    }

    private static void AddMimiTransformer(Dictionary<string, Tensor> w, string prefix, MimiConfig c)
    {
        int d = c.TransformerDim;
        for (int i = 0; i < c.TransformerLayers; i++)
        {
            string p = $"{prefix}.layers.{i}";
            // HF Mimi layout: input_layernorm / post_attention_layernorm (with bias), self_attn.{q,k,v,o}_proj,
            // mlp.fc1/fc2, and per-sublayer LayerScale (.scale).
            w[$"{p}.input_layernorm.weight"] = Ones(d); w[$"{p}.input_layernorm.bias"] = R(d);
            w[$"{p}.post_attention_layernorm.weight"] = Ones(d); w[$"{p}.post_attention_layernorm.bias"] = R(d);
            w[$"{p}.self_attn.q_proj.weight"] = R(d, d);
            w[$"{p}.self_attn.k_proj.weight"] = R(d, d);
            w[$"{p}.self_attn.v_proj.weight"] = R(d, d);
            w[$"{p}.self_attn.o_proj.weight"] = R(d, d);
            w[$"{p}.mlp.fc1.weight"] = R(c.TransformerFfnDim, d);
            w[$"{p}.mlp.fc2.weight"] = R(d, c.TransformerFfnDim);
            w[$"{p}.self_attn_layer_scale.scale"] = R(d);
            w[$"{p}.mlp_layer_scale.scale"] = R(d);
        }
    }

    // Conv1d weight: [outCh, inCh, k]. weight_g/weight_v with g = ones (fused weight == normalized v).
    private static void AddConv(Dictionary<string, Tensor> w, string prefix, int outCh, int inCh, int k)
    {
        w[$"{prefix}.weight_g"] = Ones(outCh);
        w[$"{prefix}.weight_v"] = R(outCh, inCh, k);
        w[$"{prefix}.bias"] = R(outCh);
    }

    // ConvTranspose1d weight: [inCh, outCh, k] (PyTorch layout); weight-norm is along dim 0 = inCh (C_in).
    private static void AddConvTranspose(Dictionary<string, Tensor> w, string prefix, int inCh, int outCh, int k)
    {
        w[$"{prefix}.weight_g"] = Ones(inCh);   // WeightNormT dim0 = C_in
        w[$"{prefix}.weight_v"] = R(inCh, outCh, k);
        w[$"{prefix}.bias"] = R(outCh);
    }

    private static void AssertFinite(Tensor t, string label)
    {
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++)
            Assert.True(float.IsFinite(p[i]), $"{label}: non-finite at {i}");
    }

    private static void DisposeAll(Dictionary<string, Tensor> w) { foreach (Tensor t in w.Values) t.Dispose(); }
}
