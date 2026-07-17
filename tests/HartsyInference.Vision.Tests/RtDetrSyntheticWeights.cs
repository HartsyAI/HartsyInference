using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Detection;

namespace HartsyInference.Vision.Tests;

/// <summary>Builds a full deterministic synthetic weight dictionary for an <see cref="RtDetrConfig"/>,
/// matching every key <see cref="RtDetrModel.LoadWeights"/> expects. Weights are small (bounded) so a
/// many-layer forward stays numerically finite; LayerNorm weights are 1 / biases 0.</summary>
public static class RtDetrSyntheticWeights
{
    /// <summary>Assembles the weight map for a config's full module tree.</summary>
    public static Dictionary<string, Tensor> Build(RtDetrConfig c)
    {
        Dictionary<string, Tensor> w = new();
        int hidden = c.HiddenDim;
        int ffn = c.FeedForwardDim;
        int nc = c.NumClasses;
        int stem0 = c.BackboneChannels[0], stem1 = c.BackboneChannels[1];
        int c3 = c.BackboneChannels[2], c4 = c.BackboneChannels[3], c5 = c.BackboneChannels[4];
        int deformDim = c.NumHeads * c.NumLevels * c.NumPoints;

        // ── Backbone ──
        Conv(w, "backbone.stem.0.conv", stem0, 3, 3);
        Conv(w, "backbone.stem.1.conv", stem1, stem0, 3);
        Conv(w, "backbone.stage3.downsample.conv", c3, stem1, 3);
        C2f(w, "backbone.stage3.c2f", c3, c3, c.BackboneRepeat);
        Conv(w, "backbone.stage4.downsample.conv", c4, c3, 3);
        C2f(w, "backbone.stage4.c2f", c4, c4, c.BackboneRepeat);
        Conv(w, "backbone.stage5.downsample.conv", c5, c4, 3);
        C2f(w, "backbone.stage5.c2f", c5, c5, c.BackboneRepeat);

        // ── Encoder ──
        Conv(w, "encoder.input_proj.0.conv", hidden, c3, 1);
        Conv(w, "encoder.input_proj.1.conv", hidden, c4, 1);
        Conv(w, "encoder.input_proj.2.conv", hidden, c5, 1);
        Linear(w, "encoder.aifi.qkv", 3 * hidden, hidden);
        Linear(w, "encoder.aifi.proj", hidden, hidden);
        Norm(w, "encoder.aifi.norm1", hidden);
        Norm(w, "encoder.aifi.norm2", hidden);
        Linear(w, "encoder.aifi.ffn.fc1", ffn, hidden);
        Linear(w, "encoder.aifi.ffn.fc2", hidden, ffn);
        Conv(w, "encoder.ccfm.fuse4.conv", hidden, 2 * hidden, 1);
        Conv(w, "encoder.ccfm.fuse3.conv", hidden, 2 * hidden, 1);
        Conv(w, "encoder.ccfm.down3.conv", hidden, hidden, 3);
        Conv(w, "encoder.ccfm.fuse_n4.conv", hidden, 2 * hidden, 1);
        Conv(w, "encoder.ccfm.down4.conv", hidden, hidden, 3);
        Conv(w, "encoder.ccfm.fuse_n5.conv", hidden, 2 * hidden, 1);

        // ── Decoder ──
        Linear(w, "decoder.enc_output.linear", hidden, hidden);
        Norm(w, "decoder.enc_output.norm", hidden);
        Linear(w, "decoder.enc_score", nc, hidden);
        Linear(w, "decoder.enc_bbox.0", hidden, hidden);
        Linear(w, "decoder.enc_bbox.1", hidden, hidden);
        Linear(w, "decoder.enc_bbox.2", 4, hidden);
        Linear(w, "decoder.query_pos_head.0", 2 * hidden, 4);
        Linear(w, "decoder.query_pos_head.1", hidden, 2 * hidden);
        for (int i = 0; i < c.NumDecoderLayers; i++)
        {
            string b = $"decoder.layers.{i}";
            Linear(w, $"{b}.self_attn.qkv", 3 * hidden, hidden);
            Linear(w, $"{b}.self_attn.proj", hidden, hidden);
            Norm(w, $"{b}.norm1", hidden);
            Linear(w, $"{b}.cross_attn.sampling_offsets", deformDim * 2, hidden);
            Linear(w, $"{b}.cross_attn.attention_weights", deformDim, hidden);
            Linear(w, $"{b}.cross_attn.value_proj", hidden, hidden);
            Linear(w, $"{b}.cross_attn.output_proj", hidden, hidden);
            Norm(w, $"{b}.norm2", hidden);
            Linear(w, $"{b}.ffn.fc1", ffn, hidden);
            Linear(w, $"{b}.ffn.fc2", hidden, ffn);
            Norm(w, $"{b}.norm3", hidden);
            Linear(w, $"decoder.dec_score.{i}", nc, hidden);
            Linear(w, $"decoder.dec_bbox.{i}.0", hidden, hidden);
            Linear(w, $"decoder.dec_bbox.{i}.1", hidden, hidden);
            Linear(w, $"decoder.dec_bbox.{i}.2", 4, hidden);
        }
        return w;
    }

    private static void Conv(Dictionary<string, Tensor> w, string prefix, int outC, int inC, int k)
    {
        w[$"{prefix}.weight"] = Rand([outC, inC, k, k]);
        w[$"{prefix}.bias"] = Zeros([outC]);
    }

    private static void Linear(Dictionary<string, Tensor> w, string prefix, int outF, int inF)
    {
        w[$"{prefix}.weight"] = Rand([outF, inF]);
        w[$"{prefix}.bias"] = Zeros([outF]);
    }

    private static void Norm(Dictionary<string, Tensor> w, string prefix, int dim)
    {
        w[$"{prefix}.weight"] = Ones([dim]);
        w[$"{prefix}.bias"] = Zeros([dim]);
    }

    private static void C2f(Dictionary<string, Tensor> w, string prefix, int inC, int outC, int n)
    {
        int hid = (int)(outC * 0.5f);
        Conv(w, $"{prefix}.cv1.conv", 2 * hid, inC, 1);
        Conv(w, $"{prefix}.cv2.conv", outC, (2 + n) * hid, 1);
        for (int i = 0; i < n; i++)
        {
            Conv(w, $"{prefix}.m.{i}.cv1.conv", hid, hid, 3);
            Conv(w, $"{prefix}.m.{i}.cv2.conv", hid, hid, 3);
        }
    }

    private static Tensor Rand(int[] dims)
    {
        long[] longDims = Array.ConvertAll(dims, x => (long)x);
        Tensor t = new(new TensorShape(longDims), DType.F32);
        Span<float> span = t.AsSpan<float>();
        // Deterministic small values keyed to the tensor size — bounded so the deep forward stays finite.
        int state = unchecked((int)0x9E3779B1) ^ dims.Length;
        for (int i = 0; i < dims.Length; i++)
            state = state * 31 + dims[i];
        for (int i = 0; i < span.Length; i++)
        {
            state = state * 1103515245 + 12345;
            float u = ((state >> 8) & 0xFFFF) / 65535f;   // [0,1)
            span[i] = (u - 0.5f) * 0.1f;                  // [-0.05, 0.05)
        }
        return t;
    }

    private static Tensor Ones(int[] dims)
    {
        Tensor t = new(new TensorShape(Array.ConvertAll(dims, x => (long)x)), DType.F32);
        t.AsSpan<float>().Fill(1f);
        return t;
    }

    private static Tensor Zeros(int[] dims)
    {
        Tensor t = new(new TensorShape(Array.ConvertAll(dims, x => (long)x)), DType.F32);
        t.AsSpan<float>().Clear();
        return t;
    }
}
