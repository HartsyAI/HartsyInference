using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Models.Denoisers.DiTBlocks;
using SharpInference.ModelHandler.CheckpointConverters;
using Xunit;

namespace SharpInference.Diffusion.Tests;

/// <summary>Unit tests for Microsoft Lens scaffolding — config presets, RoPE math, transformer construction, checkpoint-converter QKV split. End-to-end generation requires the actual `microsoft/Lens` checkpoint + GPT-OSS encoder weights and is gated separately.</summary>
public sealed class LensTests
{
    [Fact]
    public void Config_Default_Matches_TransformerJson()
    {
        LensConfig c = LensConfig.Default;
        Assert.Equal(1536, c.HiddenSize);
        Assert.Equal(48, c.NumLayers);
        Assert.Equal(24, c.NumHeads);
        Assert.Equal(64, c.HeadDim);
        Assert.Equal(4096, c.MlpDim);
        Assert.Equal(2, c.PatchSize);
        Assert.Equal(128, c.InChannels);
        Assert.Equal(32, c.OutChannels);
        Assert.Equal(2880, c.EncoderHiddenDim);
        Assert.Equal(new[] { 5, 11, 17, 23 }, c.SelectedEncoderLayers);
        Assert.Equal(new[] { 8, 28, 28 }, c.AxesDimsRope);
        Assert.Equal(10000, c.RopeTheta);
        Assert.Equal(20, c.DefaultSteps);
        Assert.Equal(5.0f, c.DefaultCfgScale);
    }

    [Fact]
    public void Config_Turbo_Differs_Only_In_Sampling_Defaults()
    {
        LensConfig def = LensConfig.Default;
        LensConfig turbo = LensConfig.Turbo;
        Assert.Equal(def.HiddenSize, turbo.HiddenSize);
        Assert.Equal(def.NumLayers, turbo.NumLayers);
        Assert.Equal(def.NumHeads, turbo.NumHeads);
        Assert.Equal(def.SelectedEncoderLayers, turbo.SelectedEncoderLayers);
        Assert.Equal(4, turbo.DefaultSteps);
        Assert.Equal(1.0f, turbo.DefaultCfgScale);
    }

    [Fact]
    public void Config_Base_Uses_50_Steps()
    {
        LensConfig basePreset = LensConfig.Base;
        Assert.Equal(50, basePreset.DefaultSteps);
        Assert.Equal(5.0f, basePreset.DefaultCfgScale);
        Assert.Equal(LensConfig.Default.NumLayers, basePreset.NumLayers);
    }

    [Fact]
    public void Rope_HeadDim_Equals_AxisSum()
    {
        LensRope rope = new LensRope([8, 28, 28], theta: 10000);
        Assert.Equal(64, rope.HeadDim);
        Assert.Equal(new[] { 8, 28, 28 }, rope.AxesDim.ToArray());
    }

    [Fact]
    public void Rope_Rejects_NonThreeAxis()
    {
        Assert.Throws<ArgumentException>(() => new LensRope([8, 28], theta: 10000));
        Assert.Throws<ArgumentException>(() => new LensRope([8, 28, 28, 16], theta: 10000));
    }

    [Fact]
    public void Rope_TextPositionStart_Matches_ScaleRopeTrue()
    {
        // scale_rope=True: max_vid_index = max(h//2, w//2).
        Assert.Equal(32, LensRope.ComputeTextPositionStart(64, 64));
        Assert.Equal(45, LensRope.ComputeTextPositionStart(90, 80));
        Assert.Equal(32, LensRope.ComputeTextPositionStart(64, 48));
    }

    [Fact]
    public unsafe void Rope_ApplyImage_DoesNot_Crash_For_Default_Grid()
    {
        LensRope rope = new LensRope([8, 28, 28], theta: 10000);
        int batch = 1, numHeads = 24, hPacked = 4, wPacked = 4;
        int imgSeqLen = hPacked * wPacked;
        int headDim = 64;
        TensorShape shape = new TensorShape(batch, numHeads, imgSeqLen, headDim);
        Tensor q = new Tensor(shape, DType.F32);
        Tensor k = new Tensor(shape, DType.F32);
        // Init to 1.0 so we know rotation actually mutates values.
        float* qPtr = (float*)q.DataPointer;
        float* kPtr = (float*)k.DataPointer;
        long count = shape.ElementCount;
        for (long i = 0; i < count; i++) { qPtr[i] = 1.0f; kPtr[i] = 1.0f; }

        rope.ApplyImage(q, k, batch, numHeads, hPacked, wPacked);

        // For the centered position grid: the row/col positions are not all zero, so most values rotate.
        // Just check that at least one position changed (not still 1.0).
        bool anyChanged = false;
        for (long i = 0; i < count && !anyChanged; i++)
            if (MathF.Abs(qPtr[i] - 1.0f) > 1e-5f) anyChanged = true;
        Assert.True(anyChanged, "Image RoPE should rotate at least one value away from 1.0");

        q.Dispose();
        k.Dispose();
    }

    [Fact]
    public unsafe void Rope_ApplyText_AtZeroPosition_IsIdentity()
    {
        LensRope rope = new LensRope([8, 28, 28], theta: 10000);
        int batch = 1, numHeads = 24, txtSeqLen = 1;
        int headDim = 64;
        TensorShape shape = new TensorShape(batch, numHeads, txtSeqLen, headDim);
        Tensor q = new Tensor(shape, DType.F32);
        Tensor k = new Tensor(shape, DType.F32);
        float* qPtr = (float*)q.DataPointer;
        float* kPtr = (float*)k.DataPointer;
        long count = shape.ElementCount;
        for (long i = 0; i < count; i++) { qPtr[i] = 1.0f; kPtr[i] = 1.0f; }

        // positionStart = 0, txtSeqLen = 1 → only token gets position [0, 0, 0] → cos=1, sin=0 → identity.
        rope.ApplyText(q, k, batch, numHeads, txtSeqLen, positionStart: 0);

        for (long i = 0; i < count; i++)
        {
            Assert.True(MathF.Abs(qPtr[i] - 1.0f) < 1e-6f, $"At position 0, RoPE should be identity; q[{i}]={qPtr[i]}");
            Assert.True(MathF.Abs(kPtr[i] - 1.0f) < 1e-6f, $"At position 0, RoPE should be identity; k[{i}]={kPtr[i]}");
        }

        q.Dispose();
        k.Dispose();
    }

    [Fact]
    public void Transformer_Construction_DoesNotAllocateWeights()
    {
        LensTransformer t = new LensTransformer(LensConfig.Default);
        Assert.Empty(t.EnumerateWeights());
        t.Dispose();
    }

    [Fact]
    public void Transformer_LoadWeights_ThrowsOnMissingRequiredKey()
    {
        LensTransformer t = new LensTransformer(LensConfig.Default);
        try
        {
            Dictionary<string, Tensor> empty = new();
            Assert.Throws<KeyNotFoundException>(() => t.LoadWeights(empty));
        }
        finally
        {
            t.Dispose();
        }
    }

    [Fact]
    public unsafe void CheckpointConverter_Splits_FusedImgQkv_Into_ToQKV()
    {
        // Build a synthetic fused img_qkv weight + bias matching the upstream layout.
        const int innerDim = 1536;
        const int inDim = 1536;
        TensorShape weightShape = new TensorShape(3 * innerDim, inDim);
        Tensor fusedW = new Tensor(weightShape, DType.F32);
        float* wPtr = (float*)fusedW.DataPointer;
        // Q rows = 1.0, K rows = 2.0, V rows = 3.0 — easy to distinguish after split.
        for (int r = 0; r < innerDim; r++)
            for (int c = 0; c < inDim; c++) wPtr[r * inDim + c] = 1.0f;
        for (int r = 0; r < innerDim; r++)
            for (int c = 0; c < inDim; c++) wPtr[(innerDim + r) * inDim + c] = 2.0f;
        for (int r = 0; r < innerDim; r++)
            for (int c = 0; c < inDim; c++) wPtr[(2 * innerDim + r) * inDim + c] = 3.0f;

        TensorShape biasShape = new TensorShape(3 * innerDim);
        Tensor fusedB = new Tensor(biasShape, DType.F32);
        float* bPtr = (float*)fusedB.DataPointer;
        for (int i = 0; i < innerDim; i++) bPtr[i] = 10.0f;
        for (int i = 0; i < innerDim; i++) bPtr[innerDim + i] = 20.0f;
        for (int i = 0; i < innerDim; i++) bPtr[2 * innerDim + i] = 30.0f;

        Dictionary<string, Tensor> input = new()
        {
            ["transformer_blocks.0.attn.img_qkv.weight"] = fusedW,
            ["transformer_blocks.0.attn.img_qkv.bias"] = fusedB,
        };

        LensCheckpointConverter.ConvertedWeights converted = LensCheckpointConverter.Convert(input);

        Assert.True(converted.Transformer.ContainsKey("transformer_blocks.0.attn.to_q.weight"));
        Assert.True(converted.Transformer.ContainsKey("transformer_blocks.0.attn.to_k.weight"));
        Assert.True(converted.Transformer.ContainsKey("transformer_blocks.0.attn.to_v.weight"));
        Assert.True(converted.Transformer.ContainsKey("transformer_blocks.0.attn.to_q.bias"));
        Assert.True(converted.Transformer.ContainsKey("transformer_blocks.0.attn.to_k.bias"));
        Assert.True(converted.Transformer.ContainsKey("transformer_blocks.0.attn.to_v.bias"));

        Tensor qW = converted.Transformer["transformer_blocks.0.attn.to_q.weight"];
        Tensor kW = converted.Transformer["transformer_blocks.0.attn.to_k.weight"];
        Tensor vW = converted.Transformer["transformer_blocks.0.attn.to_v.weight"];
        Assert.Equal(innerDim, (int)qW.Shape[0]);
        Assert.Equal(inDim, (int)qW.Shape[1]);
        Assert.Equal(1.0f, ((float*)qW.DataPointer)[0]);
        Assert.Equal(2.0f, ((float*)kW.DataPointer)[0]);
        Assert.Equal(3.0f, ((float*)vW.DataPointer)[0]);

        Tensor qB = converted.Transformer["transformer_blocks.0.attn.to_q.bias"];
        Tensor kB = converted.Transformer["transformer_blocks.0.attn.to_k.bias"];
        Tensor vB = converted.Transformer["transformer_blocks.0.attn.to_v.bias"];
        Assert.Equal(10.0f, ((float*)qB.DataPointer)[0]);
        Assert.Equal(20.0f, ((float*)kB.DataPointer)[0]);
        Assert.Equal(30.0f, ((float*)vB.DataPointer)[0]);

        // Dispose synthetic + split tensors.
        fusedW.Dispose();
        fusedB.Dispose();
        qW.Dispose(); kW.Dispose(); vW.Dispose();
        qB.Dispose(); kB.Dispose(); vB.Dispose();
    }

    [Fact]
    public unsafe void CheckpointConverter_Splits_FusedTxtQkv_Into_AddQKVProj()
    {
        const int innerDim = 1536;
        const int inDim = 1536;
        TensorShape weightShape = new TensorShape(3 * innerDim, inDim);
        Tensor fusedW = new Tensor(weightShape, DType.F32);
        TensorShape biasShape = new TensorShape(3 * innerDim);
        Tensor fusedB = new Tensor(biasShape, DType.F32);

        Dictionary<string, Tensor> input = new()
        {
            ["transformer_blocks.0.attn.txt_qkv.weight"] = fusedW,
            ["transformer_blocks.0.attn.txt_qkv.bias"] = fusedB,
        };

        LensCheckpointConverter.ConvertedWeights converted = LensCheckpointConverter.Convert(input);

        Assert.True(converted.Transformer.ContainsKey("transformer_blocks.0.attn.add_q_proj.weight"));
        Assert.True(converted.Transformer.ContainsKey("transformer_blocks.0.attn.add_k_proj.weight"));
        Assert.True(converted.Transformer.ContainsKey("transformer_blocks.0.attn.add_v_proj.weight"));
        Assert.True(converted.Transformer.ContainsKey("transformer_blocks.0.attn.add_q_proj.bias"));

        fusedW.Dispose();
        fusedB.Dispose();
        foreach (Tensor t in converted.Transformer.Values) t.Dispose();
    }

    [Fact]
    public void CheckpointConverter_PassesThrough_NonFused_Keys()
    {
        TensorShape simpleShape = new TensorShape(1536);
        Tensor norm = new Tensor(simpleShape, DType.F32);
        Dictionary<string, Tensor> input = new()
        {
            ["transformer_blocks.0.img_norm1.weight"] = norm,
            ["img_in.bias"] = norm,
            ["txt_norm.0.weight"] = norm,
        };

        LensCheckpointConverter.ConvertedWeights converted = LensCheckpointConverter.Convert(input);

        Assert.True(converted.Transformer.ContainsKey("transformer_blocks.0.img_norm1.weight"));
        Assert.True(converted.Transformer.ContainsKey("img_in.bias"));
        Assert.True(converted.Transformer.ContainsKey("txt_norm.0.weight"));

        norm.Dispose();
    }

    [Fact]
    public void CheckpointConverter_Buckets_Encoder_And_Vae_Prefixes()
    {
        TensorShape shape = new TensorShape(16);
        Tensor t = new Tensor(shape, DType.F32);
        Dictionary<string, Tensor> input = new()
        {
            ["transformer.img_in.weight"] = t,
            ["text_encoder.model.embed_tokens.weight"] = t,
            ["vae.decoder.conv_in.weight"] = t,
        };

        LensCheckpointConverter.ConvertedWeights converted = LensCheckpointConverter.Convert(input);
        Assert.True(converted.Transformer.ContainsKey("img_in.weight"));
        Assert.True(converted.TextEncoder.ContainsKey("model.embed_tokens.weight"));
        Assert.True(converted.Vae.ContainsKey("decoder.conv_in.weight"));

        t.Dispose();
    }
}
