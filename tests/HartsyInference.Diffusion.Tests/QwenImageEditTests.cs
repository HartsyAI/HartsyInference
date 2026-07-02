using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Models.Vae.QwenImage;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Wiring tests for the Qwen-Image-Edit reference-latent path (<c>editRefImage</c> on
/// <see cref="QwenImagePipeline.GenerateFromTokens"/>): no-encoder throws, bad-ref-shape throws, plus a
/// <see cref="QwenImageRope.ApplyJoint"/> unit test proving the appended ref-token rows get their own RoPE section
/// (frame axis = ref index, spatial centered on the REF grid) while the txt/img rows stay bit-identical to the
/// no-ref path. End-to-end edit quality needs the real Qwen-Image-Edit checkpoint (and the Qwen2.5-VL vision
/// encoder for the VL-token half, which is deferred).</summary>
public sealed unsafe class QwenImageEditTests
{
    private static QwenImagePipeline MakePipeline(CpuBackend backend, bool withEncoder)
    {
        LlamaStyleEncoder textEncoder = new(LlamaStyleEncoderConfig.Qwen2_5_VL_7B);
        QwenImageTransformer transformer = new(QwenImageConfig.V1);
        QwenImageVaeDecoder vaeDecoder = new(VaeConfig.QwenImage);
        return withEncoder
            ? new QwenImagePipeline(backend, textEncoder, transformer, vaeDecoder, new QwenImageVaeEncoder(VaeConfig.QwenImage), QwenImageConfig.V1)
            : new QwenImagePipeline(backend, textEncoder, transformer, vaeDecoder, QwenImageConfig.V1);
    }

    [Fact]
    public void GenerateFromTokens_EditRefWithoutVaeEncoder_Throws()
    {
        using CpuBackend backend = new();
        using QwenImagePipeline pipeline = MakePipeline(backend, withEncoder: false);

        Tensor refImage = new Tensor(new TensorShape(1, 3, 64, 64), DType.F32);
        TextToImageRequest request = new()
        {
            Prompt = "make the sky purple",
            Width = 64,
            Height = 64,
        };

        Assert.Throws<InvalidOperationException>(() =>
            pipeline.GenerateFromTokens(promptTokenIds: [0], negativeTokenIds: [0], request, editRefImage: refImage));

        refImage.Dispose();
    }

    [Fact]
    public void GenerateFromTokens_EditRefWrongShape_Throws()
    {
        using CpuBackend backend = new();
        using QwenImagePipeline pipeline = MakePipeline(backend, withEncoder: true);

        // 40 is not divisible by 16 (VAE 8x + 2x2 packing), so the ref cannot be packed.
        Tensor refImage = new Tensor(new TensorShape(1, 3, 40, 40), DType.F32);
        TextToImageRequest request = new()
        {
            Prompt = "make the sky purple",
            Width = 64,
            Height = 64,
        };

        Assert.Throws<ArgumentException>(() =>
            pipeline.GenerateFromTokens(promptTokenIds: [0], negativeTokenIds: [0], request, editRefImage: refImage));

        refImage.Dispose();
    }

    [Fact]
    public void GenerateFromTokens_EditRefWrongRank_Throws()
    {
        using CpuBackend backend = new();
        using QwenImagePipeline pipeline = MakePipeline(backend, withEncoder: true);

        Tensor refImage = new Tensor(new TensorShape(3, 64, 64), DType.F32);
        TextToImageRequest request = new()
        {
            Prompt = "make the sky purple",
            Width = 64,
            Height = 64,
        };

        Assert.Throws<ArgumentException>(() =>
            pipeline.GenerateFromTokens(promptTokenIds: [0], negativeTokenIds: [0], request, editRefImage: refImage));

        refImage.Dispose();
    }

    /// <summary>The joint RoPE with a trailing ref section must (a) leave the txt+img rows bit-identical to the
    /// no-ref call, and (b) rotate a ref row exactly like the corresponding MAIN image row when the ref grid matches
    /// the main grid and the ref frame index is forced to 0 — the ref section differs from the image section only by
    /// its frame-axis position.</summary>
    [Fact]
    public void ApplyJoint_RefSection_MatchesImageRowsAtFrame0_AndLeavesMainRowsUnchanged()
    {
        const int batch = 1, heads = 2, headDim = 128, hPacked = 4, wPacked = 4, txtSeq = 3;
        int imgSeq = hPacked * wPacked;
        int total = txtSeq + imgSeq + imgSeq;   // txt + main + ref (same grid)
        int posStart = QwenImageRope.ComputeTextPositionStart(hPacked, wPacked);
        QwenImageRope rope = new();

        static void Fill(Tensor t, int seed)
        {
            Span<float> s = t.AsSpan<float>();
            uint x = (uint)(seed * 2654435761u + 1);
            for (int i = 0; i < s.Length; i++) { x ^= x << 13; x ^= x >> 17; x ^= x << 5; s[i] = ((x >> 8) / (float)(1 << 24)) * 2f - 1f; }
        }

        // Joint tensor with the REF rows duplicating the MAIN image rows (same pre-rotation values).
        TensorShape jointShape = new(batch, heads, total, headDim);
        Tensor q = new(jointShape, DType.F32);
        Tensor k = new(jointShape, DType.F32);
        Fill(q, 1);
        Fill(k, 2);
        for (int h = 0; h < heads; h++)
        {
            long rowFloats = headDim;
            for (int s = 0; s < imgSeq; s++)
            {
                long mainOff = ((long)h * total + txtSeq + s) * rowFloats;
                long refOff = ((long)h * total + txtSeq + imgSeq + s) * rowFloats;
                for (int d = 0; d < headDim; d++)
                {
                    q.AsSpan<float>()[(int)(refOff + d)] = q.AsSpan<float>()[(int)(mainOff + d)];
                    k.AsSpan<float>()[(int)(refOff + d)] = k.AsSpan<float>()[(int)(mainOff + d)];
                }
            }
        }

        // Reference: the no-ref joint rope over the same txt+img prefix.
        TensorShape prefixShape = new(batch, heads, txtSeq + imgSeq, headDim);
        Tensor qPrefix = new(prefixShape, DType.F32);
        Tensor kPrefix = new(prefixShape, DType.F32);
        for (int h = 0; h < heads; h++)
        {
            for (int s = 0; s < txtSeq + imgSeq; s++)
            {
                long srcOff = ((long)h * total + s) * headDim;
                long dstOff = ((long)h * (txtSeq + imgSeq) + s) * headDim;
                for (int d = 0; d < headDim; d++)
                {
                    qPrefix.AsSpan<float>()[(int)(dstOff + d)] = q.AsSpan<float>()[(int)(srcOff + d)];
                    kPrefix.AsSpan<float>()[(int)(dstOff + d)] = k.AsSpan<float>()[(int)(srcOff + d)];
                }
            }
        }
        rope.ApplyJoint(qPrefix, kPrefix, batch, heads, hPacked, wPacked, txtSeq, posStart);

        // Frame index 0 + identical grid + identical input rows ⇒ ref rows must rotate exactly like main rows.
        rope.ApplyJoint(q, k, batch, heads, hPacked, wPacked, txtSeq, posStart,
            refPackedH: hPacked, refPackedW: wPacked, refFrameIndex: 0);

        ReadOnlySpan<float> qs = q.AsReadOnlySpan<float>();
        ReadOnlySpan<float> qp = qPrefix.AsReadOnlySpan<float>();
        for (int h = 0; h < heads; h++)
        {
            for (int s = 0; s < txtSeq + imgSeq; s++)
            {
                for (int d = 0; d < headDim; d++)
                {
                    int joint = (int)(((long)h * total + s) * headDim + d);
                    int prefix = (int)(((long)h * (txtSeq + imgSeq) + s) * headDim + d);
                    Assert.Equal(qp[prefix], qs[joint]);
                }
            }
            for (int s = 0; s < imgSeq; s++)
            {
                for (int d = 0; d < headDim; d++)
                {
                    int refIdx = (int)(((long)h * total + txtSeq + imgSeq + s) * headDim + d);
                    int mainIdx = (int)(((long)h * total + txtSeq + s) * headDim + d);
                    Assert.Equal(qs[mainIdx], qs[refIdx]);
                }
            }
        }

        // Frame index 1 must CHANGE the ref rows (the frame sub-band rotates) while main rows stay put.
        Tensor q2 = new(jointShape, DType.F32);
        Tensor k2 = new(jointShape, DType.F32);
        Fill(q2, 1);
        Fill(k2, 2);
        for (int h = 0; h < heads; h++)
        {
            for (int s = 0; s < imgSeq; s++)
            {
                long mainOff = ((long)h * total + txtSeq + s) * headDim;
                long refOff = ((long)h * total + txtSeq + imgSeq + s) * headDim;
                for (int d = 0; d < headDim; d++)
                {
                    q2.AsSpan<float>()[(int)(refOff + d)] = q2.AsSpan<float>()[(int)(mainOff + d)];
                    k2.AsSpan<float>()[(int)(refOff + d)] = k2.AsSpan<float>()[(int)(mainOff + d)];
                }
            }
        }
        rope.ApplyJoint(q2, k2, batch, heads, hPacked, wPacked, txtSeq, posStart,
            refPackedH: hPacked, refPackedW: wPacked, refFrameIndex: 1);

        bool refDiffers = false;
        ReadOnlySpan<float> q2s = q2.AsReadOnlySpan<float>();
        for (int s = 0; s < imgSeq && !refDiffers; s++)
        {
            for (int d = 0; d < headDim; d++)
            {
                int refIdx = (int)(((long)0 * total + txtSeq + imgSeq + s) * headDim + d);
                int mainIdx = (int)(((long)0 * total + txtSeq + s) * headDim + d);
                if (q2s[refIdx] != q2s[mainIdx]) { refDiffers = true; break; }
            }
        }
        Assert.True(refDiffers, "frame index 1 should rotate the ref rows' frame sub-band differently from the main rows");

        q.Dispose(); k.Dispose();
        qPrefix.Dispose(); kPrefix.Dispose();
        q2.Dispose(); k2.Dispose();
    }
}
