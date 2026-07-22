using System;
using System.Collections.Generic;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Pipelines;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Vision.Segmentation;

/// <summary>
/// CLIPSeg (rd64-refined) text-prompted image segmentation — the missing half of <c>&lt;segment:text&gt;</c>/<c>&lt;clear:text&gt;</c>.
/// A CLIP ViT-B/16 vision tower is run on the image with intermediate activations captured at layers [3,6,9]; a CLIP
/// text tower encodes the prompt into a 512-d conditional embedding; the CLIPSeg decoder fuses the three activations
/// (each linearly reduced 768→64), FiLM-conditions on the text embedding at the first decoder layer, runs three
/// post-norm transformer layers, then upsamples to a per-pixel mask via a 3-stage transposed-convolution head.
///
/// <para>Runs at the vision tower's native 224² (position-embeddings match, no interpolation). Weights are the
/// HuggingFace <c>CIDAS/clipseg-rd64-refined</c> layout under a <c>clip.</c> prefix + a <c>decoder.</c> block.</para>
/// </summary>
public sealed unsafe class ClipSegPipeline : IVisionPipeline, ITextSegmenter
{
    private const int ImageSize = 224;
    private const int VisionHidden = 768;
    private const int ReduceDim = 64;
    private const int ProjDim = 512;
    private const int DecoderHeads = 4;
    private const int DecoderFfn = 2048;
    private const float LnEps = 1e-5f;
    private static readonly int[] ExtractLayers = { 3, 6, 9 };
    // CLIPSeg's HF image processor uses ImageNet normalization (not CLIP's).
    private static readonly float[] Mean = { 0.485f, 0.456f, 0.406f };
    private static readonly float[] Std = { 0.229f, 0.224f, 0.225f };

    private readonly ClipVisionEncoder _vision;
    private readonly ClipTextEncoder _text;
    private readonly ClipTokenizer _tokenizer;

    private readonly Tensor[] _reduceW = new Tensor[3];
    private readonly Tensor[] _reduceB = new Tensor[3];
    private readonly Tensor _filmMulW, _filmMulB, _filmAddW, _filmAddB;
    private readonly DecoderLayerWeights[] _layers = new DecoderLayerWeights[3];
    private readonly Tensor _tconv0W, _tconv0B, _tconv2W, _tconv2B, _tconv4W, _tconv4B;

    public string ModelName => "clipseg-rd64-refined";

    private sealed class DecoderLayerWeights
    {
        public required Tensor QW, QB, KW, KB, VW, VB, OW, OB;
        public required Tensor Ln1W, Ln1B, Ln2W, Ln2B;
        public required Tensor Fc1W, Fc1B, Fc2W, Fc2B;
    }

    /// <summary>Loads a CLIPSeg model from its HuggingFace directory (containing <c>model.safetensors</c>).</summary>
    public ClipSegPipeline(string modelDir)
    {
        string modelPath = System.IO.Path.Combine(modelDir, "model.safetensors");
        if (!System.IO.File.Exists(modelPath))
        {
            throw new System.IO.FileNotFoundException($"CLIPSeg weights not found: {modelPath}");
        }

        Dictionary<string, Tensor> clip = new();
        Dictionary<string, Tensor> dec = new();
        using (SafeTensorsLoader loader = new SafeTensorsLoader())
        {
            loader.Load(modelPath);
            foreach (KeyValuePair<string, Tensor> kv in loader.GetAllTensors())
            {
                // HF's CLIPTextEmbeddings ships a `position_ids` I64 buffer (a derivable arange, not a learned
                // weight); ClipTextEncoder/ClipVisionEncoder never read it, and it isn't float-castable.
                if (kv.Key.EndsWith(".position_ids", StringComparison.Ordinal))
                {
                    continue;
                }
                // Copy each to an owned F32 tensor while the loader mmap is still alive (borrowed tensors dangle after Dispose).
                if (kv.Key.StartsWith("clip.", StringComparison.Ordinal))
                {
                    clip[kv.Key["clip.".Length..]] = OwnedF32(kv.Value);
                }
                else if (kv.Key.StartsWith("decoder.", StringComparison.Ordinal))
                {
                    dec[kv.Key] = OwnedF32(kv.Value);
                }
            }
        }

        _vision = new ClipVisionEncoder(new ClipVisionEncoderConfig
        {
            HiddenSize = VisionHidden, NumLayers = 12, NumHeads = 12, IntermediateSize = 3072,
            ImageSize = ImageSize, PatchSize = 16, ProjectionDim = ProjDim, UseQuickGelu = true, LayerNormEps = LnEps,
        });
        _vision.LoadWeights(clip, prefix: "vision_model");

        _text = new ClipTextEncoder(new ClipTextEncoderConfig
        {
            HiddenSize = ProjDim, IntermediateSize = 2048, NumLayers = 12, NumHeads = 8,
            MaxPositionEmbeddings = 77, VocabSize = 49408, UseQuickGelu = true, ProjectionDim = ProjDim, LayerNormEps = LnEps,
        });
        _text.LoadWeights(clip, prefix: "text_model");

        _tokenizer = new ClipTokenizer();

        for (int i = 0; i < 3; i++)
        {
            _reduceW[i] = dec[$"decoder.reduces.{i}.weight"];
            _reduceB[i] = dec[$"decoder.reduces.{i}.bias"];
            _layers[i] = new DecoderLayerWeights
            {
                QW = dec[$"decoder.layers.{i}.self_attn.q_proj.weight"], QB = dec[$"decoder.layers.{i}.self_attn.q_proj.bias"],
                KW = dec[$"decoder.layers.{i}.self_attn.k_proj.weight"], KB = dec[$"decoder.layers.{i}.self_attn.k_proj.bias"],
                VW = dec[$"decoder.layers.{i}.self_attn.v_proj.weight"], VB = dec[$"decoder.layers.{i}.self_attn.v_proj.bias"],
                OW = dec[$"decoder.layers.{i}.self_attn.out_proj.weight"], OB = dec[$"decoder.layers.{i}.self_attn.out_proj.bias"],
                Ln1W = dec[$"decoder.layers.{i}.layer_norm1.weight"], Ln1B = dec[$"decoder.layers.{i}.layer_norm1.bias"],
                Ln2W = dec[$"decoder.layers.{i}.layer_norm2.weight"], Ln2B = dec[$"decoder.layers.{i}.layer_norm2.bias"],
                Fc1W = dec[$"decoder.layers.{i}.mlp.fc1.weight"], Fc1B = dec[$"decoder.layers.{i}.mlp.fc1.bias"],
                Fc2W = dec[$"decoder.layers.{i}.mlp.fc2.weight"], Fc2B = dec[$"decoder.layers.{i}.mlp.fc2.bias"],
            };
        }
        _filmMulW = dec["decoder.film_mul.weight"]; _filmMulB = dec["decoder.film_mul.bias"];
        _filmAddW = dec["decoder.film_add.weight"]; _filmAddB = dec["decoder.film_add.bias"];
        // Head is Sequential[Conv2d(64,64,k3s1p1), ReLU, ConvTranspose2d(64,32,k4s4), ReLU, ConvTranspose2d(32,1,k4s4)].
        _tconv0W = dec["decoder.transposed_convolution.0.weight"]; _tconv0B = dec["decoder.transposed_convolution.0.bias"];
        _tconv2W = dec["decoder.transposed_convolution.2.weight"]; _tconv2B = dec["decoder.transposed_convolution.2.bias"];
        _tconv4W = dec["decoder.transposed_convolution.4.weight"]; _tconv4B = dec["decoder.transposed_convolution.4.bias"];
    }

    private static Tensor OwnedF32(Tensor t)
    {
        Tensor f32 = t.DType == DType.F32 ? t : t.CastTo(DType.F32);
        Tensor copy = new Tensor(f32.Shape, DType.F32);
        long bytes = f32.Shape.ElementCount * sizeof(float);
        Buffer.MemoryCopy((void*)f32.DataPointer, (void*)copy.DataPointer, bytes, bytes);
        if (!ReferenceEquals(f32, t))
        {
            f32.Dispose();
        }
        return copy;
    }

    /// <inheritdoc/>
    public float[] Segment(IBackend backend, Tensor image, string prompt, out int width, out int height)
    {
        width = ImageSize;
        height = ImageSize;

        Tensor pixels = Preprocess(image);
        Tensor[] acts = _vision.EncodeExtractLayers(backend, pixels, ExtractLayers); // 3 × [1, 197, 768]
        pixels.Dispose();

        Tensor cond = EncodeConditional(backend, prompt); // [1, 512]

        Tensor mask = Decode(backend, acts, cond); // [1, 1, 224, 224] in [0,1]
        foreach (Tensor a in acts)
        {
            a.Dispose();
        }
        cond.Dispose();

        float[] result = new float[ImageSize * ImageSize];
        mask.AsReadOnlySpan<float>().Slice(0, result.Length).CopyTo(result);
        mask.Dispose();
        return result;
    }

    private Tensor EncodeConditional(IBackend backend, string prompt)
    {
        int[] tokens = _tokenizer.Encode(prompt);
        int eos = ClipTokenizer.FindEosPosition(tokens);
        (Tensor hidden, Tensor? pooled) = _text.EncodePenultimate(backend, new[] { tokens }, new[] { eos }, layersFromEnd: 1);
        hidden.Dispose();
        if (pooled is null)
        {
            throw new InvalidOperationException("CLIPSeg text encoder produced no pooled output — text_projection weight missing.");
        }
        return pooled; // [1, 512]
    }

    private Tensor Decode(IBackend backend, Tensor[] acts, Tensor cond)
    {
        int seq = (int)acts[0].Shape[1]; // 197
        TensorShape rShape = new TensorShape(1, seq, ReduceDim);

        Tensor filmMul = new Tensor(new TensorShape(1, ReduceDim), DType.F32);
        backend.Linear(filmMul, cond, _filmMulW, _filmMulB);
        Tensor filmAdd = new Tensor(new TensorShape(1, ReduceDim), DType.F32);
        backend.Linear(filmAdd, cond, _filmAddW, _filmAddB);

        // activations reversed: reduces[0]→layer9, [1]→layer6, [2]→layer3.
        Tensor output = null!;
        for (int i = 0; i < 3; i++)
        {
            Tensor act = acts[2 - i];
            Tensor reduced = new Tensor(rShape, DType.F32);
            backend.Linear(reduced, act, _reduceW[i], _reduceB[i]);
            if (output is null)
            {
                output = reduced;
            }
            else
            {
                backend.Add(output, output, reduced);
                reduced.Dispose();
            }
            if (i == 0)
            {
                backend.AffineBroadcastLastDim(output, output, filmMul, filmAdd); // output = filmMul*output + filmAdd
            }
            Tensor layered = DecoderLayer(backend, output, _layers[i], seq);
            output.Dispose();
            output = layered;
        }
        filmMul.Dispose();
        filmAdd.Dispose();

        int patches = seq - 1;
        int grid = (int)MathF.Round(MathF.Sqrt(patches)); // 14
        Tensor noCls = new Tensor(new TensorShape(1, patches, ReduceDim), DType.F32);
        CopyDropFirst(output, noCls, ReduceDim);
        output.Dispose();

        Tensor chanFirst = new Tensor(new TensorShape(1, ReduceDim, patches), DType.F32);
        backend.Transpose2D(chanFirst, noCls, patches, ReduceDim);
        noCls.Dispose();
        // Materialize a real [1,64,grid,grid] tensor (a reshape *view* of chanFirst confuses ConvTranspose2d).
        Tensor spatial = new Tensor(new TensorShape(1, ReduceDim, grid, grid), DType.F32);
        long spBytes = spatial.Shape.ElementCount * sizeof(float);
        Buffer.MemoryCopy((void*)chanFirst.DataPointer, (void*)spatial.DataPointer, spBytes, spBytes);

        // Head layer 0 is a REGULAR Conv2d(64,64,k3,s1,p1) — NOT a transposed conv (only sc[2]/sc[4] are).
        // Weight is [Cout=64, Cin=64, 3, 3]; ambiguous with the transposed [Cin,Cout] convention because
        // Cin==Cout, which is why the wrong op silently ran on correctly-loaded weights.
        Tensor t0 = new Tensor(new TensorShape(1, ReduceDim, grid, grid), DType.F32);
        backend.Conv2D(t0, spatial, _tconv0W, _tconv0B, 1, 1, 1, 1);
        chanFirst.Dispose();
        backend.LeakyRelu(t0, t0, 0f);

        int g2 = grid * 4;
        Tensor t2 = new Tensor(new TensorShape(1, ReduceDim / 2, g2, g2), DType.F32);
        backend.ConvTranspose2d(t2, t0, _tconv2W, _tconv2B, 4, 4, 0, 0);
        t0.Dispose();
        backend.LeakyRelu(t2, t2, 0f);

        int g3 = g2 * 4; // 224
        Tensor logits = new Tensor(new TensorShape(1, 1, g3, g3), DType.F32);
        backend.ConvTranspose2d(logits, t2, _tconv4W, _tconv4B, 4, 4, 0, 0);
        t2.Dispose();

        Tensor mask = new Tensor(new TensorShape(1, 1, g3, g3), DType.F32);
        backend.Sigmoid(mask, logits);
        logits.Dispose();
        return mask;
    }

    private Tensor DecoderLayer(IBackend backend, Tensor x, DecoderLayerWeights w, int seq)
    {
        TensorShape shape = new TensorShape(1, seq, ReduceDim);
        int headDim = ReduceDim / DecoderHeads; // 16

        Tensor q = new Tensor(shape, DType.F32); backend.Linear(q, x, w.QW, w.QB);
        Tensor k = new Tensor(shape, DType.F32); backend.Linear(k, x, w.KW, w.KB);
        Tensor v = new Tensor(shape, DType.F32); backend.Linear(v, x, w.VW, w.VB);

        TensorShape mh = new TensorShape(1, DecoderHeads, seq, headDim);
        Tensor qMh = new Tensor(mh, DType.F32); backend.Permute0213(qMh, q, seq, DecoderHeads, headDim); q.Dispose();
        Tensor kMh = new Tensor(mh, DType.F32); backend.Permute0213(kMh, k, seq, DecoderHeads, headDim); k.Dispose();
        Tensor vMh = new Tensor(mh, DType.F32); backend.Permute0213(vMh, v, seq, DecoderHeads, headDim); v.Dispose();

        Tensor attn = new Tensor(mh, DType.F32);
        backend.ScaledDotProductAttention(attn, qMh, kMh, vMh, null, 1.0f / MathF.Sqrt(headDim));
        qMh.Dispose(); kMh.Dispose(); vMh.Dispose();

        Tensor attnFlat = new Tensor(shape, DType.F32);
        backend.Permute0213(attnFlat, attn, DecoderHeads, seq, headDim);
        attn.Dispose();
        Tensor attnOut = new Tensor(shape, DType.F32);
        backend.Linear(attnOut, attnFlat, w.OW, w.OB);
        attnFlat.Dispose();

        Tensor afterAttn = new Tensor(shape, DType.F32);
        backend.Add(afterAttn, x, attnOut);
        attnOut.Dispose();
        Tensor normed1 = new Tensor(shape, DType.F32);
        backend.LayerNorm(normed1, afterAttn, w.Ln1W, w.Ln1B, LnEps);
        afterAttn.Dispose();

        Tensor fc1 = new Tensor(new TensorShape(1, seq, DecoderFfn), DType.F32);
        backend.Linear(fc1, normed1, w.Fc1W, w.Fc1B);
        backend.LeakyRelu(fc1, fc1, 0f); // CLIPSeg's decoder MLP uses ReLU (HF overrides config's quick_gelu)
        Tensor fc2 = new Tensor(shape, DType.F32);
        backend.Linear(fc2, fc1, w.Fc2W, w.Fc2B);
        fc1.Dispose();

        Tensor afterMlp = new Tensor(shape, DType.F32);
        backend.Add(afterMlp, normed1, fc2);
        fc2.Dispose();
        normed1.Dispose();
        Tensor normed2 = new Tensor(shape, DType.F32);
        backend.LayerNorm(normed2, afterMlp, w.Ln2W, w.Ln2B, LnEps);
        afterMlp.Dispose();
        return normed2;
    }

    /// <summary>quick_gelu in place: x * sigmoid(1.702 * x).</summary>
    private static void QuickGelu(IBackend backend, Tensor x)
    {
        Tensor scaled = new Tensor(x.Shape, DType.F32);
        backend.Scale(scaled, x, 1.702f);
        backend.Sigmoid(scaled, scaled);
        backend.Mul(x, x, scaled);
        scaled.Dispose();
    }

    private static void CopyDropFirst(Tensor src, Tensor dst, int dim)
    {
        float* s = (float*)src.DataPointer;
        float* d = (float*)dst.DataPointer;
        long bytes = dst.Shape.ElementCount * sizeof(float);
        Buffer.MemoryCopy(s + dim, d, bytes, bytes); // skip position 0 (CLS)
    }

    /// <summary>Resizes <paramref name="image"/> ([1,3,H,W], RGB in [0,1]) to [1,3,224,224] with ImageNet normalization.</summary>
    private static Tensor Preprocess(Tensor image)
    {
        int srcC = (int)image.Shape[1];
        int srcH = (int)image.Shape[2];
        int srcW = (int)image.Shape[3];
        Tensor outT = new Tensor(new TensorShape(1, 3, ImageSize, ImageSize), DType.F32);
        float* src = (float*)image.DataPointer;
        float* dst = (float*)outT.DataPointer;
        for (int c = 0; c < 3; c++)
        {
            int sc = Math.Min(c, srcC - 1);
            for (int y = 0; y < ImageSize; y++)
            {
                float fy = (y + 0.5f) * srcH / ImageSize - 0.5f;
                int y0 = Math.Clamp((int)MathF.Floor(fy), 0, srcH - 1);
                int y1 = Math.Min(y0 + 1, srcH - 1);
                float wy = MathF.Max(0f, fy - y0);
                for (int x = 0; x < ImageSize; x++)
                {
                    float fx = (x + 0.5f) * srcW / ImageSize - 0.5f;
                    int x0 = Math.Clamp((int)MathF.Floor(fx), 0, srcW - 1);
                    int x1 = Math.Min(x0 + 1, srcW - 1);
                    float wx = MathF.Max(0f, fx - x0);
                    float p00 = src[(sc * srcH + y0) * srcW + x0];
                    float p01 = src[(sc * srcH + y0) * srcW + x1];
                    float p10 = src[(sc * srcH + y1) * srcW + x0];
                    float p11 = src[(sc * srcH + y1) * srcW + x1];
                    float top = p00 + (p01 - p00) * wx;
                    float bot = p10 + (p11 - p10) * wx;
                    float val = top + (bot - top) * wy;
                    dst[(c * ImageSize + y) * ImageSize + x] = (val - Mean[c]) / Std[c];
                }
            }
        }
        return outT;
    }

    public void Dispose()
    {
        foreach (Tensor t in _reduceW)
        {
            t?.Dispose();
        }
        foreach (Tensor t in _reduceB)
        {
            t?.Dispose();
        }
        GC.SuppressFinalize(this);
    }
}
