using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Dinov2;

namespace HartsyInference.Vision.Annotators;

/// <summary>ControlNetHED soft-edge detector (<c>ControlNetHED_Apache2</c> from comfyui_controlnet_aux /
/// the ControlNet-v1-1 annotators): a VGG16-style trunk of 5 double/triple-conv blocks with 2×2 max-pool
/// downsampling between blocks, each block emitting a 1-channel 1×1 side projection. The five side logits
/// are bilinearly resized to the input resolution, averaged, and sigmoided into the [0,1] soft edge map
/// the softedge_hed / scribble_hed ControlNets expect. Input is raw [0,255] RGB minus a learned
/// per-channel shift — see <see cref="HedPreprocessor"/>.</summary>
public sealed unsafe class HedModel
{
    private static readonly int[] BlockChannels = [64, 128, 256, 512, 512];
    private static readonly int[] BlockDepth = [2, 2, 3, 3, 3];

    private readonly Tensor?[][] _convW = new Tensor?[5][];
    private readonly Tensor?[][] _convB = new Tensor?[5][];
    private readonly Tensor?[] _projW = new Tensor?[5];
    private readonly Tensor?[] _projB = new Tensor?[5];
    private readonly float[] _norm = new float[3];

    public HedModel()
    {
        for (int b = 0; b < 5; b++)
        {
            _convW[b] = new Tensor?[BlockDepth[b]];
            _convB[b] = new Tensor?[BlockDepth[b]];
        }
    }

    /// <summary>Loads the official <c>ControlNetHED.pth</c> state dict (keys <c>norm</c>,
    /// <c>block{1..5}.convs.{i}.weight/bias</c>, <c>block{1..5}.projection.weight/bias</c>).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        Tensor norm = Dinov2VisionEncoder.F32(w["norm"]);
        if (norm.ElementCount != 3)
            throw new HartsyInferenceException($"HED norm parameter must have 3 elements; got {norm.Shape}.");
        float* np = (float*)norm.DataPointer;
        for (int c = 0; c < 3; c++) _norm[c] = np[c];

        for (int b = 0; b < 5; b++)
        {
            for (int i = 0; i < BlockDepth[b]; i++)
            {
                _convW[b][i] = Dinov2VisionEncoder.F32(w[$"block{b + 1}.convs.{i}.weight"]);
                _convB[b][i] = Dinov2VisionEncoder.F32(w[$"block{b + 1}.convs.{i}.bias"]);
            }
            _projW[b] = Dinov2VisionEncoder.F32(w[$"block{b + 1}.projection.weight"]);
            _projB[b] = Dinov2VisionEncoder.F32(w[$"block{b + 1}.projection.bias"]);
        }
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        for (int b = 0; b < 5; b++)
        {
            for (int i = 0; i < BlockDepth[b]; i++)
            {
                if (_convW[b][i] is not null) yield return _convW[b][i]!;
                if (_convB[b][i] is not null) yield return _convB[b][i]!;
            }
            if (_projW[b] is not null) yield return _projW[b]!;
            if (_projB[b] is not null) yield return _projB[b]!;
        }
    }

    /// <summary>Computes the fused soft edge map <c>[1,1,H,W]</c> in [0,1] for raw [0,255] RGB pixels
    /// <c>[1,3,H,W]</c> (from <see cref="Codec.ImageTensor.RgbToTensor255"/>). The learned input shift is
    /// applied host-side before the first backend op — the input tensor is host-resident by contract.
    /// <paramref name="tap"/> receives the five side logits (<c>proj_1..proj_5</c>) for the parity harness.</summary>
    public Tensor Forward(IBackend backend, Tensor pixelValues, Action<string, Tensor>? tap = null)
    {
        if (pixelValues.Shape.Rank != 4 || pixelValues.Shape[0] != 1 || pixelValues.Shape[1] != 3)
            throw new ArgumentException($"HED input must be [1,3,H,W]; got {pixelValues.Shape}.", nameof(pixelValues));
        int fullH = (int)pixelValues.Shape[2];
        int fullW = (int)pixelValues.Shape[3];

        Tensor h = new(pixelValues.Shape, DType.F32);
        float* src = (float*)pixelValues.DataPointer;
        float* dst = (float*)h.DataPointer;
        long plane = (long)fullH * fullW;
        for (int c = 0; c < 3; c++)
        {
            float shift = _norm[c];
            for (long i = 0; i < plane; i++) dst[c * plane + i] = src[c * plane + i] - shift;
        }

        Tensor[] projections = new Tensor[5];
        int curH = fullH, curW = fullW, curC = 3;
        for (int b = 0; b < 5; b++)
        {
            if (b > 0)
            {
                int pooledH = curH / 2, pooledW = curW / 2;
                Tensor pooled = new(new TensorShape(1, curC, pooledH, pooledW), DType.F32);
                backend.MaxPool2D(pooled, h, 2, 2, 2, 2, 0, 0);
                h.Dispose();
                h = pooled;
                curH = pooledH; curW = pooledW;
            }
            for (int i = 0; i < BlockDepth[b]; i++)
            {
                Tensor conv = new(new TensorShape(1, BlockChannels[b], curH, curW), DType.F32);
                backend.Conv2D(conv, h, _convW[b][i]!, _convB[b][i]!, 1, 1, 1, 1);
                h.Dispose();
                Tensor relu = new(conv.Shape, DType.F32);
                backend.LeakyRelu(relu, conv, 0f);
                conv.Dispose();
                h = relu;
            }
            curC = BlockChannels[b];
            projections[b] = new Tensor(new TensorShape(1, 1, curH, curW), DType.F32);
            backend.Conv2D(projections[b], h, _projW[b]!, _projB[b]!, 1, 1, 0, 0);
            tap?.Invoke($"proj_{b + 1}", projections[b]);
        }
        h.Dispose();

        // Fuse: bilinear (half-pixel, matching cv2.INTER_LINEAR) resize of every side logit to the input
        // resolution, mean over the five scales, sigmoid.
        Tensor? acc = null;
        for (int b = 0; b < 5; b++)
        {
            Tensor up = new(new TensorShape(1, 1, fullH, fullW), DType.F32);
            backend.InterpolateBilinear2D(up, projections[b], alignCorners: false);
            projections[b].Dispose();
            if (acc is null)
            {
                acc = up;
            }
            else
            {
                Tensor sum = new(up.Shape, DType.F32);
                backend.Add(sum, acc, up);
                acc.Dispose();
                up.Dispose();
                acc = sum;
            }
        }
        Tensor mean = new(acc!.Shape, DType.F32);
        backend.Scale(mean, acc, 1f / 5f);
        acc.Dispose();
        Tensor edge = new(mean.Shape, DType.F32);
        backend.Sigmoid(edge, mean);
        mean.Dispose();
        return edge;
    }
}
