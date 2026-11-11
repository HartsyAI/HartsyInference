using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Dinov2;

namespace HartsyInference.ThreeD.Models.Hunyuan3D.Paint;

/// <summary>Paint 2.1's reference conditioning: DINOv2-giant at 224 px (the <c>AutoImageProcessor</c> defaults: shortest edge 256, centre crop 224, ImageNet
/// statistics), the last hidden state after the final layer norm, then <c>ImageProjModel</c>: each of the 257 tokens becomes four tokens through a linear layer
/// and a layer norm, giving 1028 tokens of width 1024.</summary>
public sealed class PaintDinoConditioner : IPaintConditioner
{
    private const int Crop = 224, ShortestEdge = 256, ExtraTokens = 4;
    private static readonly float[] Mean = [0.485f, 0.456f, 0.406f], Std = [0.229f, 0.224f, 0.225f];
    private readonly Dinov2VisionEncoder _dino;
    private readonly Tensor _projW, _projB, _normW, _normB;
    private readonly int _outputDim;

    /// <summary>Wraps a loaded DINOv2 encoder and the projection weights (<c>proj.weight</c> <c>[4 * dim, 1536]</c>, <c>proj.bias</c>, <c>norm.weight</c>, <c>norm.bias</c>).</summary>
    public PaintDinoConditioner(Dinov2VisionEncoder dino, Tensor projWeight, Tensor projBias, Tensor normWeight, Tensor normBias)
    {
        _dino = dino ?? throw new ArgumentNullException(nameof(dino));
        _projW = Hunyuan3DShapeVae.F32(projWeight);
        _projB = Hunyuan3DShapeVae.F32(projBias);
        _normW = Hunyuan3DShapeVae.F32(normWeight);
        _normB = Hunyuan3DShapeVae.F32(normBias);
        _outputDim = (int)_normW.Shape[0];
        if (_projW.Shape[0] != ExtraTokens * _outputDim)
            throw new ArgumentException("The projection must produce four tokens per DINO token.", nameof(projWeight));
    }

    /// <inheritdoc/>
    public unsafe Tensor Tokens(IBackend backend, ReadOnlySpan<byte> rgb, int width, int height)
    {
        byte[] resized = HunyuanPaintImages.Resize(rgb, width, height, ShortestEdge, ShortestEdge);
        int offset = (ShortestEdge - Crop) / 2;
        Tensor pixels = new(new TensorShape(1, 3, Crop, Crop), DType.F32);
        float* p = (float*)pixels.DataPointer;
        for (int y = 0; y < Crop; y++)
            for (int x = 0; x < Crop; x++)
                for (int c = 0; c < 3; c++)
                    p[(c * Crop + y) * Crop + x] = (resized[((y + offset) * ShortestEdge + x + offset) * 3 + c] / 255f - Mean[c]) / Std[c];
        Tensor[] last = _dino.EncodeIntermediates(backend, pixels, [_dino.Preset.NumLayers - 1]);
        pixels.Dispose();
        Tensor hidden = last[0];
        int tokens = (int)hidden.Shape[1];

        Tensor projected = new(new TensorShape(1, tokens, ExtraTokens * _outputDim), DType.F32);
        backend.Linear(projected, hidden, _projW, _projB);
        hidden.Dispose();
        Tensor normed = new(new TensorShape(1, tokens * ExtraTokens, _outputDim), DType.F32);
        backend.LayerNorm(normed, projected.Reshape(normed.Shape), _normW, _normB, 1e-5f);
        projected.Dispose();
        return normed;
    }
}
