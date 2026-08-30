using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Builds the published 49-channel Fun input rows in the order control24, visibility1, masked-source24.</summary>
public static unsafe class MiniMaxH3FunControlInputBuilder
{
    private const int LatentChannels = 24;
    private const int VisibilityChannels = 1;
    private const int ControlChannels = LatentChannels + VisibilityChannels + LatentChannels;

    /// <summary>Patchifies one pure-control or inpaint stream, zero-padding channels 24..48 for pure control.</summary>
    public static Tensor Build(Tensor control, Tensor? visibility, Tensor? maskedSource,
        int patchT = 1, int patchH = 2, int patchW = 2)
    {
        ValidateLatent(control, LatentChannels, "control");
        bool inpaint = visibility is not null || maskedSource is not null;
        if (inpaint && (visibility is null || maskedSource is null))
        {
            throw new HartsyInferenceException(
                "MiniMax-H3 Fun inpaint input requires both visibility and masked-source latents.");
        }
        if (visibility is not null)
        {
            ValidateLatent(visibility, VisibilityChannels, "visibility");
            ValidateLatent(maskedSource!, LatentChannels, "masked source");
            RequireSameGeometry(control, visibility, "visibility");
            RequireSameGeometry(control, maskedSource!, "masked source");
        }
        if (patchT <= 0 || patchH <= 0 || patchW <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(patchT), "Control patch dimensions must be positive.");
        }

        int sourceT = checked((int)control.Shape[2]);
        int sourceH = checked((int)control.Shape[3]);
        int sourceW = checked((int)control.Shape[4]);
        int targetT = DivideRoundUp(sourceT, patchT);
        int targetH = DivideRoundUp(sourceH, patchH);
        int targetW = DivideRoundUp(sourceW, patchW);
        int rowWidth = ControlChannels * patchT * patchH * patchW;
        Tensor rows = new Tensor(new TensorShape((long)targetT * targetH * targetW, rowWidth), DType.F32);
        float* controlPointer = (float*)control.DataPointer;
        float* visibilityPointer = visibility is null ? null : (float*)visibility.DataPointer;
        float* sourcePointer = maskedSource is null ? null : (float*)maskedSource.DataPointer;
        float* output = (float*)rows.DataPointer;

        for (int t = 0; t < targetT; t++)
        {
            for (int y = 0; y < targetH; y++)
            {
                for (int x = 0; x < targetW; x++)
                {
                    float* row = output + ((long)(t * targetH + y) * targetW + x) * rowWidth;
                    for (int channel = 0; channel < ControlChannels; channel++)
                    {
                        for (int rt = 0; rt < patchT; rt++)
                        {
                            for (int ry = 0; ry < patchH; ry++)
                            {
                                for (int rx = 0; rx < patchW; rx++)
                                {
                                    int sourceFrame = (t * patchT + rt) % sourceT;
                                    int sourceY = (y * patchH + ry) % sourceH;
                                    int sourceX = (x * patchW + rx) % sourceW;
                                    float value = ReadChannel(channel, sourceFrame, sourceY, sourceX,
                                        sourceT, sourceH, sourceW, controlPointer, visibilityPointer, sourcePointer);
                                    int offset = ((channel * patchT + rt) * patchH + ry) * patchW + rx;
                                    row[offset] = value;
                                }
                            }
                        }
                    }
                }
            }
        }
        return rows;
    }

    private static float ReadChannel(int channel, int frame, int y, int x, int frames, int height, int width,
        float* control, float* visibility, float* source)
    {
        long spatial = (long)height * width;
        if (channel < LatentChannels)
        {
            return control[((long)channel * frames + frame) * spatial + (long)y * width + x];
        }
        if (channel == LatentChannels)
        {
            return visibility == null ? 0f
                : Math.Clamp(visibility[(long)frame * spatial + (long)y * width + x], 0f, 1f);
        }
        return source == null ? 0f
            : source[((long)(channel - LatentChannels - 1) * frames + frame) * spatial + (long)y * width + x];
    }

    private static void ValidateLatent(Tensor tensor, int channels, string label)
    {
        ArgumentNullException.ThrowIfNull(tensor);
        if (tensor.DType != DType.F32 || tensor.Shape.Rank != 5 || tensor.Shape[0] != 1
            || tensor.Shape[1] != channels || tensor.Shape[2] <= 0 || tensor.Shape[3] <= 0 || tensor.Shape[4] <= 0)
        {
            throw new HartsyInferenceException(
                $"MiniMax-H3 Fun {label} must be F32 [1,{channels},T,H,W], got {tensor.DType} {tensor.Shape}.");
        }
    }

    private static void RequireSameGeometry(Tensor control, Tensor other, string label)
    {
        if (control.Shape[2] != other.Shape[2] || control.Shape[3] != other.Shape[3]
            || control.Shape[4] != other.Shape[4])
        {
            throw new HartsyInferenceException(
                $"MiniMax-H3 Fun {label} geometry {other.Shape} does not match control {control.Shape}.");
        }
    }

    private static int DivideRoundUp(int value, int divisor) => checked((value + divisor - 1) / divisor);
}
