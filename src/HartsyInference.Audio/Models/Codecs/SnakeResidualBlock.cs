using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Codecs;

/// <summary>The <c>Snake → dilated Conv1d → Snake → 1×1 Conv1d → residual add</c> unit that DAC and SNAC share.</summary>
internal static unsafe class SnakeResidualBlock
{
    /// <summary>Runs the unit over channels-first <c>[batch, dim, t]</c> and returns a fresh tensor; the input is
    /// NOT disposed. Padding is symmetric, not causal, and the residual is center-cropped to the conv output
    /// length because symmetric padding of an even receptive field rounds down.</summary>
    /// <param name="groups">Grouping of the dilated conv — the only place SNAC diverges from DAC.</param>
    public static Tensor Forward(
        IBackend backend,
        Tensor x,
        int batch,
        int t,
        int dim,
        int kernel,
        int dilation,
        int groups,
        Tensor snake1Alpha,
        Tensor conv1W,
        Tensor? conv1B,
        Tensor snake2Alpha,
        Tensor conv2W,
        Tensor? conv2B)
    {
        Tensor a1 = new(x.Shape, DType.F32);
        backend.Snake(a1, x, snake1Alpha, null);

        int pad = (kernel - 1) * dilation / 2;
        int tConv1 = t + 2 * pad - dilation * (kernel - 1);
        Tensor mid = new(new TensorShape(batch, dim, tConv1), DType.F32);
        backend.Conv1d(mid, a1, conv1W, conv1B,
            stride: 1, padLeft: pad, padRight: pad, dilation: dilation, groups: groups);
        a1.Dispose();

        Tensor a2 = new(mid.Shape, DType.F32);
        backend.Snake(a2, mid, snake2Alpha, null);
        mid.Dispose();

        Tensor proj = new(new TensorShape(batch, dim, tConv1), DType.F32);
        backend.Conv1d(proj, a2, conv2W, conv2B, stride: 1, padLeft: 0, padRight: 0, dilation: 1, groups: 1);
        a2.Dispose();

        int tProj = (int)proj.Shape[2];
        int diff = t - tProj;
        int cropLeft = diff / 2;
        Tensor result = new(proj.Shape, DType.F32);

        if (diff > 0)
        {
            float* xp = (float*)x.DataPointer;
            float* pp = (float*)proj.DataPointer;
            float* rp = (float*)result.DataPointer;
            for (int b = 0; b < batch; b++)
                for (int c = 0; c < dim; c++)
                {
                    int srcXBase = (b * dim + c) * t + cropLeft;
                    int srcPBase = (b * dim + c) * tProj;
                    int dstBase = (b * dim + c) * tProj;
                    for (int j = 0; j < tProj; j++) rp[dstBase + j] = xp[srcXBase + j] + pp[srcPBase + j];
                }
        }
        else
        {
            backend.Add(result, x, proj);
        }
        proj.Dispose();
        return result;
    }
}
