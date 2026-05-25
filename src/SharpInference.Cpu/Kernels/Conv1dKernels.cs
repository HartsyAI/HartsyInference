using SharpInference.Core.Tensors;

namespace SharpInference.Cpu.Kernels;

/// <summary>CPU compute kernels for 1D convolution and transposed convolution. F32
/// scalar implementations — SIMD vectorization can wait until profiling shows the
/// codec hot path is here. At the sizes these run for typical audio codecs
/// (channels &lt;= 2048, T &lt;= a few thousand frames per chunk) scalar is in the
/// low-milliseconds range.
///
/// <para>Layout assumptions:</para>
/// <list type="bullet">
///   <item>input / output: channels-first <c>[B, C, T]</c></item>
///   <item>Conv1d weight: <c>[C_out, C_in / groups, K]</c> (PyTorch nn.Conv1d)</item>
///   <item>ConvTranspose1d weight: <c>[C_in, C_out, K]</c> (PyTorch nn.ConvTranspose1d)</item>
///   <item>bias: <c>[C_out]</c> (optional)</item>
/// </list>
///
/// <para>Padding is exposed as separate <c>padLeft</c> / <c>padRight</c> values rather
/// than a single symmetric value — most audio codecs need asymmetric (causal) padding,
/// and giving the caller control here saves a kludge layer in every codec.</para></summary>
public static class Conv1dKernels
{
    /// <summary>1D convolution. Output shape must be
    /// <c>[B, C_out, (T_in + padLeft + padRight - dilation*(K-1) - 1) / stride + 1]</c>.
    /// Caller pre-allocates the output tensor.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void Conv1d(
        Tensor output, Tensor input, Tensor weight, Tensor? bias,
        int stride, int padLeft, int padRight, int dilation, int groups)
    {
        if (input.Shape.Rank != 3) throw new ArgumentException($"input must be rank-3, got {input.Shape}.");
        if (weight.Shape.Rank != 3) throw new ArgumentException($"weight must be rank-3, got {weight.Shape}.");
        if (output.Shape.Rank != 3) throw new ArgumentException($"output must be rank-3, got {output.Shape}.");

        int batch = (int)input.Shape[0];
        int cIn = (int)input.Shape[1];
        int tIn = (int)input.Shape[2];
        int cOut = (int)weight.Shape[0];
        int weightCInPerGroup = (int)weight.Shape[1];
        int kernel = (int)weight.Shape[2];

        if (groups <= 0) throw new ArgumentException($"groups must be positive, got {groups}.");
        if (cIn % groups != 0 || cOut % groups != 0)
            throw new ArgumentException($"channels must divide evenly by groups (cIn={cIn}, cOut={cOut}, groups={groups}).");
        if (weightCInPerGroup != cIn / groups)
            throw new ArgumentException($"weight dim 1 ({weightCInPerGroup}) must equal cIn/groups ({cIn / groups}).");

        int inPerGroup = cIn / groups;
        int outPerGroup = cOut / groups;
        int tOut = (tIn + padLeft + padRight - dilation * (kernel - 1) - 1) / stride + 1;
        if ((int)output.Shape[2] != tOut || (int)output.Shape[1] != cOut || (int)output.Shape[0] != batch)
            throw new ArgumentException($"output shape mismatch: expected [{batch}, {cOut}, {tOut}], got {output.Shape}.");

        float* ip = (float*)input.DataPointer;
        float* op = (float*)output.DataPointer;
        float* wp = (float*)weight.DataPointer;
        float* bp = bias is null ? null : (float*)bias.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int oc = 0; oc < cOut; oc++)
            {
                int group = oc / outPerGroup;
                int icStart = group * inPerGroup;
                float biasV = bp is null ? 0f : bp[oc];
                int wRow = oc * inPerGroup * kernel;
                int outBase = (b * cOut + oc) * tOut;

                for (int j = 0; j < tOut; j++)
                {
                    int srcLeftmost = j * stride - padLeft;
                    float acc = biasV;
                    for (int ic = 0; ic < inPerGroup; ic++)
                    {
                        int inCh = icStart + ic;
                        int inBase = (b * cIn + inCh) * tIn;
                        int wBase = wRow + ic * kernel;
                        for (int k = 0; k < kernel; k++)
                        {
                            int src = srcLeftmost + k * dilation;
                            if ((uint)src < (uint)tIn)
                                acc += ip[inBase + src] * wp[wBase + k];
                        }
                    }
                    op[outBase + j] = acc;
                }
            }
        }
    }

    /// <summary>1D transposed convolution. PyTorch weight layout <c>[C_in, C_out, K]</c>.
    /// Output length is <c>(T_in - 1) * stride + dilation * (K - 1) + 1 - padLeft - padRight</c>.
    ///
    /// <para>Implemented as input-driven scatter: each <c>x[b, ic, i]</c> contributes
    /// <c>x[..] * w[ic, oc, k]</c> to <c>out[b, oc, i * stride + k * dilation - padLeft]</c>
    /// for every <c>k</c> in <c>[0, K)</c> whose target position lies inside
    /// <c>[0, T_out)</c>. Output is initialised with broadcast bias up front so the inner
    /// scatter is a pure accumulate.</para></summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void ConvTranspose1d(
        Tensor output, Tensor input, Tensor weight, Tensor? bias,
        int stride, int padLeft, int padRight, int dilation)
    {
        if (input.Shape.Rank != 3) throw new ArgumentException($"input must be rank-3, got {input.Shape}.");
        if (weight.Shape.Rank != 3) throw new ArgumentException($"weight must be rank-3, got {weight.Shape}.");
        if (output.Shape.Rank != 3) throw new ArgumentException($"output must be rank-3, got {output.Shape}.");

        int batch = (int)input.Shape[0];
        int cIn = (int)input.Shape[1];
        int tIn = (int)input.Shape[2];
        int cOut = (int)weight.Shape[1];
        int kernel = (int)weight.Shape[2];

        if ((int)weight.Shape[0] != cIn)
            throw new ArgumentException($"weight dim 0 ({weight.Shape[0]}) must equal cIn ({cIn}).");

        int tOutRaw = (tIn - 1) * stride + dilation * (kernel - 1) + 1;
        int tOut = tOutRaw - padLeft - padRight;
        if (tOut < 0) throw new ArgumentException($"computed T_out is negative: {tOut} (padLeft={padLeft}, padRight={padRight}, raw={tOutRaw}).");
        if ((int)output.Shape[2] != tOut || (int)output.Shape[1] != cOut || (int)output.Shape[0] != batch)
            throw new ArgumentException($"output shape mismatch: expected [{batch}, {cOut}, {tOut}], got {output.Shape}.");

        float* ip = (float*)input.DataPointer;
        float* op = (float*)output.DataPointer;
        float* wp = (float*)weight.DataPointer;
        float* bp = bias is null ? null : (float*)bias.DataPointer;

        // Init with broadcast bias (or zero).
        for (int b = 0; b < batch; b++)
        {
            for (int oc = 0; oc < cOut; oc++)
            {
                float biasV = bp is null ? 0f : bp[oc];
                int outBase = (b * cOut + oc) * tOut;
                for (int j = 0; j < tOut; j++) op[outBase + j] = biasV;
            }
        }

        // Scatter accumulate.
        for (int b = 0; b < batch; b++)
        {
            for (int ic = 0; ic < cIn; ic++)
            {
                int inBase = (b * cIn + ic) * tIn;
                for (int oc = 0; oc < cOut; oc++)
                {
                    int wBase = (ic * cOut + oc) * kernel;
                    int outBase = (b * cOut + oc) * tOut;
                    for (int i = 0; i < tIn; i++)
                    {
                        float xv = ip[inBase + i];
                        if (xv == 0f) continue;     // sparse-ish skip; helps when input has many zeros
                        int outStart = i * stride - padLeft;
                        for (int k = 0; k < kernel; k++)
                        {
                            int j = outStart + k * dilation;
                            if ((uint)j < (uint)tOut)
                                op[outBase + j] += xv * wp[wBase + k];
                        }
                    }
                }
            }
        }
    }
}
