using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Cpu.Kernels;

/// <summary>CPU compute kernels for 1D convolution and transposed convolution over channels-first <c>[B, C, T]</c> tensors.</summary>
/// <remarks>These are the whole of Piper's forward pass — its projections and feed-forwards are all expressed as
/// 1x1 convolutions — so both kernels fan out across cores via <see cref="CpuParallel"/>. The work is split over
/// output channels and, when a layer has too few of those to fill the machine, over the time axis as well: a
/// vocoder's last convolution has a single output channel and tens of thousands of samples, and splitting only
/// on channels would leave exactly that layer serial. Small calls still run inline — see
/// <see cref="CpuParallel.MinWorkForParallel"/>.
///
/// <para>Layout assumptions: Conv1d weight is <c>[C_out, C_in / groups, K]</c> (PyTorch nn.Conv1d),
/// ConvTranspose1d weight is <c>[C_in, C_out, K]</c> (PyTorch nn.ConvTranspose1d), and bias is <c>[C_out]</c>
/// (optional).</para>
///
/// <para>Padding is exposed as separate <c>padLeft</c> / <c>padRight</c> values rather than a single symmetric
/// value — most audio codecs need asymmetric (causal) padding, and giving the caller control here saves a kludge
/// layer in every codec.</para></remarks>
public static class Conv1dKernels
{
    /// <summary>1D convolution. Caller pre-allocates the output tensor.</summary>
    /// <remarks>Output shape must be <c>[B, C_out, (T_in + padLeft + padRight - dilation*(K-1) - 1) / stride + 1]</c>.</remarks>
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

        // Each (batch, out-channel, time-slice) task owns a disjoint span of the output, so the split is
        // race-free however the two axes divide up. Time is only chopped when channels alone cannot fill the
        // worker pool, because a contiguous run of j reuses the same weight row out of cache.
        int rows = batch * cOut;
        int timeChunks = CpuParallel.TargetChunks(rows);
        int chunkLen = CpuParallel.ChunkSize(tOut, timeChunks);
        timeChunks = chunkLen > 0 ? (tOut + chunkLen - 1) / chunkLen : 1;
        if (timeChunks < 1) timeChunks = 1;
        long work = (long)rows * tOut * inPerGroup * kernel;

        CpuParallel.For(rows * timeChunks, work, task =>
        {
            int row = task / timeChunks;
            int chunk = task - row * timeChunks;
            int b = row / cOut;
            int oc = row - b * cOut;
            int jStart = chunk * chunkLen;
            int jEnd = Math.Min(tOut, jStart + chunkLen);

            int group = oc / outPerGroup;
            int icStart = group * inPerGroup;
            float biasV = bp is null ? 0f : bp[oc];
            int wRow = oc * inPerGroup * kernel;
            int outBase = (b * cOut + oc) * tOut;

            if (stride == 1)
            {
                // Stride 1 is almost everything a VITS/HiFi-GAN graph runs: its projections and feed-forwards
                // are 1x1 convolutions and its residual stacks are dilated k3. With unit stride the source
                // index moves in lockstep with j, so one (input channel, tap) pair contributes a scaled copy
                // of a contiguous input run to a contiguous output run — a vectorizable accumulate, instead of
                // the gather's strided walk across channels for every single output element.
                //
                // The per-element order of accumulation is unchanged (input channel major, then tap), so this
                // produces bit-identical results to the gather below.
                for (int j = jStart; j < jEnd; j++) op[outBase + j] = biasV;

                for (int ic = 0; ic < inPerGroup; ic++)
                {
                    int inBase = (b * cIn + icStart + ic) * tIn;
                    int wBase = wRow + ic * kernel;
                    for (int k = 0; k < kernel; k++)
                    {
                        float w = wp[wBase + k];
                        if (w == 0f) continue;
                        int shift = k * dilation - padLeft;
                        // Clamp to the window where src = j + shift is a real input sample; outside it the
                        // gather form contributed nothing, so neither does this.
                        int lo = Math.Max(jStart, -shift);
                        int hi = Math.Min(jEnd, tIn - shift);
                        if (lo >= hi) continue;

                        float* src = ip + inBase + shift;
                        float* dst = op + outBase;
                        int j = lo;
                        if (Vector.IsHardwareAccelerated && hi - lo >= Vector<float>.Count)
                        {
                            Vector<float> wv = new(w);
                            int vecEnd = hi - Vector<float>.Count;
                            for (; j <= vecEnd; j += Vector<float>.Count)
                            {
                                Vector<float> acc = Vector.Load(dst + j);
                                Vector<float> x = Vector.Load(src + j);
                                Vector.Store(acc + x * wv, dst + j);
                            }
                        }
                        for (; j < hi; j++) dst[j] += src[j] * w;
                    }
                }
                return;
            }

            for (int j = jStart; j < jEnd; j++)
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
        });
    }

    /// <summary>1D transposed convolution. PyTorch weight layout <c>[C_in, C_out, K]</c>.</summary>
    /// <remarks>Output length is <c>(T_in - 1) * stride + dilation * (K - 1) + 1 - padLeft - padRight</c>.</remarks>
    /// <remarks>Each <c>x[b, ic, i]</c> contributes <c>x[..] * w[ic, oc, k]</c> to
    /// <c>out[b, oc, i * stride + k * dilation - padLeft]</c> for every <c>k</c> in <c>[0, K)</c> whose target
    /// position lies inside <c>[0, T_out)</c>. The loops are ordered output-channel-first so that one task owns
    /// one output row end to end — bias fill included — which is what lets the work fan out across cores; the
    /// older input-channel-first ordering computed the same sums but had every channel in a group accumulating
    /// into shared rows.</remarks>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void ConvTranspose1d(
        Tensor output, Tensor input, Tensor weight, Tensor? bias,
        int stride, int padLeft, int padRight, int dilation, int groups)
    {
        if (input.Shape.Rank != 3) throw new ArgumentException($"input must be rank-3, got {input.Shape}.");
        if (weight.Shape.Rank != 3) throw new ArgumentException($"weight must be rank-3, got {weight.Shape}.");
        if (output.Shape.Rank != 3) throw new ArgumentException($"output must be rank-3, got {output.Shape}.");
        if (groups < 1) throw new ArgumentException($"groups must be >= 1, got {groups}.");

        int batch = (int)input.Shape[0];
        int cIn = (int)input.Shape[1];
        int tIn = (int)input.Shape[2];
        // PyTorch ConvTranspose1d weight is [cIn, cOut/groups, K]; full cOut = ocPerG * groups.
        int ocPerG = (int)weight.Shape[1];
        int cOut = ocPerG * groups;
        int kernel = (int)weight.Shape[2];

        if ((int)weight.Shape[0] != cIn)
            throw new ArgumentException($"weight dim 0 ({weight.Shape[0]}) must equal cIn ({cIn}).");
        if (cIn % groups != 0 || cOut % groups != 0)
            throw new ArgumentException($"channels must divide groups: cIn={cIn}, cOut={cOut}, groups={groups}.");
        int icPerG = cIn / groups;

        int tOutRaw = (tIn - 1) * stride + dilation * (kernel - 1) + 1;
        int tOut = tOutRaw - padLeft - padRight;
        if (tOut < 0) throw new ArgumentException($"computed T_out is negative: {tOut} (padLeft={padLeft}, padRight={padRight}, raw={tOutRaw}).");
        if ((int)output.Shape[2] != tOut || (int)output.Shape[1] != cOut || (int)output.Shape[0] != batch)
            throw new ArgumentException($"output shape mismatch: expected [{batch}, {cOut}, {tOut}], got {output.Shape}.");

        float* ip = (float*)input.DataPointer;
        float* op = (float*)output.DataPointer;
        float* wp = (float*)weight.DataPointer;
        float* bp = bias is null ? null : (float*)bias.DataPointer;

        // Output-channel-major, so each task owns one whole [b, oc, :] row: it writes the bias in and then
        // accumulates every contributing input channel into it. The arithmetic is identical to the
        // input-driven form, but ordering it this way is what makes the split safe — with the input channel
        // outermost, every channel in a group accumulates into the same output row, and two workers would
        // collide on the += below.
        int rows = batch * cOut;
        long work = (long)rows * icPerG * tIn * kernel;

        CpuParallel.For(rows, work, row =>
        {
            int b = row / cOut;
            int oc = row - b * cOut;
            int g = oc / ocPerG;
            int ocLocal = oc - g * ocPerG;
            int outBase = (b * cOut + oc) * tOut;

            float biasV = bp is null ? 0f : bp[oc];
            for (int j = 0; j < tOut; j++) op[outBase + j] = biasV;

            int icStart = g * icPerG;
            for (int ic = icStart; ic < icStart + icPerG; ic++)
            {
                int inBase = (b * cIn + ic) * tIn;
                int wBase = (ic * ocPerG + ocLocal) * kernel;
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
        });
    }
}
