using HartsyInference.Core.Backends;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Tests.Attention;

/// <summary>An <see cref="IBackend"/> over <see cref="CpuBackend"/> that serves the device-position decode ops (position buffer, RoPE row, KV append, flash, embed gather, argmax) on the host,
/// so the graph-decode steps run eagerly on the CPU. Digests taken through it compare before and after a refactor and are not comparable with the plain CPU backend's.</summary>
internal sealed unsafe class DevicePositionShimBackend(bool graphDecodeSupported = false) : IBackend
{
    private readonly IBackend _cpu = new CpuBackend();
    private readonly Dictionary<ulong, (int KvLen, int QOffset)> _positions = [];
    private readonly Dictionary<ulong, int> _tokens = [];
    private ulong _next = 1;

    public bool GraphDecodeSupported => graphDecodeSupported;

    public bool LlmGraphDecodeSupported => true;

    public void Dispose() => _cpu.Dispose();

    public ulong AllocDevicePos()
    {
        ulong handle = _next++;
        _positions[handle] = (0, 0);
        return handle;
    }

    public void WriteDevicePos(ulong handle, int kvLen, int qOffset) => _positions[handle] = (kvLen, qOffset);

    public void FreeDevicePos(ulong handle) => _positions.Remove(handle);

    /// <summary>Appends one step at the position's qOffset; the composed full-norm epilogue hands K/V over token-major, so a rank-3 step is viewed head-major first.</summary>
    public void KvCacheAppendDev(Tensor buffer, Tensor newKv, int offset, ulong devicePos)
    {
        int at = devicePos == 0 ? offset : _positions[devicePos].QOffset;
        if (newKv.Shape.Rank == 4)
        {
            _cpu.KvCacheAppend(buffer, newKv, at);
            return;
        }
        using Tensor view = newKv.Reshape(new TensorShape(1, buffer.Shape[1], 1, buffer.Shape[3]));
        _cpu.KvCacheAppend(buffer, view, at);
    }

    public void FlashAttentionDev(Tensor output, Tensor query, Tensor key, Tensor value, int kvLen, int kvGroup, bool causal, int qOffset, float scale,
        ulong devicePos, float softcap = 0f, int slidingWindow = 0)
    {
        (int devKv, int devOffset) = devicePos == 0 ? (kvLen, qOffset) : _positions[devicePos];
        _cpu.FlashAttention(output, query, key, value, devKv, kvGroup, causal, devOffset, scale, softcap, sink: null, slidingWindow, alibiSlopes: null);
    }

    public (Tensor cos, Tensor sin) BuildRopeTableDevice(int maxPos, int headDim, int rotaryDim, float theta, HartsyInference.Core.Rope.RopeScaling scaling,
        bool splitHalfPartial = false)
    {
        Tensor cos = new(new TensorShape(1, maxPos, headDim), DType.F32);
        Tensor sin = new(new TensorShape(1, maxPos, headDim), DType.F32);
        RopeTables.BuildRope(cos, sin, maxPos, 0, headDim, rotaryDim, theta, scaling);
        return (cos, sin);
    }

    /// <summary>Rotates <paramref name="x"/> (<c>[1, heads, 1, d]</c>, or token-major with the same element order) with table row qOffset through the same reference rotation the eager path uses (split-half only).</summary>
    public void RopeApplyDecodeStep(Tensor x, Tensor cosTable, Tensor sinTable, int rotaryDim, bool interleaved, ulong devicePos)
    {
        if (interleaved) throw new NotSupportedException("The shim rotates split-half only.");
        int d = (int)cosTable.Shape[2];
        int row = _positions[devicePos].QOffset;
        using Tensor cos = new(new TensorShape(1, 1, d), DType.F32);
        using Tensor sin = new(new TensorShape(1, 1, d), DType.F32);
        new ReadOnlySpan<float>((float*)cosTable.DataPointer + (long)row * d, d).CopyTo(cos.AsSpan<float>());
        new ReadOnlySpan<float>((float*)sinTable.DataPointer + (long)row * d, d).CopyTo(sin.AsSpan<float>());
        using Tensor view = x.Reshape(new TensorShape(1, 1, (int)(x.ElementCount / d), d));
        _cpu.ApplyRopeSingle(view, cos, sin, rotaryDim);
    }

    public ulong AllocDeviceTokenId()
    {
        ulong handle = _next++;
        _tokens[handle] = 0;
        return handle;
    }

    public void WriteDeviceTokenId(ulong handle, int tokenId) => _tokens[handle] = tokenId;

    public void FreeDeviceTokenId(ulong handle) => _tokens.Remove(handle);

    public int ReadDeviceTokenId(ulong handle) => _tokens[handle];

    public void EmbedGatherDecodeStep(Tensor output, Tensor embedTable, ulong tokenId)
    {
        int width = (int)output.ElementCount;
        new ReadOnlySpan<float>((float*)embedTable.DataPointer + (long)_tokens[tokenId] * width, width).CopyTo(output.AsSpan<float>());
    }

    public void ArgMaxInto(ulong outputTokenId, Tensor input)
    {
        ReadOnlySpan<float> logits = input.AsSpan<float>();
        int best = 0;
        for (int i = 1; i < logits.Length; i++)
        {
            if (logits[i] > logits[best]) best = i;
        }
        _tokens[outputTokenId] = best;
    }

    // CpuBackend's own members, forwarded.
#nullable disable
    /// <summary>Serves a device-position call through this shim's <see cref="FlashAttentionDev"/>, as a real backend re-enters its own legacy entry.</summary>
    public void Attention(in AttentionCall call)
    {
        if (call.Route == AttentionRoute.FlashDevice && call.LegacyGap() is null)
            FlashAttentionDev(call.Output.Storage, call.Query.Storage, call.Key.Storage, call.Value.Storage, call.ValidKvLength, call.KvGroup,
                call.Mask.Causal, call.Positions.QueryOffset, call.Scale, call.Positions.DeviceHandle, call.Logits.Softcap, call.Mask.LegacySlidingWindow);
        else
            _cpu.Attention(in call);
    }
    public AttentionPlan PlanAttention(in AttentionCall call) => _cpu.PlanAttention(in call);
    public void SparseLatentAttention(Tensor output, Tensor query, in LatentSource window, in LatentSource main, Tensor indices, int windowSlots, Tensor sink, float scale) => _cpu.SparseLatentAttention(output, query, in window, in main, indices, windowSlots, sink, scale);
    public void IndexerScores(Tensor scores, Tensor query, in LatentSource keys, Tensor headWeights, Tensor compressLens, Tensor candidates, float scale) => _cpu.IndexerScores(scores, query, in keys, headWeights, compressLens, candidates, scale);
    public void HcSplitSinkhorn(Tensor pre, Tensor post, Tensor comb, Tensor mixes, Tensor scale, Tensor bias, int hc, int iters, float eps) => _cpu.HcSplitSinkhorn(pre, post, comb, mixes, scale, bias, hc, iters, eps);
    public void HcPreMix(Tensor output, Tensor x, Tensor pre) => _cpu.HcPreMix(output, x, pre);
    public void HcPostMix(Tensor output, Tensor x, Tensor residual, Tensor post, Tensor comb) => _cpu.HcPostMix(output, x, residual, post, comb);
    public void QuantizeLatentRows(in LatentSource dest, Tensor rows, Tensor physicalRows) => _cpu.QuantizeLatentRows(in dest, rows, physicalRows);
    public void ActQuantDequantInPlace(Tensor x, LatentEncoding encoding) => _cpu.ActQuantDequantInPlace(x, encoding);
    public void BuildWindowIndices(Tensor indices, int windowSize, int seqLen, int startPos) => _cpu.BuildWindowIndices(indices, windowSize, seqLen, startPos);
    public void ApplyRopeInterleaved(Tensor x, Tensor cos, Tensor sin, int rotaryDim, int dimOffset) => _cpu.ApplyRopeInterleaved(x, cos, sin, rotaryDim, dimOffset);
    public void MatMul(Tensor output, Tensor a, Tensor b) => _cpu.MatMul(output, a, b);
    public void BatchedMatMul(Tensor output, Tensor a, Tensor b) => _cpu.BatchedMatMul(output, a, b);
    public void Linear(Tensor output, Tensor input, Tensor weight, Tensor bias) => _cpu.Linear(output, input, weight, bias);
    public void Conv2D(Tensor output, Tensor input, Tensor weight, Tensor bias, int strideH, int strideW, int padH, int padW) => _cpu.Conv2D(output, input, weight, bias, strideH, strideW, padH, padW);
    public void GroupNorm(Tensor output, Tensor input, Tensor weight, Tensor bias, int groups, float eps) => _cpu.GroupNorm(output, input, weight, bias, groups, eps);
    public void LayerNorm(Tensor output, Tensor input, Tensor weight, Tensor bias, float eps) => _cpu.LayerNorm(output, input, weight, bias, eps);
    public void RmsNorm(Tensor output, Tensor input, Tensor weight, float eps) => _cpu.RmsNorm(output, input, weight, eps);
    public void AdaInstanceNorm1d(Tensor output, Tensor input, Tensor gamma, Tensor beta, float eps) => _cpu.AdaInstanceNorm1d(output, input, gamma, beta, eps);
    public void QuantizeActivationsU8(Tensor output, Tensor input) => _cpu.QuantizeActivationsU8(output, input);
    public void LinearI8U8(Tensor output, Tensor input, Tensor weight, Tensor scale, Tensor bias, Tensor diag, Tensor diagInput) => _cpu.LinearI8U8(output, input, weight, scale, bias, diag, diagInput);
    public void ScaledDotProductAttention(Tensor output, Tensor query, Tensor key, Tensor value, Tensor mask, float scale, bool allowF16) => _cpu.ScaledDotProductAttention(output, query, key, value, mask, scale, allowF16);
    public void Gelu(Tensor output, Tensor input) => _cpu.Gelu(output, input);
    public void Silu(Tensor output, Tensor input) => _cpu.Silu(output, input);
    public void Sigmoid(Tensor output, Tensor input) => _cpu.Sigmoid(output, input);
    public void Tanh(Tensor output, Tensor input) => _cpu.Tanh(output, input);
    public void Elu(Tensor output, Tensor input, float alpha) => _cpu.Elu(output, input, alpha);
    public void LeakyRelu(Tensor output, Tensor input, float slope) => _cpu.LeakyRelu(output, input, slope);
    public void Snake(Tensor output, Tensor input, Tensor alpha, Tensor beta) => _cpu.Snake(output, input, alpha, beta);
    public void Add(Tensor output, Tensor a, Tensor b) => _cpu.Add(output, a, b);
    public void Mul(Tensor output, Tensor a, Tensor b) => _cpu.Mul(output, a, b);
    public void Scale(Tensor output, Tensor input, float scalar) => _cpu.Scale(output, input, scalar);
    public void Clamp(Tensor output, Tensor input, float min, float max) => _cpu.Clamp(output, input, min, max);
    public void Transpose2D(Tensor output, Tensor input, int d1, int d2) => _cpu.Transpose2D(output, input, d1, d2);
    public void Permute0213(Tensor output, Tensor input, int s, int h, int d) => _cpu.Permute0213(output, input, s, h, d);
    public void GeGlu(Tensor output, Tensor input) => _cpu.GeGlu(output, input);
    public void BroadcastAdd(Tensor hidden, Tensor bias, int channels, int spatial) => _cpu.BroadcastAdd(hidden, bias, channels, spatial);
    public void Concat(Tensor output, ReadOnlySpan<Tensor> inputs, int dim) => _cpu.Concat(output, inputs, dim);
    public void Split(ReadOnlySpan<Tensor> outputs, Tensor input, int dim) => _cpu.Split(outputs, input, dim);
    public void Conv1d(Tensor output, Tensor input, Tensor weight, Tensor bias, int stride, int padLeft, int padRight, int dilation, int groups) => _cpu.Conv1d(output, input, weight, bias, stride, padLeft, padRight, dilation, groups);
    public void ConvTranspose1d(Tensor output, Tensor input, Tensor weight, Tensor bias, int stride, int padLeft, int padRight, int dilation, int groups) => _cpu.ConvTranspose1d(output, input, weight, bias, stride, padLeft, padRight, dilation, groups);
    public void UpsampleNearest2D(Tensor output, Tensor input, int scaleH, int scaleW) => _cpu.UpsampleNearest2D(output, input, scaleH, scaleW);
    public void UpsampleBilinear2D(Tensor output, Tensor input, int scaleH, int scaleW) => _cpu.UpsampleBilinear2D(output, input, scaleH, scaleW);
    public void CopyTo(Tensor destination, Tensor source) => _cpu.CopyTo(destination, source);
    public void Fill(Tensor tensor, float value) => _cpu.Fill(tensor, value);
    public void Fft(Tensor output, Tensor input) => _cpu.Fft(output, input);
    public void Stft(Tensor output, Tensor input, int fftSize, int hopLength, Tensor window) => _cpu.Stft(output, input, fftSize, hopLength, window);
    public void MelFilterbank(Tensor output, Tensor input, Tensor filters) => _cpu.MelFilterbank(output, input, filters);
    public void Softplus(Tensor output, Tensor input) => _cpu.Softplus(output, input);
    public void MoeRoute(Tensor topkIdx, Tensor topkWeight, Tensor logits, in MoeRouteArgs args, Tensor bias, Tensor altBias, Tensor tokenKinds) => _cpu.MoeRoute(topkIdx, topkWeight, logits, in args, bias, altBias, tokenKinds);
    public void MoeBuildDispatch(Tensor counts, Tensor offsets, Tensor permutedToken, Tensor pairSlot, Tensor topkIdx, int numExperts) => _cpu.MoeBuildDispatch(counts, offsets, permutedToken, pairSlot, topkIdx, numExperts);
    public void MoeCombine(Tensor output, Tensor expertOut, Tensor pairSlot, Tensor topkWeight, int k, bool accumulate) => _cpu.MoeCombine(output, expertOut, pairSlot, topkWeight, k, accumulate);
    public void TopKLastDim(Tensor values, Tensor indices, Tensor input, int k, Tensor validLengths, bool sortByIndex) => _cpu.TopKLastDim(values, indices, input, k, validLengths, sortByIndex);
    public DeviceKind Device { get => _cpu.Device; }
    public BackendCapabilities Capabilities { get => _cpu.Capabilities; }
    public bool SupportsDeviceStepCacheGate { get => _cpu.SupportsDeviceStepCacheGate; }
#nullable restore
}
