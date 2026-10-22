using System.Globalization;
using System.Runtime.CompilerServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Tests.Common.Attention;

/// <summary>Records the ops an attention layer issues, in order, without computing anything (no tensor is ever read).</summary>
public sealed class AttentionTraceBackend : IBackend
{
    /// <summary>Every recorded op, in order.</summary>
    public List<string> Ops { get; } = new();

    /// <summary>The number of tensors the stub has seen as an activation operand that are not disposed yet, at each dense attention call.</summary>
    public List<int> LiveAtAttention { get; } = new();

    private readonly HashSet<Tensor> _seen = new(ReferenceEqualityComparer.Instance);

    private static readonly System.Reflection.FieldInfo Disposed =
        typeof(Tensor).GetField("_disposed", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!;

    private int Live => _seen.Count(t => !(bool)Disposed.GetValue(t)!);

    private void Track(params Tensor[] tensors)
    {
        if (Silent) return;
        foreach (Tensor t in tensors) _seen.Add(t);
    }

    /// <summary>The attention calls, in order.</summary>
    public List<AttentionCall> Calls { get; } = new();

    public bool SupportsTokenMajorAttention { get; init; }

    public bool SupportsTokenMajorGqaAttention { get; init; }

    /// <summary>Makes every attention call throw after it is recorded.</summary>
    public bool FailAttention { get; init; }

    /// <summary>Records nothing from <see cref="Permute0213"/>, <see cref="Attention"/> and <see cref="FlashAttention"/>, so an allocation measurement sees only the caller's.</summary>
    public bool Silent { get; init; }

    /// <summary>Records a Flash-route call as one <c>Attention</c> line with its stage parameters instead of re-entering the legacy entry.</summary>
    public bool FlashAsAttention { get; init; }

    /// <summary>Skips the F32-only default bodies of the adaLN norm and modulation ops, so an F16 block reaches its attention.</summary>
    public bool SkipAdaLayerNorm { get; init; }

    public void LayerNormNoAffine(Tensor output, Tensor input, float eps)
    {
        if (!SkipAdaLayerNorm) IBackend.LayerNormNoAffineReference(output, input, eps);
    }

    public void LayerNormModulate(Tensor output, Tensor input, Tensor scale, Tensor shift, float eps)
    {
        if (!SkipAdaLayerNorm) IBackend.LayerNormModulateReference(output, input, scale, shift, eps);
    }

    public void AffineBroadcastLastDim(Tensor output, Tensor input, Tensor scale, Tensor? shift)
    {
        if (!SkipAdaLayerNorm) IBackend.AffineBroadcastLastDimReference(output, input, scale, shift);
    }

    public void Linear(Tensor output, Tensor input, Tensor weight, Tensor? bias)
    {
        Track(output, input);
        Ops.Add($"Linear {Name(weight)}");
    }

    public void LinearMulti(Tensor input, ReadOnlySpan<LinearOp> ops)
    {
        List<string> names = new();
        foreach (LinearOp op in ops) names.Add(Name(op.Weight));
        Ops.Add($"LinearMulti {string.Join(",", names)}");
    }

    public void LinearHeadGated(Tensor output, Tensor input, Tensor weight, Tensor? bias, Tensor gateLogits, int heads, int headDim) =>
        Ops.Add($"LinearHeadGated {Name(weight)}");

    public void RmsNorm(Tensor output, Tensor input, Tensor weight, float eps)
    {
        Track(output, input);
        Ops.Add($"RmsNorm {Name(weight)}");
    }

    public void Ltx2QkNormRopeHeadMajor(Tensor output, Tensor input, Tensor normWeight, Tensor? cos, Tensor? sin, int seq, int heads,
        int headDim, float eps) => Ops.Add($"NormRopeHeadMajor {Name(normWeight)} {(cos is null ? "unrotated" : "rotated")}");

    public void Ltx2QkNormRopeTokenMajor(Tensor output, Tensor input, Tensor normWeight, Tensor? cos, Tensor? sin, int seq, int heads,
        int headDim, float eps) => Ops.Add($"NormRopeTokenMajor {Name(normWeight)} {(cos is null ? "unrotated" : "rotated")}");

    public void WanRopeInterleaved(Tensor x, Tensor cos, Tensor sin, int seqLen, int heads, int headDim) =>
        Ops.Add($"WanRopeInterleaved heads={heads} d={headDim} seq={seqLen}");

    public void WanRopeInterleavedPerHead(Tensor x, Tensor cos, Tensor sin, int seqLen, int heads, int headDim) =>
        Ops.Add($"WanRopeInterleavedPerHead heads={heads} d={headDim} seq={seqLen}");

    public void ApplyRopeSingle(Tensor x, Tensor cos, Tensor sin, int rotaryDim = 0) => Ops.Add("ApplyRopeSingle");

    public void ApplyRope(Tensor q, Tensor k, Tensor cos, Tensor sin) => Ops.Add("ApplyRope");

    public void QkvSplitNorm(Tensor q, Tensor k, Tensor v, Tensor qkv, Tensor qWeight, Tensor kWeight, float eps)
    {
        Track(q, k, v, qkv);
        Ops.Add($"QkvSplitNorm {Name(qWeight)},{Name(kWeight)} {qkv.Shape}");
    }

    public void SliceLastDim(Tensor output, Tensor input, int offset)
    {
        Track(output, input);
        Ops.Add($"SliceLastDim @{offset}");
    }

    public void Concat(Tensor output, ReadOnlySpan<Tensor> inputs, int dim)
    {
        Track(output);
        foreach (Tensor t in inputs) Track(t);
        List<string> rows = new();
        foreach (Tensor t in inputs) rows.Add(t.Shape[1].ToString());
        Ops.Add($"Concat rows {string.Join("+", rows)}");
    }

    public void SliceRows(Tensor output, Tensor input, int rowOffset)
    {
        Track(output, input);
        Ops.Add($"SliceRows {output.Shape[1]}@{rowOffset}");
    }

    public void GatherRows(Tensor output, Tensor input, ReadOnlySpan<int> rowIndices) => Ops.Add($"GatherRows {rowIndices.Length}");

    public void CastToF16(Tensor output, Tensor input)
    {
        Track(output, input);
        Ops.Add($"CastToF16 {input.Shape}");
    }

    public void Permute0213(Tensor output, Tensor input, int s, int h, int d)
    {
        Track(output, input);
        if (!Silent) Ops.Add($"Permute0213 {s},{h},{d}");
    }

    public void RepeatKvHeads(Tensor output, Tensor input, int kvHeads, int groupSize) => Ops.Add($"RepeatKvHeads {kvHeads}x{groupSize}");

    public void Mul(Tensor output, Tensor a, Tensor b)
    {
        Track(output, a, b);
        Ops.Add("Mul");
    }

    public void Sigmoid(Tensor output, Tensor input)
    {
        Track(output, input);
        Ops.Add("Sigmoid");
    }

    public void Attention(in AttentionCall call)
    {
        if (Silent) return;
        Calls.Add(call);
        bool flash = call.Route is AttentionRoute.Flash or AttentionRoute.FlashDevice;
        if (!flash)
        {
            Track(call.Output.Storage, call.Query.Storage, call.Key.Storage, call.Value.Storage);
            LiveAtAttention.Add(Live);
        }
        // Otherwise the Flash routes re-enter their legacy entry, as the real backends do, so the call lands in FlashCalls.
        if (flash && FlashAsAttention)
        {
            if (!Silent) Ops.Add(Describe(in call));
            if (FailAttention) throw new OutOfMemoryException("attention failed");
            return;
        }
        if (call.Route == AttentionRoute.Flash)
        {
            FlashAttention(call.Output.Storage, call.Query.Storage, call.Key.Storage, call.Value.Storage, call.ValidKvLength, call.KvGroup,
                call.Mask.Causal, call.Positions.QueryOffset, call.Scale, call.Logits.Softcap, call.Logits.Sinks, call.Mask.LegacySlidingWindow);
            return;
        }
        if (call.Route == AttentionRoute.FlashDevice)
        {
            FlashAttentionDev(call.Output.Storage, call.Query.Storage, call.Key.Storage, call.Value.Storage, call.ValidKvLength, call.KvGroup,
                call.Mask.Causal, call.Positions.QueryOffset, call.Scale, call.Positions.DeviceHandle, call.Logits.Softcap, call.Mask.LegacySlidingWindow);
            return;
        }
        if (!Silent) Ops.Add(Describe(in call));
        if (FailAttention) throw new OutOfMemoryException("attention failed");
    }

    /// <summary>The trace line of an attention call.</summary>
    public static string Describe(in AttentionCall call) => call.Route is AttentionRoute.Flash or AttentionRoute.FlashDevice
        ? $"Attention {call.Route} heads={call.Heads}/{call.KvHeads} kv={call.ValidKvLength} q@{call.Positions.QueryOffset} "
            + $"{(call.Mask.Causal ? "causal" : "full")} window={call.Mask.LegacySlidingWindow} softcap={call.Logits.Softcap.ToString(CultureInfo.InvariantCulture)}"
            + (call.Positions.OnDevice ? $" dev={call.Positions.DeviceHandle}" : "")
        : $"Attention {call.Route} heads={call.Heads}/{call.KvHeads} bounded={call.Numerics.LegacyAssertedBounded}";

    public void BatchedMatMul(Tensor output, Tensor a, Tensor b) => Ops.Add($"BatchedMatMul {a.Shape} x {b.Shape}");

    public void LayerNorm(Tensor output, Tensor input, Tensor weight, Tensor bias, float eps)
    {
        Track(output, input);
        Ops.Add($"LayerNorm {Name(weight)}");
    }

    public void Split(ReadOnlySpan<Tensor> outputs, Tensor input, int dim)
    {
        foreach (Tensor t in outputs) Track(t);
        Track(input);
        Ops.Add($"Split {outputs.Length} dim={dim}");
    }

    public void Transpose2D(Tensor output, Tensor input, int d1, int d2)
    {
        Track(output, input);
        Ops.Add($"Transpose2D {d1},{d2}");
    }

    public void Add(Tensor output, Tensor a, Tensor b)
    {
        Track(output, a, b);
        Ops.Add("Add");
    }

    public void Gelu(Tensor output, Tensor input)
    {
        Track(output, input);
        Ops.Add("Gelu");
    }

    public void Silu(Tensor output, Tensor input)
    {
        Track(output, input);
        Ops.Add("Silu");
    }

    public void ApplyRopeInterleaved(Tensor x, Tensor cos, Tensor sin, int rotaryDim = 0) => Ops.Add("ApplyRopeInterleaved");

    public void KvCacheAppend(Tensor buffer, Tensor newKv, int offset) => Ops.Add($"KvCacheAppend @{offset} {newKv.Shape}");

    public void KvCacheAppendDev(Tensor buffer, Tensor newKv, int offset, ulong devicePos) => Ops.Add($"KvCacheAppendDev @{offset} dev={devicePos != 0}");

    public void SliceTimeRange(Tensor output, Tensor input, int start, int len) => Ops.Add($"SliceTimeRange {start}+{len}");

    /// <summary>The legacy SDPA calls, in order.</summary>
    public List<SdpaCall> SdpaCalls { get; } = new();

    public void ScaledDotProductAttention(Tensor output, Tensor query, Tensor key, Tensor value, Tensor? mask, float scale, bool allowF16 = false)
    {
        Track(output, query, key, value);
        LiveAtAttention.Add(Live);
        SdpaCalls.Add(new SdpaCall(scale, allowF16, output, query, key, value, mask));
        Ops.Add($"SDPA scale={scale:0.######} allowF16={allowF16} masked={mask is not null}");
        if (FailAttention) throw new OutOfMemoryException("attention failed");
    }

    /// <summary>The Flash calls, in order (the Flash entries are not <see cref="AttentionCall"/>s on this stub).</summary>
    public List<FlashCall> FlashCalls { get; } = new();

    public void FlashAttention(Tensor output, Tensor query, Tensor key, Tensor value, int kvLen, int kvGroup, bool causal, int qOffset, float scale,
        float softcap = 0f, Tensor? sink = null, int slidingWindow = 0, Tensor? alibiSlopes = null)
    {
        FlashCalls.Add(new FlashCall(kvLen, kvGroup, causal, qOffset, scale, softcap, sink is not null, slidingWindow, DevicePos: 0));
        if (!Silent) Ops.Add($"FlashAttention kv={kvLen} group={kvGroup} causal={causal} qOffset={qOffset} window={slidingWindow} sinks={sink is not null}");
        if (FailAttention) throw new OutOfMemoryException("attention failed");
    }

    public void FlashAttentionDev(Tensor output, Tensor query, Tensor key, Tensor value, int kvLen, int kvGroup, bool causal, int qOffset, float scale,
        ulong devicePos, float softcap = 0f, int slidingWindow = 0)
    {
        FlashCalls.Add(new FlashCall(kvLen, kvGroup, causal, qOffset, scale, softcap, false, slidingWindow, devicePos));
        if (!Silent) Ops.Add($"FlashAttentionDev kv={kvLen} group={kvGroup} causal={causal} qOffset={qOffset} window={slidingWindow} dev={devicePos != 0}");
        if (FailAttention) throw new OutOfMemoryException("attention failed");
    }

    /// <summary>Weights are named by their first dim so a trace says which projection ran.</summary>
    private static string Name(Tensor t) => t.Shape.Rank == 0 ? "?" : $"w{t.Shape[0]}";

    /// <summary>Lets the resident (device-cache Flash) decode branches run against the stub.</summary>
    public bool FlashDecodeSupported { get; init; }

    public IStreamingWeightCache? StreamingCache => null;
    public void TrimMemoryPool() { }
    public long FreeMemoryBytes() => long.MaxValue;
    public void PreloadWeights(IEnumerable<Tensor> weights) { }
    public void FreeWeights(IEnumerable<Tensor> weights) { }
    public bool StepGraphSupported => false;
    public bool StepGraphReady { get; set; }
    public object? StepGraphOwner { get; set; }
    public void StepGraphReset() { }
    public DeviceKind Device => DeviceKind.Cpu;
    public BackendCapabilities Capabilities { get; } = new BackendCapabilities { Name = "attention-trace-stub" };
    public void Dispose() { }

    public void MatMul(Tensor output, Tensor a, Tensor b) => throw Unused();
    public void Conv2D(Tensor output, Tensor input, Tensor weight, Tensor? bias, int strideH, int strideW, int padH, int padW) =>
        Ops.Add($"Conv2D {Name(weight)}");
    /// <summary>Records <c>Conv1d</c> and <c>CopyTo</c> instead of throwing, so a site that convolves or clones before it attends reaches its attention.</summary>
    public bool ServeConvs { get; init; }

    public void Conv1d(Tensor output, Tensor input, Tensor weight, Tensor? bias, int stride, int padding, int dilation, int groups, int outputPadding)
    {
        if (!ServeConvs) throw Unused();
        Ops.Add("Conv1d");
    }

    public void ConvTranspose1d(Tensor output, Tensor input, Tensor weight, Tensor? bias, int stride, int padding, int dilation, int groups, int outputPadding) => throw Unused();
    public void GroupNorm(Tensor output, Tensor input, Tensor weight, Tensor bias, int groups, float eps) => Ops.Add($"GroupNorm {groups}");
    public void AdaInstanceNorm1d(Tensor output, Tensor input, Tensor gamma, Tensor beta, float eps) => throw Unused();
    public void Tanh(Tensor output, Tensor input) => throw Unused();
    public void Elu(Tensor output, Tensor input, float alpha) => throw Unused();
    public void LeakyRelu(Tensor output, Tensor input, float slope) => throw Unused();
    public void Snake(Tensor output, Tensor input, Tensor alpha, Tensor? beta) => throw Unused();
    public void Scale(Tensor output, Tensor input, float scalar)
    {
        Track(output, input);
        Ops.Add("Scale");
    }
    public void Clamp(Tensor output, Tensor input, float min, float max) => throw Unused();
    public void GeGlu(Tensor output, Tensor input) => throw Unused();
    public void BroadcastAdd(Tensor hidden, Tensor bias, int channels, int spatial) => throw Unused();
    public void UpsampleNearest2D(Tensor output, Tensor input, int scaleH, int scaleW) => throw Unused();
    public void UpsampleBilinear2D(Tensor output, Tensor input, int scaleH, int scaleW) => throw Unused();
    public void CopyTo(Tensor destination, Tensor source)
    {
        if (!ServeConvs) throw Unused();
        Ops.Add("CopyTo");
    }

    public void Fill(Tensor tensor, float value) => throw Unused();
    public void Fft(Tensor output, Tensor input) => throw Unused();
    public void Stft(Tensor output, Tensor input, int fftSize, int hopLength, Tensor window) => throw Unused();
    public void MelFilterbank(Tensor output, Tensor input, Tensor filters) => throw Unused();

    private static Exception Unused([CallerMemberName] string member = "") =>
        new NotSupportedException($"{member} is not reachable from an attention layer.");
}

/// <summary>One recorded <c>FlashAttention</c> / <c>FlashAttentionDev</c> call.</summary>
public readonly record struct FlashCall(int KvLen, int KvGroup, bool Causal, int QOffset, float Scale, float Softcap, bool HasSinks, int SlidingWindow, ulong DevicePos);

/// <summary>One recorded legacy <c>ScaledDotProductAttention</c> call.</summary>
public readonly record struct SdpaCall(float Scale, bool AllowF16, Tensor Output, Tensor Query, Tensor Key, Tensor Value, Tensor? Mask);
