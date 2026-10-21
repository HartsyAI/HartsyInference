using System.Runtime.CompilerServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Tests.Layers;

/// <summary>Records the ops an attention layer issues, in order, without computing anything (no tensor is ever read).</summary>
internal sealed class AttentionTraceBackend : IBackend
{
    /// <summary>Every recorded op, in order.</summary>
    public List<string> Ops { get; } = new();

    /// <summary>The attention calls, in order.</summary>
    public List<AttentionCall> Calls { get; } = new();

    public bool SupportsTokenMajorAttention { get; init; }

    public bool SupportsTokenMajorGqaAttention { get; init; }

    /// <summary>Makes every attention call throw after it is recorded.</summary>
    public bool FailAttention { get; init; }

    public void Linear(Tensor output, Tensor input, Tensor weight, Tensor? bias) => Ops.Add($"Linear {Name(weight)}");

    public void LinearMulti(Tensor input, ReadOnlySpan<LinearOp> ops)
    {
        List<string> names = new();
        foreach (LinearOp op in ops) names.Add(Name(op.Weight));
        Ops.Add($"LinearMulti {string.Join(",", names)}");
    }

    public void LinearHeadGated(Tensor output, Tensor input, Tensor weight, Tensor? bias, Tensor gateLogits, int heads, int headDim) =>
        Ops.Add($"LinearHeadGated {Name(weight)}");

    public void RmsNorm(Tensor output, Tensor input, Tensor weight, float eps) => Ops.Add($"RmsNorm {Name(weight)}");

    public void Ltx2QkNormRopeHeadMajor(Tensor output, Tensor input, Tensor normWeight, Tensor? cos, Tensor? sin, int seq, int heads,
        int headDim, float eps) => Ops.Add($"NormRopeHeadMajor {Name(normWeight)} {(cos is null ? "unrotated" : "rotated")}");

    public void Ltx2QkNormRopeTokenMajor(Tensor output, Tensor input, Tensor normWeight, Tensor? cos, Tensor? sin, int seq, int heads,
        int headDim, float eps) => Ops.Add($"NormRopeTokenMajor {Name(normWeight)} {(cos is null ? "unrotated" : "rotated")}");

    public void WanRopeInterleaved(Tensor x, Tensor cos, Tensor sin, int seqLen, int heads, int headDim) =>
        Ops.Add($"WanRopeInterleaved heads={heads} d={headDim} seq={seqLen}");

    public void ApplyRopeSingle(Tensor x, Tensor cos, Tensor sin, int rotaryDim = 0) => Ops.Add("ApplyRopeSingle");

    public void ApplyRope(Tensor q, Tensor k, Tensor cos, Tensor sin) => Ops.Add("ApplyRope");

    public void Concat(Tensor output, ReadOnlySpan<Tensor> inputs, int dim)
    {
        List<string> rows = new();
        foreach (Tensor t in inputs) rows.Add(t.Shape[1].ToString());
        Ops.Add($"Concat rows {string.Join("+", rows)}");
    }

    public void SliceRows(Tensor output, Tensor input, int rowOffset) => Ops.Add($"SliceRows {output.Shape[1]}@{rowOffset}");

    public void GatherRows(Tensor output, Tensor input, ReadOnlySpan<int> rowIndices) => Ops.Add($"GatherRows {rowIndices.Length}");

    public void CastToF16(Tensor output, Tensor input) => Ops.Add($"CastToF16 {input.Shape}");

    public void Permute0213(Tensor output, Tensor input, int s, int h, int d) => Ops.Add($"Permute0213 {s},{h},{d}");

    public void RepeatKvHeads(Tensor output, Tensor input, int kvHeads, int groupSize) => Ops.Add($"RepeatKvHeads {kvHeads}x{groupSize}");

    public void Mul(Tensor output, Tensor a, Tensor b) => Ops.Add("Mul");

    public void Sigmoid(Tensor output, Tensor input) => Ops.Add("Sigmoid");

    public void Attention(in AttentionCall call)
    {
        Calls.Add(call);
        Ops.Add($"Attention {call.Route} heads={call.Heads}/{call.KvHeads} bounded={call.Numerics.LegacyAssertedBounded}");
        if (FailAttention) throw new OutOfMemoryException("attention failed");
    }

    /// <summary>Weights are named by their first dim so a trace says which projection ran.</summary>
    private static string Name(Tensor t) => t.Shape.Rank == 0 ? "?" : $"w{t.Shape[0]}";

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
    public void BatchedMatMul(Tensor output, Tensor a, Tensor b) => throw Unused();
    public void Conv2D(Tensor output, Tensor input, Tensor weight, Tensor? bias, int strideH, int strideW, int padH, int padW) => throw Unused();
    public void Conv1d(Tensor output, Tensor input, Tensor weight, Tensor? bias, int stride, int padding, int dilation, int groups, int outputPadding) => throw Unused();
    public void ConvTranspose1d(Tensor output, Tensor input, Tensor weight, Tensor? bias, int stride, int padding, int dilation, int groups, int outputPadding) => throw Unused();
    public void GroupNorm(Tensor output, Tensor input, Tensor weight, Tensor bias, int groups, float eps) => throw Unused();
    public void LayerNorm(Tensor output, Tensor input, Tensor weight, Tensor bias, float eps) => throw Unused();
    public void AdaInstanceNorm1d(Tensor output, Tensor input, Tensor gamma, Tensor beta, float eps) => throw Unused();
    public void ScaledDotProductAttention(Tensor output, Tensor query, Tensor key, Tensor value, Tensor? mask, float scale, bool allowF16 = false) => throw Unused();
    public void Gelu(Tensor output, Tensor input) => throw Unused();
    public void Silu(Tensor output, Tensor input) => throw Unused();
    public void Tanh(Tensor output, Tensor input) => throw Unused();
    public void Elu(Tensor output, Tensor input, float alpha) => throw Unused();
    public void LeakyRelu(Tensor output, Tensor input, float slope) => throw Unused();
    public void Snake(Tensor output, Tensor input, Tensor alpha, Tensor? beta) => throw Unused();
    public void Add(Tensor output, Tensor a, Tensor b) => throw Unused();
    public void Scale(Tensor output, Tensor input, float scalar) => throw Unused();
    public void Clamp(Tensor output, Tensor input, float min, float max) => throw Unused();
    public void Transpose2D(Tensor output, Tensor input, int d1, int d2) => throw Unused();
    public void GeGlu(Tensor output, Tensor input) => throw Unused();
    public void BroadcastAdd(Tensor hidden, Tensor bias, int channels, int spatial) => throw Unused();
    public void Split(ReadOnlySpan<Tensor> outputs, Tensor input, int dim) => throw Unused();
    public void UpsampleNearest2D(Tensor output, Tensor input, int scaleH, int scaleW) => throw Unused();
    public void UpsampleBilinear2D(Tensor output, Tensor input, int scaleH, int scaleW) => throw Unused();
    public void CopyTo(Tensor destination, Tensor source) => throw Unused();
    public void Fill(Tensor tensor, float value) => throw Unused();
    public void Fft(Tensor output, Tensor input) => throw Unused();
    public void Stft(Tensor output, Tensor input, int fftSize, int hopLength, Tensor window) => throw Unused();
    public void MelFilterbank(Tensor output, Tensor input, Tensor filters) => throw Unused();

    private static Exception Unused([CallerMemberName] string member = "") =>
        new NotSupportedException($"{member} is not reachable from an attention layer.");
}
