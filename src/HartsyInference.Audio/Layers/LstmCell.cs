using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Layers;

/// <summary>One step of a vanilla PyTorch <c>nn.LSTMCell</c>. Stateless — the caller
/// owns the <c>(h_prev, c_prev)</c> pair across timesteps. Mirrors PyTorch's gate order
/// and bias convention exactly so safetensors weights load without remapping:
/// <code>
///   gates = x @ W_ih.T + b_ih + h_prev @ W_hh.T + b_hh        # [B, 4*hidden]
///   i, f, g, o = gates.chunk(4, dim=-1)                       # PyTorch order: i,f,g,o
///   c_new = sigmoid(f) * c_prev + sigmoid(i) * tanh(g)
///   h_new = sigmoid(o) * tanh(c_new)
/// </code>
///
/// <para>Used by Kokoro's prosody predictor + duration encoder, GPT-SoVITS, and several
/// VITS-derived TTS. Wrapped by <see cref="BiLstm"/> for full bidirectional sequence
/// passes (the much more common usage in practice).</para>
///
/// <para>PyTorch stores LSTM weights with the standard <c>weight_ih_l0</c> / <c>weight_hh_l0</c>
/// / <c>bias_ih_l0</c> / <c>bias_hh_l0</c> naming — and <c>_l0_reverse</c> variants for the
/// backward direction. Weight tensors come in as <c>[4*hidden, input/hidden]</c> in the
/// expected order; we hand them directly to <see cref="WhisperOps.ProjectLinear"/> which
/// does the transpose-and-bias-add internally.</para></summary>
internal sealed class LstmCell
{
    public int InputDim { get; }
    public int HiddenDim { get; }

    private Tensor? _wIh;       // [4*hidden, input]
    private Tensor? _wHh;       // [4*hidden, hidden]
    private Tensor? _bIh;       // [4*hidden]
    private Tensor? _bHh;       // [4*hidden]

    public LstmCell(int inputDim, int hiddenDim)
    {
        if (inputDim <= 0) throw new ArgumentOutOfRangeException(nameof(inputDim));
        if (hiddenDim <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenDim));
        InputDim = inputDim;
        HiddenDim = hiddenDim;
    }

    /// <summary>Input-to-hidden weight <c>[4*hidden, input]</c>; null until <see cref="BindWeights"/>.</summary>
    public Tensor? WeightIh => _wIh;

    /// <summary>Hidden-to-hidden weight <c>[4*hidden, hidden]</c>; null until <see cref="BindWeights"/>.</summary>
    public Tensor? WeightHh => _wHh;

    /// <summary>Input bias <c>[4*hidden]</c>; null until <see cref="BindWeights"/>.</summary>
    public Tensor? BiasIh => _bIh;

    /// <summary>Hidden bias <c>[4*hidden]</c>; null until <see cref="BindWeights"/>.</summary>
    public Tensor? BiasHh => _bHh;

    /// <summary>Hands the cell its four weight tensors (already F32). Caller resolves
    /// the PyTorch state-dict key paths — typically via <see cref="BiLstm.LoadWeights"/>
    /// which knows about the <c>_l0</c> / <c>_l0_reverse</c> direction suffix.</summary>
    public void BindWeights(Tensor weightIh, Tensor weightHh, Tensor biasIh, Tensor biasHh)
    {
        _wIh = weightIh;
        _wHh = weightHh;
        _bIh = biasIh;
        _bHh = biasHh;
    }

    /// <summary>One LSTM step. Inputs are rank-2 <c>[batch, dim]</c>; outputs are
    /// freshly-allocated rank-2 <c>(h_new, c_new)</c> each <c>[batch, hidden]</c>.
    /// Caller owns disposal of the returned tensors.</summary>
    public (Tensor HNew, Tensor CNew) Step(IBackend backend, Tensor x, Tensor hPrev, Tensor cPrev, int batch)
    {
        if (_wIh is null || _wHh is null || _bIh is null || _bHh is null)
            throw new InvalidOperationException("LstmCell weights not bound.");
        int hidden4 = 4 * HiddenDim;

        // Reshape rank-2 inputs to [1, B, D] views so ProjectLinear can run them.
        Tensor x3 = x.Reshape(new TensorShape(1, batch, InputDim));
        Tensor hPrev3 = hPrev.Reshape(new TensorShape(1, batch, HiddenDim));

        Tensor gatesIh = WhisperOps.ProjectLinear(backend, x3, _wIh, _bIh, 1, batch, InputDim, hidden4);
        Tensor gatesHh = WhisperOps.ProjectLinear(backend, hPrev3, _wHh, _bHh, 1, batch, HiddenDim, hidden4);

        Tensor gates = new(gatesIh.Shape, DType.F32);
        backend.Add(gates, gatesIh, gatesHh);
        gatesIh.Dispose();
        gatesHh.Dispose();

        Tensor hNew = new(new TensorShape(batch, HiddenDim), DType.F32);
        Tensor cNew = new(new TensorShape(batch, HiddenDim), DType.F32);
        LstmOps.GateAndUpdate(gates, cPrev, hNew, cNew, batch, HiddenDim);
        gates.Dispose();
        return (hNew, cNew);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_wIh, _wHh, _bIh, _bHh];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }
}

/// <summary>Fused LSTM gate-and-update step. Combines the four gate activations
/// (sigmoid / sigmoid / tanh / sigmoid) and the c/h update math into one sweep over the
/// <c>[B, 4*hidden]</c> gate matrix so we don't allocate 4 intermediate gate tensors,
/// 2 activation tensors, and a final tanh tensor for every LSTM step.
///
/// <para>The 4-way split along the last dim follows PyTorch's gate order: the first
/// <c>hidden</c> columns are the input gate, the next <c>hidden</c> the forget gate, the
/// next the cell-update candidate, the last the output gate.</para></summary>
internal static unsafe class LstmOps
{
    public static void GateAndUpdate(Tensor gates, Tensor cPrev, Tensor hNew, Tensor cNew, int batch, int hidden)
    {
        int hidden4 = 4 * hidden;
        float* g = (float*)gates.DataPointer;
        float* c0 = (float*)cPrev.DataPointer;
        float* hOut = (float*)hNew.DataPointer;
        float* cOut = (float*)cNew.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            int gateRow = b * hidden4;
            int outRow = b * hidden;
            GateAndUpdateRow(g + gateRow, c0 + outRow, hOut + outRow, cOut + outRow, hidden);
        }
    }

    /// <summary>Runs one direction of a single-layer LSTM over a whole sequence on the host, from input
    /// projections computed up front: <paramref name="gatesIn"/> holds <c>x·W_ihᵀ + b_ih</c> for every
    /// timestep (row <c>t</c> at <c>gatesIn + t·gatesInStride</c>, <c>4·hidden</c> wide). Each step adds
    /// <c>h·W_hhᵀ + b_hh</c> and applies the gate math; <c>h</c> lands in <paramref name="output"/> at
    /// <c>t·outputStride + outputOffset</c>. The recurrence is the only sequential part of an LSTM and its weight is
    /// cache-sized, so a SIMD dot per gate row here beats a per-step device launch (each one a stream drain) and
    /// never touches the backend, leaving the surrounding graph resident.</summary>
    /// <param name="reverse">Walks <c>t = T-1 … 0</c> (the backward direction of a BiLSTM).</param>
    /// <param name="h">Scratch <c>[hidden]</c>, zeroed here (PyTorch's zero initial state).</param>
    /// <param name="c">Scratch <c>[hidden]</c>, zeroed here.</param>
    /// <param name="gates">Scratch <c>[4·hidden]</c>.</param>
    public static void RunSequence(float* gatesIn, int gatesInStride, float* wHh, float* bHh, int t, int hidden,
        bool reverse, float* output, int outputStride, int outputOffset, float* h, float* c, float* gates)
    {
        int hidden4 = 4 * hidden;
        new Span<float>(h, hidden).Clear();
        new Span<float>(c, hidden).Clear();
        for (int i = 0; i < t; i++)
        {
            int step = reverse ? t - 1 - i : i;
            float* inRow = gatesIn + (long)step * gatesInStride;
            for (int k = 0; k < hidden4; k++)
            {
                gates[k] = inRow[k] + bHh[k] + Dot(wHh + (long)k * hidden, h, hidden);
            }
            // c is updated in place; h is read by every gate row above, so it is written only after the sweep.
            GateAndUpdateRow(gates, c, h, c, hidden);
            new ReadOnlySpan<float>(h, hidden).CopyTo(new Span<float>(output + (long)step * outputStride + outputOffset, hidden));
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void GateAndUpdateRow(float* g, float* c0, float* hOut, float* cOut, int hidden)
    {
        for (int k = 0; k < hidden; k++)
        {
            float iGate = Activations.SigmoidS(g[k]);
            float fGate = Activations.SigmoidS(g[hidden + k]);
            float gGate = MathF.Tanh(g[2 * hidden + k]);
            float oGate = Activations.SigmoidS(g[3 * hidden + k]);

            float cNext = fGate * c0[k] + iGate * gGate;
            cOut[k] = cNext;
            hOut[k] = oGate * MathF.Tanh(cNext);
        }
    }

    /// <summary>Four independent accumulators: a single chain is bound by FMA latency, not throughput. The x86 FMA
    /// intrinsics and an explicit lane reduction are used on every target framework — <c>Vector.FusedMultiplyAdd</c>
    /// does not exist on net8.0, and a framework-specific reduction order would make the net8.0 and net10.0 builds
    /// disagree in the last bit on the same machine.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Dot(float* a, float* b, int n)
    {
        int i = 0;
        float sum = 0f;
        if (Fma.IsSupported)
        {
            Vector256<float> acc0 = Vector256<float>.Zero, acc1 = acc0, acc2 = acc0, acc3 = acc0;
            for (; i <= n - 32; i += 32)
            {
                acc0 = Fma.MultiplyAdd(Avx.LoadVector256(a + i), Avx.LoadVector256(b + i), acc0);
                acc1 = Fma.MultiplyAdd(Avx.LoadVector256(a + i + 8), Avx.LoadVector256(b + i + 8), acc1);
                acc2 = Fma.MultiplyAdd(Avx.LoadVector256(a + i + 16), Avx.LoadVector256(b + i + 16), acc2);
                acc3 = Fma.MultiplyAdd(Avx.LoadVector256(a + i + 24), Avx.LoadVector256(b + i + 24), acc3);
            }
            for (; i <= n - 8; i += 8)
            {
                acc0 = Fma.MultiplyAdd(Avx.LoadVector256(a + i), Avx.LoadVector256(b + i), acc0);
            }
            Vector256<float> total = Avx.Add(Avx.Add(acc0, acc1), Avx.Add(acc2, acc3));
            Vector128<float> half = Sse.Add(total.GetLower(), total.GetUpper());
            sum = (half.GetElement(0) + half.GetElement(2)) + (half.GetElement(1) + half.GetElement(3));
        }
        else if (Vector.IsHardwareAccelerated)
        {
            int width = Vector<float>.Count;
            Vector<float> acc = Vector<float>.Zero;
            for (; i <= n - width; i += width)
            {
                acc += Vector.Load(a + i) * Vector.Load(b + i);
            }
            sum = Vector.Sum(acc);
        }
        for (; i < n; i++) sum += a[i] * b[i];
        return sum;
    }
}
