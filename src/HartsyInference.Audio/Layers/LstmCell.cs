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
            for (int k = 0; k < hidden; k++)
            {
                float iGate = SigmoidS(g[gateRow + k]);
                float fGate = SigmoidS(g[gateRow + hidden + k]);
                float gGate = MathF.Tanh(g[gateRow + 2 * hidden + k]);
                float oGate = SigmoidS(g[gateRow + 3 * hidden + k]);

                float cNext = fGate * c0[outRow + k] + iGate * gGate;
                cOut[outRow + k] = cNext;
                hOut[outRow + k] = oGate * MathF.Tanh(cNext);
            }
        }
    }

    private static float SigmoidS(float x)
    {
        // Sign-aware to avoid Inf in exp.
        if (x >= 0f) return 1f / (1f + MathF.Exp(-x));
        float ex = MathF.Exp(x);
        return ex / (1f + ex);
    }
}
