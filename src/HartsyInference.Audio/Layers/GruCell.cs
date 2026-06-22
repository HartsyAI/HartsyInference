using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Layers;

/// <summary>One step of a vanilla PyTorch <c>nn.GRUCell</c>. Stateless — the caller owns <c>h_prev</c> across
/// timesteps. Mirrors PyTorch's gate order (r, z, n) and the reset-gate-on-hidden convention exactly so
/// safetensors / <c>.pth</c> weights load without remapping:
/// <code>
///   gi = x @ W_ih.T + b_ih          # [B, 3*hidden]  (r, z, n)
///   gh = h_prev @ W_hh.T + b_hh      # [B, 3*hidden]
///   r = sigmoid(gi_r + gh_r)
///   z = sigmoid(gi_z + gh_z)
///   n = tanh(gi_n + r * gh_n)        # reset gate multiplies the hidden contribution, not the sum
///   h_new = (1 - z) * n + z * h_prev
/// </code>
///
/// <para>Unlike LSTM, the input/hidden contributions can't be summed before activation (the reset gate
/// scales only the hidden side of the new-gate), so <c>gi</c> and <c>gh</c> are kept separate. Wrapped by
/// <see cref="Gru"/> for a full unidirectional sequence pass (e.g. OpenVoice's <c>ReferenceEncoder</c>).
/// Weights use the standard <c>weight_ih_l0</c> / <c>weight_hh_l0</c> / <c>bias_ih_l0</c> / <c>bias_hh_l0</c>
/// naming, tensors shaped <c>[3*hidden, input/hidden]</c>, handed straight to
/// <see cref="WhisperOps.ProjectLinear"/> which transposes and bias-adds internally.</para></summary>
internal sealed class GruCell
{
    public int InputDim { get; }
    public int HiddenDim { get; }

    private Tensor? _wIh;       // [3*hidden, input]
    private Tensor? _wHh;       // [3*hidden, hidden]
    private Tensor? _bIh;       // [3*hidden]
    private Tensor? _bHh;       // [3*hidden]

    public GruCell(int inputDim, int hiddenDim)
    {
        if (inputDim <= 0) throw new ArgumentOutOfRangeException(nameof(inputDim));
        if (hiddenDim <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenDim));
        InputDim = inputDim;
        HiddenDim = hiddenDim;
    }

    /// <summary>Binds the four weight tensors (already F32). Caller resolves the state-dict key paths
    /// (typically via <see cref="Gru.LoadWeights"/>).</summary>
    public void BindWeights(Tensor weightIh, Tensor weightHh, Tensor biasIh, Tensor biasHh)
    {
        _wIh = weightIh;
        _wHh = weightHh;
        _bIh = biasIh;
        _bHh = biasHh;
    }

    /// <summary>One GRU step. Inputs are rank-2 <c>[batch, dim]</c>; the returned <c>h_new</c> is a freshly
    /// allocated <c>[batch, hidden]</c> the caller owns.</summary>
    public Tensor Step(IBackend backend, Tensor x, Tensor hPrev, int batch)
    {
        if (_wIh is null || _wHh is null || _bIh is null || _bHh is null)
            throw new InvalidOperationException("GruCell weights not bound.");
        int hidden3 = 3 * HiddenDim;

        Tensor x3 = x.Reshape(new TensorShape(1, batch, InputDim));
        Tensor hPrev3 = hPrev.Reshape(new TensorShape(1, batch, HiddenDim));

        Tensor gi = WhisperOps.ProjectLinear(backend, x3, _wIh, _bIh, 1, batch, InputDim, hidden3);
        Tensor gh = WhisperOps.ProjectLinear(backend, hPrev3, _wHh, _bHh, 1, batch, HiddenDim, hidden3);

        Tensor hNew = new(new TensorShape(batch, HiddenDim), DType.F32);
        GruOps.GateAndUpdate(gi, gh, hPrev, hNew, batch, HiddenDim);
        gi.Dispose();
        gh.Dispose();
        return hNew;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_wIh, _wHh, _bIh, _bHh];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }
}

/// <summary>Fused GRU gate-and-update step. Combines the three gate activations and the hidden update into one
/// sweep over the <c>[B, 3*hidden]</c> input/hidden gate matrices, avoiding intermediate gate/activation
/// tensors. Gate order follows PyTorch: first <c>hidden</c> columns are reset (r), next update (z), last
/// new (n).</summary>
internal static unsafe class GruOps
{
    public static void GateAndUpdate(Tensor gi, Tensor gh, Tensor hPrev, Tensor hNew, int batch, int hidden)
    {
        int hidden3 = 3 * hidden;
        float* gip = (float*)gi.DataPointer;
        float* ghp = (float*)gh.DataPointer;
        float* h0 = (float*)hPrev.DataPointer;
        float* hOut = (float*)hNew.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            int gateRow = b * hidden3;
            int outRow = b * hidden;
            for (int k = 0; k < hidden; k++)
            {
                float r = SigmoidS(gip[gateRow + k] + ghp[gateRow + k]);
                float z = SigmoidS(gip[gateRow + hidden + k] + ghp[gateRow + hidden + k]);
                float n = MathF.Tanh(gip[gateRow + 2 * hidden + k] + r * ghp[gateRow + 2 * hidden + k]);
                float hp = h0[outRow + k];
                hOut[outRow + k] = (1f - z) * n + z * hp;
            }
        }
    }

    private static float SigmoidS(float x)
    {
        if (x >= 0f) return 1f / (1f + MathF.Exp(-x));
        float ex = MathF.Exp(x);
        return ex / (1f + ex);
    }
}
