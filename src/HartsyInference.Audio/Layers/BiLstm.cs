using System.Runtime.InteropServices;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Layers;

/// <summary>Single-layer bidirectional LSTM. Mirrors PyTorch's
/// <c>nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)</c> with
/// <c>num_layers=1</c>. Used by Kokoro's <c>KokoroTextEncoder</c> (after Conv1D × 3) and
/// the prosody-predictor duration encoder, plus GPT-SoVITS and the VITS-derived TTS
/// family.
///
/// <para>Input is <c>[B, T, input_dim]</c> channels-last. The forward LSTM runs left-to-
/// right; the backward LSTM runs right-to-left over the same input. At each timestep the
/// output concatenates the two hidden states along the last dim, so the result is
/// <c>[B, T, 2*hidden]</c>.</para>
///
/// <para>Execution split: the input projection of BOTH directions is one backend GEMM over the whole
/// sequence (<c>[B, T, in] → [B, T, 8·hidden]</c>, the two <c>W_ih</c> stacked at load time), read back to
/// the host once; the sequential recurrence then runs on the host (<see cref="LstmOps.RunSequence"/>), the
/// two directions through <see cref="CpuParallel"/>, and the result is handed back as one host tensor — one
/// device sync per layer instead of one per timestep.</para>
///
/// <para>Weight key convention (PyTorch state dict):
/// <code>
///   {prefix}.weight_ih_l0          # forward input-to-hidden    [4*hidden, input]
///   {prefix}.weight_hh_l0          # forward hidden-to-hidden   [4*hidden, hidden]
///   {prefix}.bias_ih_l0            # [4*hidden]
///   {prefix}.bias_hh_l0            # [4*hidden]
///   {prefix}.weight_ih_l0_reverse  # backward direction
///   {prefix}.weight_hh_l0_reverse
///   {prefix}.bias_ih_l0_reverse
///   {prefix}.bias_hh_l0_reverse
/// </code></para>
///
/// <para>For larger stacked LSTMs (<c>num_layers &gt; 1</c>) wrap multiple <see cref="BiLstm"/>
/// instances and feed each one's output into the next. The official models we need to
/// support all use <c>num_layers=1</c>, so no built-in stacking is provided.</para></summary>
internal sealed unsafe class BiLstm
{
    public int InputDim { get; }
    public int HiddenDim { get; }

    private readonly LstmCell _fwd;
    private readonly LstmCell _bwd;
    // Both directions' input projections stacked: [8*hidden, input] / [8*hidden], forward rows first.
    private Tensor? _wIhStacked;
    private Tensor? _bIhStacked;

    public BiLstm(int inputDim, int hiddenDim)
    {
        InputDim = inputDim;
        HiddenDim = hiddenDim;
        _fwd = new LstmCell(inputDim, hiddenDim);
        _bwd = new LstmCell(inputDim, hiddenDim);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _fwd.BindWeights(
            WhisperOps.EnsureF32(w[$"{prefix}.weight_ih_l0"]),
            WhisperOps.EnsureF32(w[$"{prefix}.weight_hh_l0"]),
            WhisperOps.EnsureF32(w[$"{prefix}.bias_ih_l0"]),
            WhisperOps.EnsureF32(w[$"{prefix}.bias_hh_l0"]));
        _bwd.BindWeights(
            WhisperOps.EnsureF32(w[$"{prefix}.weight_ih_l0_reverse"]),
            WhisperOps.EnsureF32(w[$"{prefix}.weight_hh_l0_reverse"]),
            WhisperOps.EnsureF32(w[$"{prefix}.bias_ih_l0_reverse"]),
            WhisperOps.EnsureF32(w[$"{prefix}.bias_hh_l0_reverse"]));
        _wIhStacked = StackRows(_fwd.WeightIh!, _bwd.WeightIh!);
        _bIhStacked = StackRows(_fwd.BiasIh!, _bwd.BiasIh!);
    }

    /// <summary>Bidirectional sweep over <paramref name="x"/> <c>[B, T, input_dim]</c>.
    /// Returns a fresh <c>[B, T, 2*hidden]</c> tensor with the forward-direction hidden
    /// states in <c>[..., 0:hidden]</c> and the backward-direction hidden states in
    /// <c>[..., hidden:2*hidden]</c>. Initial <c>(h, c)</c> are zero per PyTorch default.</summary>
    public Tensor Forward(IBackend backend, Tensor x, int batch, int t)
    {
        if (x.Shape.Rank != 3 || (int)x.Shape[0] != batch || (int)x.Shape[1] != t || (int)x.Shape[2] != InputDim)
            throw new ArgumentException($"BiLstm input must be [{batch}, {t}, {InputDim}], got {x.Shape}.", nameof(x));
        if (_wIhStacked is null || _bIhStacked is null)
            throw new InvalidOperationException("BiLstm weights not loaded.");
        int hidden = HiddenDim;
        int hidden4 = 4 * hidden;
        int hidden8 = 8 * hidden;

        // 1. Both directions' input projections in one GEMM: [B, T, in] → [B, T, 8*hidden].
        Tensor gatesIn = WhisperOps.ProjectLinear(backend, x, _wIhStacked, _bIhStacked, batch, t, InputDim, hidden8);
        Tensor output = new(new TensorShape(batch, t, 2 * hidden), DType.F32);
        try
        {
            // 2. Host recurrence. This read is the single device→host sync of the layer.
            float* gp = (float*)gatesIn.DataPointer;
            float* op = (float*)output.DataPointer;
            float* wHhF = (float*)_fwd.WeightHh!.DataPointer;
            float* bHhF = (float*)_fwd.BiasHh!.DataPointer;
            float* wHhB = (float*)_bwd.WeightHh!.DataPointer;
            float* bHhB = (float*)_bwd.BiasHh!.DataPointer;
            int scratchFloats = 2 * hidden + hidden4;
            float* scratch = (float*)NativeMemory.AlignedAlloc((nuint)(2 * scratchFloats * sizeof(float)), 64);
            try
            {
                // Two gate rows of hidden multiply-adds per hidden unit per step, both directions.
                long work = 2L * t * hidden4 * hidden * 2;
                for (int b = 0; b < batch; b++)
                {
                    float* gIn = gp + (long)b * t * hidden8;
                    float* outB = op + (long)b * t * 2 * hidden;
                    // The directions write disjoint halves of each output row and own their scratch halves.
                    CpuParallel.For(2, work, d =>
                    {
                        float* s = scratch + d * scratchFloats;
                        LstmOps.RunSequence(gIn + d * hidden4, hidden8, d == 0 ? wHhF : wHhB, d == 0 ? bHhF : bHhB, t, hidden,
                            reverse: d == 1, outB, 2 * hidden, d * hidden, s, s + hidden, s + 2 * hidden);
                    });
                }
            }
            finally
            {
                NativeMemory.AlignedFree(scratch);
            }
        }
        finally
        {
            gatesIn.Dispose();
        }
        return output;
    }

    /// <summary>The device-side weights: the stacked input projection. The hidden-to-hidden weights and biases are
    /// read by the host recurrence only and are deliberately kept off the device.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_wIhStacked is not null) yield return _wIhStacked;
        if (_bIhStacked is not null) yield return _bIhStacked;
    }

    /// <summary>Concatenates two F32 tensors along their leading dim: <c>[a; b]</c>.</summary>
    private static Tensor StackRows(Tensor a, Tensor b)
    {
        if (a.DType != DType.F32 || b.DType != DType.F32 || a.Shape.Rank != b.Shape.Rank)
            throw new ArgumentException("StackRows expects two F32 tensors of the same rank.");
        long[] dims = new long[a.Shape.Rank];
        for (int i = 0; i < dims.Length; i++)
        {
            if (i > 0 && a.Shape[i] != b.Shape[i])
                throw new ArgumentException($"StackRows trailing dims differ: {a.Shape} vs {b.Shape}.");
            dims[i] = i == 0 ? a.Shape[0] + b.Shape[0] : a.Shape[i];
        }
        Tensor stacked = new(new TensorShape(dims), DType.F32);
        a.AsSpan<float>().CopyTo(stacked.AsSpan<float>());
        b.AsSpan<float>().CopyTo(stacked.AsSpan<float>()[(int)a.ElementCount..]);
        return stacked;
    }
}
