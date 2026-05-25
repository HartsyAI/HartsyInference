using SharpInference.Audio.Models.Whisper;
using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Layers;

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
    }

    /// <summary>Bidirectional sweep over <paramref name="x"/> <c>[B, T, input_dim]</c>.
    /// Returns a fresh <c>[B, T, 2*hidden]</c> tensor with the forward-direction hidden
    /// states in <c>[..., 0:hidden]</c> and the backward-direction hidden states in
    /// <c>[..., hidden:2*hidden]</c>. Initial <c>(h, c)</c> are zero per PyTorch default.</summary>
    public Tensor Forward(IBackend backend, Tensor x, int batch, int t)
    {
        if (x.Shape.Rank != 3 || (int)x.Shape[0] != batch || (int)x.Shape[1] != t || (int)x.Shape[2] != InputDim)
            throw new ArgumentException($"BiLstm input must be [{batch}, {t}, {InputDim}], got {x.Shape}.", nameof(x));

        Tensor output = new(new TensorShape(batch, t, 2 * HiddenDim), DType.F32);
        float* outPtr = (float*)output.DataPointer;
        float* xPtr = (float*)x.DataPointer;

        // Reusable per-step buffers.
        Tensor stepIn = new(new TensorShape(batch, InputDim), DType.F32);
        Tensor hFwd = ZeroAllocate(batch, HiddenDim);
        Tensor cFwd = ZeroAllocate(batch, HiddenDim);
        Tensor hBwd = ZeroAllocate(batch, HiddenDim);
        Tensor cBwd = ZeroAllocate(batch, HiddenDim);

        try
        {
            // Forward sweep: t = 0..T-1.
            for (int step = 0; step < t; step++)
            {
                LoadTimestep(xPtr, stepIn, batch, t, InputDim, step);
                (Tensor hNew, Tensor cNew) = _fwd.Step(backend, stepIn, hFwd, cFwd, batch);
                hFwd.Dispose();
                cFwd.Dispose();
                hFwd = hNew;
                cFwd = cNew;
                StoreOutputHalf(outPtr, hFwd, batch, t, HiddenDim, step, leftHalf: true);
            }

            // Backward sweep: t = T-1..0.
            for (int step = t - 1; step >= 0; step--)
            {
                LoadTimestep(xPtr, stepIn, batch, t, InputDim, step);
                (Tensor hNew, Tensor cNew) = _bwd.Step(backend, stepIn, hBwd, cBwd, batch);
                hBwd.Dispose();
                cBwd.Dispose();
                hBwd = hNew;
                cBwd = cNew;
                StoreOutputHalf(outPtr, hBwd, batch, t, HiddenDim, step, leftHalf: false);
            }
        }
        finally
        {
            stepIn.Dispose();
            hFwd.Dispose();
            cFwd.Dispose();
            hBwd.Dispose();
            cBwd.Dispose();
        }

        return output;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _fwd.EnumerateWeights()) yield return t;
        foreach (Tensor t in _bwd.EnumerateWeights()) yield return t;
    }

    private static Tensor ZeroAllocate(int batch, int dim)
    {
        Tensor t = new(new TensorShape(batch, dim), DType.F32);
        long n = t.ElementCount;
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < n; i++) p[i] = 0f;
        return t;
    }

    /// <summary>Copies <c>x[:, step, :]</c> from a <c>[B, T, D]</c> input into a
    /// <c>[B, D]</c> step buffer. For each batch row the stride is <c>D</c> floats —
    /// a single contiguous span per batch entry.</summary>
    private static void LoadTimestep(float* xPtr, Tensor stepIn, int batch, int t, int dim, int step)
    {
        float* dp = (float*)stepIn.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            int srcBase = (b * t + step) * dim;
            int dstBase = b * dim;
            for (int k = 0; k < dim; k++) dp[dstBase + k] = xPtr[srcBase + k];
        }
    }

    /// <summary>Writes a <c>[B, hidden]</c> step output into either the left half
    /// <c>[:, step, 0:hidden]</c> or the right half <c>[:, step, hidden:2*hidden]</c> of
    /// the bidirectional output <c>[B, T, 2*hidden]</c>.</summary>
    private static void StoreOutputHalf(float* outPtr, Tensor h, int batch, int t, int hidden, int step, bool leftHalf)
    {
        float* sp = (float*)h.DataPointer;
        int wide = 2 * hidden;
        int colOffset = leftHalf ? 0 : hidden;
        for (int b = 0; b < batch; b++)
        {
            int srcBase = b * hidden;
            int dstBase = (b * t + step) * wide + colOffset;
            for (int k = 0; k < hidden; k++) outPtr[dstBase + k] = sp[srcBase + k];
        }
    }
}
