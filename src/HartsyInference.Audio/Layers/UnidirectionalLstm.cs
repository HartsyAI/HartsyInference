using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Layers;

/// <summary>Multi-layer unidirectional LSTM that processes a full <c>[B, T, input_dim]</c>
/// sequence and emits <c>[B, T, hidden_dim]</c> — the final layer's hidden states for
/// every timestep. Used in EnCodec's bottleneck (2-layer unidir LSTM between the SEANet
/// encoder and the projection to latent space) and any future codec that adds a
/// recurrent bottleneck.
///
/// <para>Stacked LSTM topology (PyTorch <c>nn.LSTM(num_layers=N, bidirectional=False)</c>):</para>
/// <list type="bullet">
///   <item>Layer 0 input: x[t] (input_dim)</item>
///   <item>Layer 0 output: h0[t] (hidden_dim)</item>
///   <item>Layer 1 input: h0[t] (hidden_dim, matches hidden_size from layer 0)</item>
///   <item>Layer 1 output: h1[t] (hidden_dim)</item>
///   <item>...continues for layers 2..N-1</item>
/// </list>
///
/// <para>Initial <c>(h, c)</c> states are zero per PyTorch default. For bidirectional
/// versions use <see cref="BiLstm"/> (one-layer); stacked-bidirectional isn't needed by
/// any codec in scope.</para></summary>
internal sealed unsafe class UnidirectionalLstm
{
    public int InputDim { get; }
    public int HiddenDim { get; }
    public int NumLayers { get; }

    private readonly LstmCell[] _layers;

    public UnidirectionalLstm(int inputDim, int hiddenDim, int numLayers)
    {
        if (numLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numLayers));
        InputDim = inputDim;
        HiddenDim = hiddenDim;
        NumLayers = numLayers;
        _layers = new LstmCell[numLayers];
        // Layer 0 reads input_dim; subsequent layers read hidden_dim from the layer below.
        _layers[0] = new LstmCell(inputDim, hiddenDim);
        for (int i = 1; i < numLayers; i++)
            _layers[i] = new LstmCell(hiddenDim, hiddenDim);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        for (int i = 0; i < NumLayers; i++)
        {
            _layers[i].BindWeights(
                WhisperOps.EnsureF32(w[$"{prefix}.weight_ih_l{i}"]),
                WhisperOps.EnsureF32(w[$"{prefix}.weight_hh_l{i}"]),
                WhisperOps.EnsureF32(w[$"{prefix}.bias_ih_l{i}"]),
                WhisperOps.EnsureF32(w[$"{prefix}.bias_hh_l{i}"]));
        }
    }

    /// <summary>Runs the stacked LSTM over <paramref name="x"/> <c>[batch, T, input_dim]</c>.
    /// Returns a fresh <c>[batch, T, hidden_dim]</c> tensor; caller owns disposal.</summary>
    public Tensor Forward(IBackend backend, Tensor x, int batch, int t)
    {
        if (x.Shape.Rank != 3 || (int)x.Shape[0] != batch || (int)x.Shape[1] != t || (int)x.Shape[2] != InputDim)
            throw new ArgumentException($"UnidirectionalLstm input must be [{batch}, {t}, {InputDim}], got {x.Shape}.", nameof(x));

        // current[t] holds the input to the current layer's step. For layer 0 it's x;
        // for later layers it's the previous layer's output sequence stored in
        // a working tensor we allocate per layer.
        Tensor currentSeq = x;
        bool ownsCurrent = false;

        // Per-step input buffer reused across all timesteps of the current layer.
        // Allocate once at the max dim (max of InputDim and HiddenDim).
        int maxDim = Math.Max(InputDim, HiddenDim);
        Tensor stepIn = new(new TensorShape(batch, maxDim), DType.F32);

        try
        {
            for (int layer = 0; layer < NumLayers; layer++)
            {
                int dimIn = layer == 0 ? InputDim : HiddenDim;
                Tensor h = ZeroAllocate(batch, HiddenDim);
                Tensor c = ZeroAllocate(batch, HiddenDim);
                Tensor nextSeq = new(new TensorShape(batch, t, HiddenDim), DType.F32);

                float* srcSeqPtr = (float*)currentSeq.DataPointer;
                float* dstSeqPtr = (float*)nextSeq.DataPointer;

                for (int step = 0; step < t; step++)
                {
                    // Copy current-layer input at this timestep into stepIn.
                    float* dp = (float*)stepIn.DataPointer;
                    for (int b = 0; b < batch; b++)
                    {
                        int srcBase = (b * t + step) * dimIn;
                        int dstBase = b * dimIn;
                        for (int k = 0; k < dimIn; k++) dp[dstBase + k] = srcSeqPtr[srcBase + k];
                    }

                    // Reshape stepIn view to [B, dimIn] for the LSTM cell.
                    Tensor stepInView = stepIn.Reshape(new TensorShape(batch, dimIn));

                    (Tensor hNew, Tensor cNew) = _layers[layer].Step(backend, stepInView, h, c, batch);
                    h.Dispose();
                    c.Dispose();
                    h = hNew;
                    c = cNew;

                    // Copy h into nextSeq at [:, step, :].
                    float* hp = (float*)h.DataPointer;
                    for (int b = 0; b < batch; b++)
                    {
                        int srcBase = b * HiddenDim;
                        int dstBase = (b * t + step) * HiddenDim;
                        for (int k = 0; k < HiddenDim; k++) dstSeqPtr[dstBase + k] = hp[srcBase + k];
                    }
                }

                h.Dispose();
                c.Dispose();
                if (ownsCurrent) currentSeq.Dispose();
                currentSeq = nextSeq;
                ownsCurrent = true;
            }
        }
        finally
        {
            stepIn.Dispose();
        }

        return ownsCurrent ? currentSeq : x;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (LstmCell cell in _layers)
            foreach (Tensor t in cell.EnumerateWeights()) yield return t;
    }

    private static Tensor ZeroAllocate(int batch, int dim)
    {
        Tensor t = new(new TensorShape(batch, dim), DType.F32);
        long n = t.ElementCount;
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < n; i++) p[i] = 0f;
        return t;
    }
}
