using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Layers;

/// <summary>Single-layer unidirectional GRU. Mirrors PyTorch's <c>nn.GRU(input_size, hidden_size,
/// batch_first=True)</c> with <c>num_layers=1, bidirectional=False</c>. Input is <c>[B, T, input_dim]</c>;
/// <see cref="Forward"/> returns the full output sequence <c>[B, T, hidden]</c> and <see cref="LastHidden"/>
/// returns just the final timestep <c>[B, hidden]</c> (what reference encoders use). Initial <c>h</c> is zero
/// per PyTorch default.
///
/// <para>Weight key convention (PyTorch state dict): <c>{prefix}.weight_ih_l0</c> <c>[3*hidden, input]</c>,
/// <c>{prefix}.weight_hh_l0</c> <c>[3*hidden, hidden]</c>, <c>{prefix}.bias_ih_l0</c>, <c>{prefix}.bias_hh_l0</c>
/// <c>[3*hidden]</c>. Used by OpenVoice's tone-color <c>ReferenceEncoder</c>; reusable by any GRU-based head.</para></summary>
internal sealed unsafe class Gru
{
    public int InputDim { get; }
    public int HiddenDim { get; }

    private readonly GruCell _cell;

    public Gru(int inputDim, int hiddenDim)
    {
        InputDim = inputDim;
        HiddenDim = hiddenDim;
        _cell = new GruCell(inputDim, hiddenDim);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _cell.BindWeights(
            WhisperOps.EnsureF32(w[$"{prefix}.weight_ih_l0"]),
            WhisperOps.EnsureF32(w[$"{prefix}.weight_hh_l0"]),
            WhisperOps.EnsureF32(w[$"{prefix}.bias_ih_l0"]),
            WhisperOps.EnsureF32(w[$"{prefix}.bias_hh_l0"]));
    }

    /// <summary>Runs the sequence and returns the final-timestep hidden state <c>[batch, hidden]</c>. Caller
    /// owns the returned tensor. Cheaper than <see cref="Forward"/> when only the summary vector is needed.</summary>
    public Tensor LastHidden(IBackend backend, Tensor x, int batch, int t)
    {
        ValidateInput(x, batch, t);
        float* xPtr = (float*)x.DataPointer;
        Tensor stepIn = new(new TensorShape(batch, InputDim), DType.F32);
        Tensor h = ZeroAllocate(batch, HiddenDim);
        try
        {
            for (int step = 0; step < t; step++)
            {
                LoadTimestep(xPtr, stepIn, batch, t, InputDim, step);
                Tensor hNew = _cell.Step(backend, stepIn, h, batch);
                h.Dispose();
                h = hNew;
            }
        }
        catch
        {
            stepIn.Dispose();
            h.Dispose();
            throw;
        }
        stepIn.Dispose();
        return h;
    }

    /// <summary>Runs the sequence and returns the full output <c>[batch, t, hidden]</c> (every timestep's
    /// hidden state). Initial hidden is zero.</summary>
    public Tensor Forward(IBackend backend, Tensor x, int batch, int t)
    {
        ValidateInput(x, batch, t);
        Tensor output = new(new TensorShape(batch, t, HiddenDim), DType.F32);
        float* outPtr = (float*)output.DataPointer;
        float* xPtr = (float*)x.DataPointer;
        Tensor stepIn = new(new TensorShape(batch, InputDim), DType.F32);
        Tensor h = ZeroAllocate(batch, HiddenDim);
        try
        {
            for (int step = 0; step < t; step++)
            {
                LoadTimestep(xPtr, stepIn, batch, t, InputDim, step);
                Tensor hNew = _cell.Step(backend, stepIn, h, batch);
                h.Dispose();
                h = hNew;
                StoreTimestep(outPtr, h, batch, t, HiddenDim, step);
            }
        }
        finally
        {
            stepIn.Dispose();
            h.Dispose();
        }
        return output;
    }

    public IEnumerable<Tensor> EnumerateWeights() => _cell.EnumerateWeights();

    private void ValidateInput(Tensor x, int batch, int t)
    {
        if (x.Shape.Rank != 3 || (int)x.Shape[0] != batch || (int)x.Shape[1] != t || (int)x.Shape[2] != InputDim)
            throw new ArgumentException($"Gru input must be [{batch}, {t}, {InputDim}], got {x.Shape}.", nameof(x));
    }

    private static Tensor ZeroAllocate(int batch, int dim)
    {
        Tensor t = new(new TensorShape(batch, dim), DType.F32);
        long n = t.ElementCount;
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < n; i++) p[i] = 0f;
        return t;
    }

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

    private static void StoreTimestep(float* outPtr, Tensor h, int batch, int t, int hidden, int step)
    {
        float* sp = (float*)h.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            int srcBase = b * hidden;
            int dstBase = (b * t + step) * hidden;
            for (int k = 0; k < hidden; k++) outPtr[dstBase + k] = sp[srcBase + k];
        }
    }
}
