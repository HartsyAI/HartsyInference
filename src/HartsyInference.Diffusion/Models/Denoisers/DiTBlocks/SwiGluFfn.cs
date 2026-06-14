using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Feed-forward network supporting SwiGLU (w2(silu(w1(x)) * w3(x))) and GELU-approximate (linear(gelu(linear(x)))) modes. Mode is set at weight loading time to support both Stability AI and HuggingFace diffusers weight formats.</summary>
public sealed class SwiGluFfn
{
    private readonly int _hiddenSize;
    private readonly int _ffDim;
    private bool _useGeluMode;

    // SwiGLU mode: w1 = gate (SiLU), w3 = linear, w2 = output
    // GELU mode: w1 = projection (GELU), w2 = output; w3 unused
    private Tensor? _w1Weight, _w1Bias;
    private Tensor? _w2Weight, _w2Bias;
    private Tensor? _w3Weight, _w3Bias;

    /// <summary>Creates a feed-forward block with the given dimensions.</summary>
    /// <param name="hiddenSize">Model hidden dimension.</param>
    /// <param name="ffDim">Feed-forward inner dimension (typically 4 * hiddenSize).</param>
    public SwiGluFfn(int hiddenSize, int ffDim)
    {
        _hiddenSize = hiddenSize;
        _ffDim = ffDim;
    }

    /// <summary>Loads weights for SwiGLU mode (3 projections): w1 = gate, w3 = linear, w2 = output. Biases may be null when the model has bias-less linears (Flux.2).</summary>
    public void LoadSwiGluWeights(Tensor w1Weight, Tensor? w1Bias, Tensor w3Weight, Tensor? w3Bias, Tensor w2Weight, Tensor? w2Bias)
    {
        _w1Weight = w1Weight;
        _w1Bias = w1Bias;
        _w3Weight = w3Weight;
        _w3Bias = w3Bias;
        _w2Weight = w2Weight;
        _w2Bias = w2Bias;
        _useGeluMode = false;
    }

    /// <summary>Loads weights for GELU-approximate mode (2 projections): proj + output. Used by HuggingFace diffusers format.</summary>
    public void LoadGeluWeights(Tensor projWeight, Tensor projBias, Tensor outWeight, Tensor outBias)
    {
        _w1Weight = projWeight;
        _w1Bias = projBias;
        _w2Weight = outWeight;
        _w2Bias = outBias;
        _useGeluMode = true;
    }

    /// <summary>Enumerates all weight tensors for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_w1Weight is not null) yield return _w1Weight;
        if (_w1Bias is not null) yield return _w1Bias;
        if (_w2Weight is not null) yield return _w2Weight;
        if (_w2Bias is not null) yield return _w2Bias;
        if (_w3Weight is not null) yield return _w3Weight;
        if (_w3Bias is not null) yield return _w3Bias;
    }

    /// <summary>Forward pass. Input: [B, seqLen, hiddenSize] → Output: [B, seqLen, hiddenSize].</summary>
    public Tensor Forward(IBackend backend, Tensor input, int batch, int seqLen)
    {
        if (_useGeluMode)
            return ForwardGelu(backend, input, batch, seqLen);
        return ForwardSwiGlu(backend, input, batch, seqLen);
    }

    private Tensor ForwardSwiGlu(IBackend backend, Tensor input, int batch, int seqLen)
    {
        TensorShape ffShape = new TensorShape(batch, seqLen, _ffDim);

        // gate = silu(input @ w1^T + b1)
        Tensor gate = new Tensor(ffShape, input.DType);
        backend.Linear(gate, input, _w1Weight!, _w1Bias);
        Tensor gateActivated = new Tensor(ffShape, input.DType);
        backend.Silu(gateActivated, gate);
        gate.Dispose();

        // linear = input @ w3^T + b3
        Tensor linear = new Tensor(ffShape, input.DType);
        backend.Linear(linear, input, _w3Weight!, _w3Bias);

        // gated = silu(gate) * linear
        Tensor gated = new Tensor(ffShape, input.DType);
        backend.Mul(gated, gateActivated, linear);
        gateActivated.Dispose();
        linear.Dispose();

        // output = gated @ w2^T + b2
        TensorShape outShape = new TensorShape(batch, seqLen, _hiddenSize);
        Tensor output = new Tensor(outShape, input.DType);
        backend.Linear(output, gated, _w2Weight!, _w2Bias);
        gated.Dispose();

        return output;
    }

    private Tensor ForwardGelu(IBackend backend, Tensor input, int batch, int seqLen)
    {
        TensorShape ffShape = new TensorShape(batch, seqLen, _ffDim);

        // proj = gelu(input @ w1^T + b1)
        Tensor proj = new Tensor(ffShape, input.DType);
        backend.Linear(proj, input, _w1Weight!, _w1Bias);
        Tensor activated = new Tensor(ffShape, input.DType);
        backend.Gelu(activated, proj);
        proj.Dispose();

        // output = activated @ w2^T + b2
        TensorShape outShape = new TensorShape(batch, seqLen, _hiddenSize);
        Tensor output = new Tensor(outShape, input.DType);
        backend.Linear(output, activated, _w2Weight!, _w2Bias);
        activated.Dispose();

        return output;
    }
}
