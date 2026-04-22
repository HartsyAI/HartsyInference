using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Feed-forward network supporting SwiGLU (w2(silu(w1(x)) * w3(x))) and GELU-approximate (linear(gelu(linear(x)))) modes. Mode is set at weight loading time to support both Stability AI and HuggingFace diffusers weight formats.</summary>
public sealed unsafe class SwiGluFfn
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

    /// <summary>Loads weights for SwiGLU mode (3 projections): w1 = gate, w3 = linear, w2 = output.</summary>
    public void LoadSwiGluWeights(Tensor w1Weight, Tensor w1Bias, Tensor w3Weight, Tensor w3Bias, Tensor w2Weight, Tensor w2Bias)
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
        Tensor gate = LinearProjection(input, _w1Weight!, _w1Bias!, batch, seqLen, _hiddenSize, _ffDim);
        Tensor gateActivated = new Tensor(ffShape, DType.F32);
        backend.Silu(gateActivated, gate);
        gate.Dispose();

        // linear = input @ w3^T + b3
        Tensor linear = LinearProjection(input, _w3Weight!, _w3Bias!, batch, seqLen, _hiddenSize, _ffDim);

        // gated = silu(gate) * linear
        Tensor gated = new Tensor(ffShape, DType.F32);
        backend.Mul(gated, gateActivated, linear);
        gateActivated.Dispose();
        linear.Dispose();

        // output = gated @ w2^T + b2
        Tensor output = LinearProjection(gated, _w2Weight!, _w2Bias!, batch, seqLen, _ffDim, _hiddenSize);
        gated.Dispose();

        return output;
    }

    private Tensor ForwardGelu(IBackend backend, Tensor input, int batch, int seqLen)
    {
        TensorShape ffShape = new TensorShape(batch, seqLen, _ffDim);

        // proj = gelu(input @ w1^T + b1)
        Tensor proj = LinearProjection(input, _w1Weight!, _w1Bias!, batch, seqLen, _hiddenSize, _ffDim);
        Tensor activated = new Tensor(ffShape, DType.F32);
        backend.Gelu(activated, proj);
        proj.Dispose();

        // output = activated @ w2^T + b2
        Tensor output = LinearProjection(activated, _w2Weight!, _w2Bias!, batch, seqLen, _ffDim, _hiddenSize);
        activated.Dispose();

        return output;
    }

    private static Tensor LinearProjection(Tensor input, Tensor weight, Tensor bias, int batch, int seqLen, int inDim, int outDim)
    {
        TensorShape outShape = new TensorShape(batch, seqLen, outDim);
        Tensor output = new Tensor(outShape, DType.F32);

        float* inPtr = (float*)input.DataPointer;
        float* wPtr = (float*)weight.DataPointer;
        float* bPtr = (float*)bias.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int inOffset = (b * seqLen + s) * inDim;
                int outOffset = (b * seqLen + s) * outDim;
                for (int o = 0; o < outDim; o++)
                {
                    float sum = bPtr[o];
                    int wOffset = o * inDim;
                    for (int i = 0; i < inDim; i++)
                    {
                        sum += inPtr[inOffset + i] * wPtr[wOffset + i];
                    }
                    outPtr[outOffset + o] = sum;
                }
            }
        }

        return output;
    }
}
