using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>AdaLN-Zero modulation for SD3 JointBlocks. Takes timestep embedding, produces shift/scale/gate parameters via SiLU + Linear. SD3 uses 6 params per image sub-block (attn: shift, scale, gate; mlp: shift, scale, gate) and 3 per context sub-block (attn only: shift, scale, gate).</summary>
public sealed unsafe class AdaLNModulation
{
    private readonly int _hiddenSize;
    private readonly int _numParams;

    private Tensor? _linearWeight;
    private Tensor? _linearBias;

    /// <summary>Creates an AdaLN-Zero modulation layer.</summary>
    /// <param name="hiddenSize">Model hidden dimension.</param>
    /// <param name="numParams">Number of modulation parameters to produce (6 for image block, 3 for context block).</param>
    public AdaLNModulation(int hiddenSize, int numParams)
    {
        _hiddenSize = hiddenSize;
        _numParams = numParams;
    }

    /// <summary>Loads the linear projection weights. Weight shape: [numParams * hiddenSize, hiddenSize], Bias shape: [numParams * hiddenSize].</summary>
    public void LoadWeights(Tensor weight, Tensor bias)
    {
        _linearWeight = weight;
        _linearBias = bias;
    }

    /// <summary>Computes modulation parameters from timestep embedding. Input: [B, hiddenSize] → Output: numParams tensors each [B, hiddenSize].</summary>
    public Tensor[] Forward(IBackend backend, Tensor timestepEmb)
    {
        int batch = (int)timestepEmb.Shape[0];
        int outDim = _numParams * _hiddenSize;

        // SiLU activation on input
        TensorShape inputShape = new TensorShape(batch, _hiddenSize);
        Tensor activated = new Tensor(inputShape, DType.F32);
        backend.Silu(activated, timestepEmb);

        // Linear projection: [B, hiddenSize] → [B, numParams * hiddenSize]
        TensorShape outShape = new TensorShape(batch, outDim);
        Tensor projected = new Tensor(outShape, DType.F32);
        LinearProjection(projected, activated, _linearWeight!, _linearBias!, batch, _hiddenSize, outDim);
        activated.Dispose();

        // Split into numParams chunks along last dim
        Tensor[] results = new Tensor[_numParams];
        float* projPtr = (float*)projected.DataPointer;

        for (int p = 0; p < _numParams; p++)
        {
            TensorShape paramShape = new TensorShape(batch, _hiddenSize);
            Tensor param = new Tensor(paramShape, DType.F32);
            float* paramPtr = (float*)param.DataPointer;

            for (int b = 0; b < batch; b++)
            {
                int srcOffset = b * outDim + p * _hiddenSize;
                int dstOffset = b * _hiddenSize;
                for (int d = 0; d < _hiddenSize; d++)
                {
                    paramPtr[dstOffset + d] = projPtr[srcOffset + d];
                }
            }

            results[p] = param;
        }

        projected.Dispose();
        return results;
    }

    /// <summary>Applies AdaLN modulation: output = input * (1 + scale) + shift.</summary>
    public static Tensor ApplyModulation(Tensor input, Tensor shift, Tensor scale, int batch, int seqLen, int hiddenSize)
    {
        TensorShape shape = new TensorShape(batch, seqLen, hiddenSize);
        Tensor output = new Tensor(shape, DType.F32);

        float* inPtr = (float*)input.DataPointer;
        float* shiftPtr = (float*)shift.DataPointer;
        float* scalePtr = (float*)scale.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int inOffset = (b * seqLen + s) * hiddenSize;
                int condOffset = b * hiddenSize;
                for (int d = 0; d < hiddenSize; d++)
                {
                    outPtr[inOffset + d] = inPtr[inOffset + d] * (1.0f + scalePtr[condOffset + d]) + shiftPtr[condOffset + d];
                }
            }
        }

        return output;
    }

    /// <summary>Applies gated residual: output = input + gate * value. Gate is [B, hiddenSize], broadcast over sequence dim.</summary>
    public static Tensor ApplyGatedResidual(Tensor input, Tensor value, Tensor gate, int batch, int seqLen, int hiddenSize)
    {
        TensorShape shape = new TensorShape(batch, seqLen, hiddenSize);
        Tensor output = new Tensor(shape, DType.F32);

        float* inPtr = (float*)input.DataPointer;
        float* valPtr = (float*)value.DataPointer;
        float* gatePtr = (float*)gate.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int seqOffset = (b * seqLen + s) * hiddenSize;
                int condOffset = b * hiddenSize;
                for (int d = 0; d < hiddenSize; d++)
                {
                    outPtr[seqOffset + d] = inPtr[seqOffset + d] + gatePtr[condOffset + d] * valPtr[seqOffset + d];
                }
            }
        }

        return output;
    }

    private static void LinearProjection(Tensor output, Tensor input, Tensor weight, Tensor bias, int batch, int inDim, int outDim)
    {
        float* inPtr = (float*)input.DataPointer;
        float* wPtr = (float*)weight.DataPointer;
        float* bPtr = (float*)bias.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            int inOffset = b * inDim;
            int outOffset = b * outDim;
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
}
