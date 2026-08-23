using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

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

    /// <summary>Loads the linear projection weights. Weight shape: [numParams * hiddenSize, hiddenSize], Bias shape: [numParams * hiddenSize] (may be null for Flux.2 where modulation projections have no bias).</summary>
    public void LoadWeights(Tensor weight, Tensor? bias)
    {
        _linearWeight = weight;
        _linearBias = bias;
    }

    /// <summary>Enumerates all weight tensors for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_linearWeight is not null) yield return _linearWeight;
        if (_linearBias is not null) yield return _linearBias;
    }

    /// <summary>Computes modulation parameters from timestep embedding. Input: [B, hiddenSize] → Output: numParams tensors each [B, hiddenSize].</summary>
    public Tensor[] Forward(IBackend backend, Tensor timestepEmb)
    {
        int batch = (int)timestepEmb.Shape[0];
        int outDim = _numParams * _hiddenSize;

        TensorShape inputShape = new TensorShape(batch, _hiddenSize);
        Tensor activated = new Tensor(inputShape, timestepEmb.DType);
        backend.Silu(activated, timestepEmb);

        // Linear projection: [B, hiddenSize] → [B, numParams * hiddenSize]
        TensorShape outShape = new TensorShape(batch, outDim);
        Tensor projected = new Tensor(outShape, activated.DType);
        backend.Linear(projected, activated, _linearWeight!, _linearBias);
        activated.Dispose();

        // Split into numParams chunks along last dim. B=1 F32 (the inference hot path): chunk p is the
        // contiguous element range [p*hidden, (p+1)*hidden) of the flat projection, exactly SliceRows'
        // contract (rowOffset · lastDim of output) — the split stays device-resident instead of D2H-syncing
        // the projection and re-uploading every param, once per block per step. B>1 keeps the strided host copy.
        Tensor[] results = new Tensor[_numParams];
        if (batch == 1 && projected.DType == DType.F32)
        {
            for (int p = 0; p < _numParams; p++)
            {
                Tensor param = new Tensor(new TensorShape(batch, _hiddenSize), DType.F32);
                backend.SliceRows(param, projected, p);
                results[p] = param;
            }
        }
        else
        {
            float* projPtr = (float*)projected.DataPointer;

            for (int p = 0; p < _numParams; p++)
            {
                TensorShape paramShape = new TensorShape(batch, _hiddenSize);
                Tensor param = new Tensor(paramShape, projected.DType);
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
        }

        projected.Dispose();
        return results;
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

}
