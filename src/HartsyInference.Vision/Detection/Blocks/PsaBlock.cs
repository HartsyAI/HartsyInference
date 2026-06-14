using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Detection.Blocks;

/// <summary>YOLO11 Position-Sensitive Attention block — wraps a <see cref="PsaAttention"/> and a
/// 2-layer feedforward network with residual connections around both:
/// <code>
/// x = x + attn(x)
/// x = x + ffn(x)
/// </code>
/// FFN structure: <c>Conv(c → 2c, 1×1, BN+SiLU) → Conv(2c → c, 1×1, BN, no SiLU)</c>. The second
/// conv intentionally omits SiLU — the residual sum is the activation point.</summary>
public sealed unsafe class PsaBlock
{
    private readonly int _channels;
    private readonly PsaAttention _attn;
    private readonly ConvBnSilu _ffn0; // 1×1, c → 2c, SiLU
    private readonly ConvBnSilu _ffn1; // 1×1, 2c → c, no SiLU
    private readonly bool _shortcut;

    public PsaBlock(int channels, int numHeads, float attnRatio = 0.5f, bool shortcut = true)
    {
        _channels = channels;
        _attn = new PsaAttention(channels, numHeads, attnRatio);
        _ffn0 = new ConvBnSilu(channels * 2, 1, 1, 0, 0, useSilu: true);
        _ffn1 = new ConvBnSilu(channels, 1, 1, 0, 0, useSilu: false);
        _shortcut = shortcut;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _attn.LoadWeights(weights, $"{prefix}.attn");
        _ffn0.LoadWeights(weights, $"{prefix}.ffn.0.conv");
        _ffn1.LoadWeights(weights, $"{prefix}.ffn.1.conv");
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _attn.EnumerateWeights()) yield return t;
        foreach (Tensor t in _ffn0.EnumerateWeights()) yield return t;
        foreach (Tensor t in _ffn1.EnumerateWeights()) yield return t;
    }

    public Tensor Forward(IBackend backend, Tensor input)
    {
        // Attention path with optional residual.
        Tensor attnOut = _attn.Forward(backend, input);
        Tensor afterAttn;
        if (_shortcut)
        {
            afterAttn = new Tensor(input.Shape, DType.F32);
            backend.Add(afterAttn, input, attnOut);
            attnOut.Dispose();
        }
        else
        {
            afterAttn = attnOut;
        }

        // FFN path with optional residual.
        Tensor ffnMid = _ffn0.Forward(backend, afterAttn);
        Tensor ffnOut = _ffn1.Forward(backend, ffnMid);
        ffnMid.Dispose();

        Tensor result;
        if (_shortcut)
        {
            result = new Tensor(input.Shape, DType.F32);
            backend.Add(result, afterAttn, ffnOut);
            ffnOut.Dispose();
        }
        else
        {
            result = ffnOut;
        }
        afterAttn.Dispose();
        return result;
    }
}
