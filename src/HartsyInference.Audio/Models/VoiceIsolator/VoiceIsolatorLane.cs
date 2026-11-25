using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.VoiceIsolator;

/// <summary>The state one stream lane carries between blocks: an attention past per attention block and the time-Mamba carry per
/// Mamba block.</summary>
internal sealed unsafe class VoiceIsolatorLane : IDisposable
{
    public VoiceIsolatorLane(int blocks)
    {
        Attention = new VoiceIsolatorAttentionLane?[blocks];
        Carry = new VoiceIsolatorMambaCarry?[blocks];
    }

    public VoiceIsolatorAttentionLane?[] Attention { get; }

    public VoiceIsolatorMambaCarry?[] Carry { get; }

    /// <summary>A lane whose Mamba states are copies of this one's and whose attention past is empty: lane B of a crossfade.</summary>
    public VoiceIsolatorLane Fork()
    {
        VoiceIsolatorLane fork = new(Attention.Length);
        for (int i = 0; i < Attention.Length; i++)
        {
            if (Attention[i] is not null)
            {
                fork.Attention[i] = new VoiceIsolatorAttentionLane();
            }
            else if (Carry[i] is { } carry)
            {
                fork.Carry[i] = new VoiceIsolatorMambaCarry(Copy(carry.History), Copy(carry.State));
            }
        }
        return fork;
    }

    public static Tensor Zeros(params long[] dims)
    {
        Tensor t = new(new TensorShape(dims), DType.F32);
        new Span<float>((float*)t.DataPointer, (int)t.ElementCount).Clear();
        return t;
    }

    public void Dispose()
    {
        foreach (VoiceIsolatorAttentionLane? lane in Attention)
        {
            lane?.Dispose();
        }
        foreach (VoiceIsolatorMambaCarry? carry in Carry)
        {
            carry?.Dispose();
        }
    }

    private static Tensor Copy(Tensor source)
    {
        Tensor t = new(source.Shape, DType.F32);
        Buffer.MemoryCopy((void*)source.DataPointer, (void*)t.DataPointer, source.ElementCount * 4, source.ElementCount * 4);
        return t;
    }
}
