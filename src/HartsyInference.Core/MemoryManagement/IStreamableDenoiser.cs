using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.MemoryManagement;

/// <summary>A block-structured denoiser whose per-block weights can ride a <see cref="BlockStreamingController"/>.</summary>
/// <remarks>Implementing this is the whole opt-in: <see cref="BlockStreamingScope"/> drives everything else. The member
/// signatures deliberately match what the Flux/LTX-2/Krea2/Qwen/Chroma/Ideogram/Hunyuan denoisers already expose, so
/// adopting is adding the interface to the declaration.</remarks>
public interface IStreamableDenoiser
{
    /// <summary>Number of streamable blocks; <see cref="GetBlock"/> indexes <c>[0, BlockCount)</c>.</summary>
    int BlockCount { get; }

    /// <summary>The streamable block at <paramref name="idx"/>.</summary>
    /// <remarks>Must hand back the same tensor references on every call — <see cref="BlockStreamingController"/>
    /// tracks residency by reference.</remarks>
    IStreamingBlock GetBlock(int idx);

    /// <summary>Every weight that is NOT part of a block, and so must stay resident for the whole loop.</summary>
    IEnumerable<Tensor> EnumerateSharedWeights();

    /// <summary>Invoked at the top of the block loop with the same index <see cref="GetBlock"/> uses; null = all resident.</summary>
    Action<int>? BeforeBlockForward { get; set; }
}
