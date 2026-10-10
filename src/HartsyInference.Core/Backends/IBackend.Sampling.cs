using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

// Token sampling on the device, the stochastic twin of ArgMaxInto, so a captured decode step need not be greedy.
public partial interface IBackend
{
    /// <summary>True when <see cref="SampleTopKInto"/> runs on this backend.</summary>
    bool DeviceSamplingSupported => false;

    /// <summary>Largest <c>topK</c> <see cref="SampleTopKInto"/> takes.</summary>
    int DeviceSamplingMaxTopK => 0;

    /// <summary>Allocates the persistent state a captured sampler needs: a 64-bit RNG seed and draw counter, and scratch for the
    /// <paramref name="maxTopK"/> candidates. 0 when unsupported. Written outside capture, freed with <see cref="FreeDeviceRng"/>.</summary>
    ulong AllocDeviceRng(ulong seed, int maxTopK) => 0;

    /// <summary>Frees a buffer from <see cref="AllocDeviceRng"/>.</summary>
    void FreeDeviceRng(ulong handle) { }

    /// <summary>Draws the next token from the <paramref name="topK"/> most likely entries of <paramref name="logits"/> (temperature, then the
    /// nucleus cut <paramref name="topP"/> and the <paramref name="minP"/> cut, then a multinomial draw from the device RNG) and writes its
    /// id to <paramref name="outputTokenId"/>. Capturable: nothing returns to the host.</summary>
    void SampleTopKInto(ulong outputTokenId, Tensor logits, int topK, float temperature, float topP, float minP, ulong rngState) { }
}
