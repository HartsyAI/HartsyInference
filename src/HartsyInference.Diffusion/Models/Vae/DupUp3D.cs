using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>Wan2.2 VAE duplicating up-sampler (<c>DupUp3D</c> in <c>vae2_2.py</c>), reusable by any Wan/LTX-family video decoder. Expands a <c>[B, inC, T, H, W]</c> latent to <c>[B, outC, T·factorT, H·factorS, W·factorS]</c> by channel <c>repeat_interleave</c> + a reshape/permute that scatters the duplicated channels into the temporal & spatial cells (a learnable-free nearest-style expansion).
///
/// <para>Index map (verbatim from the upstream reshape/permute): for output cell <c>(oc, t·factorT+tt, h·factorS+s1, w·factorS+s2)</c>, the source is input channel <c>c' / repeats</c> where <c>c' = ((oc·factorT + tt)·factorS + s1)·factorS + s2</c> and <c>repeats = outC·factor / inC</c>, <c>factor = factorT·factorS²</c>. With <paramref name="firstChunk"/>, the leading <c>factorT−1</c> temporal frames are dropped (so a single latent frame decodes to a single output frame).</para></summary>
public static class DupUp3D
{
    public static Tensor Forward(IBackend backend, Tensor x, int outChannels, int factorT, int factorS, bool firstChunk)
    {
        ArgumentNullException.ThrowIfNull(backend);
        int b = (int)x.Shape[0], inC = (int)x.Shape[1], t = (int)x.Shape[2], h = (int)x.Shape[3], w = (int)x.Shape[4];
        int factor = factorT * factorS * factorS;
        if (outChannels * factor % inC != 0)
            throw new ArgumentException($"outChannels·factor ({outChannels}·{factor}) not divisible by inC ({inC}).");
        int dropT = firstChunk ? factorT - 1 : 0;
        Tensor output = new Tensor(new TensorShape([(long)b, outChannels, t * factorT - dropT, h * factorS, w * factorS]), x.DType);
        backend.DupUp3dVae(output, x, factorT, factorS, dropT);
        return output;
    }
}
