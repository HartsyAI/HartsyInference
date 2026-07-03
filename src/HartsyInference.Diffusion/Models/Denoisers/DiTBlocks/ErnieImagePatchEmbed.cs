using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>ERNIE-Image patch embedding. Single 1×1 (or 2×2, configurable) Conv2D-with-bias that maps the VAE latent <c>[B, in_channels, H, W]</c> → <c>[B, hidden, H/p, W/p]</c>, then flattens spatial dims to a token sequence <c>[B, (H/p)*(W/p), hidden]</c>. Patch size is 1 by default because patchification is handled inside the Flux2-style VAE; the conv is purely a linear up-projection from 128-channel latents to hidden=4096.
///
/// Reference: <c>transformer_ernie_image.py:67-76</c> (<c>ErnieImagePatchEmbedDynamic</c>).
/// Weight keys: <c>x_embedder.proj.weight [hidden, in_channels, p, p]</c>, <c>x_embedder.proj.bias [hidden]</c>.</summary>
public sealed unsafe class ErnieImagePatchEmbed
{
    private readonly int _inChannels;
    private readonly int _hidden;
    private readonly int _patchSize;

    private Tensor? _projWeight;
    private Tensor? _projBias;

    /// <summary>Creates the patch embedding.</summary>
    /// <param name="inChannels">Latent channel count (128 for the Flux2-style VAE).</param>
    /// <param name="hidden">Output hidden dim (4096 for ERNIE-Image v1).</param>
    /// <param name="patchSize">Spatial patch size. 1 for ERNIE-Image v1 (VAE already patchified).</param>
    public ErnieImagePatchEmbed(int inChannels, int hidden, int patchSize = 1)
    {
        if (patchSize < 1) throw new ArgumentException("Patch size must be >= 1.", nameof(patchSize));
        _inChannels = inChannels;
        _hidden = hidden;
        _patchSize = patchSize;
    }

    /// <summary>Number of output tokens for a given latent grid: <c>(H/p) * (W/p)</c>.</summary>
    public int PatchSize => _patchSize;

    /// <summary>Loads the conv weight + bias. Diffusers / single-file naming both expose these under <c>x_embedder.proj.{weight,bias}</c>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix = "x_embedder.proj")
    {
        _projWeight = weights[$"{prefix}.weight"];
        _projBias = weights[$"{prefix}.bias"];
    }

    /// <summary>Enumerates the conv weight + bias for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_projWeight is not null) yield return _projWeight;
        if (_projBias is not null) yield return _projBias;
    }

    /// <summary>Forward: <c>[B, inChannels, H, W] → [B, (H/p)*(W/p), hidden]</c>. Internally runs Conv2D then transposes the result to token-sequence layout.</summary>
    public Tensor Forward(IBackend backend, Tensor latent)
    {
        if (latent.Shape.Rank != 4)
            throw new ArgumentException($"Patch embed input must be 4D [B, C, H, W], got {latent.Shape}.", nameof(latent));

        int batch = (int)latent.Shape[0];
        int channels = (int)latent.Shape[1];
        int height = (int)latent.Shape[2];
        int width = (int)latent.Shape[3];
        if (channels != _inChannels)
            throw new ArgumentException($"Latent channels {channels} != configured {_inChannels}.", nameof(latent));
        if (height % _patchSize != 0 || width % _patchSize != 0)
            throw new ArgumentException($"Latent {height}x{width} must be divisible by patch size {_patchSize}.", nameof(latent));

        int outH = height / _patchSize;
        int outW = width / _patchSize;
        int seqLen = outH * outW;

        TensorShape convShape = new TensorShape(batch, _hidden, outH, outW);
        Tensor convOut = new Tensor(convShape, latent.DType);
        backend.Conv2D(convOut, latent, _projWeight!, _projBias, _patchSize, _patchSize, 0, 0);

        // Reshape [B, hidden, outH, outW] → [B, hidden, seqLen] → [B, seqLen, hidden] on the backend (the conv output
        // is contiguously [B, hidden, seqLen], so this is exactly a batched 2D transpose) — keeps the token sequence
        // device-resident for the downstream concat + block loop; identical to the old host token-major copy.
        TensorShape outShape = new TensorShape(batch, seqLen, _hidden);
        Tensor output = new Tensor(outShape, latent.DType);
        backend.Transpose2D(output, convOut, _hidden, seqLen);

        convOut.Dispose();
        return output;
    }
}
