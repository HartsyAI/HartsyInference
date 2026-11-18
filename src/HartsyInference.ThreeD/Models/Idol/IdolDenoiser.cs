using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Norm;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.ThreeD.Models.Idol;

/// <summary>Idol's denoiser (upstream <c>LatentSeqMMFlowModel</c>): a rectified-flow transformer over the 8192-token object latent and the
/// one-token input camera, conditioned on the image tokens. The latent, image and camera streams each pass two refiner blocks, then all
/// three share 24 joint blocks of rotary self-attention; the model returns the velocity of the latent and of the camera.
/// <para>The image stream does not depend on the time step, and the latent prefix (embedding and noise refiner) does not depend on the
/// condition, so a guided sampler can call <see cref="EmbedContext"/> once per image and run the prefix once per step.</para>
/// Borrows <see cref="IdolWeights"/>; owns the position embedding and the blocks it builds.</summary>
public sealed class IdolDenoiser : IDisposable
{
    private const int TimestepChannels = 256;
    private const float FinalNormEps = 1e-5f;

    private readonly IdolWeights _weights;
    private readonly IdolConfig _config;
    private readonly IdolBlock[] _noiseRefiner, _contextRefiner, _blocks;
    private readonly IdolRePo[] _noiseRepo, _contextRepo, _repo;
    private readonly Tensor _positionEmbedding;
    private int _disposed;

    public IdolDenoiser(IdolWeights weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        _weights = weights;
        _config = weights.Config;
        List<IDisposable> built = [];
        T Own<T>(T item) where T : IDisposable
        {
            built.Add(item);
            return item;
        }

        try
        {
            _noiseRefiner = [.. Enumerable.Range(0, _config.RefinerBlocks).Select(i => Own(new IdolBlock(weights, $"noise_refiner.{i}", _config, modulated: true)))];
            _contextRefiner = [.. Enumerable.Range(0, _config.RefinerBlocks).Select(i => Own(new IdolBlock(weights, $"context_refiner.{i}", _config, modulated: false)))];
            _blocks = [.. Enumerable.Range(0, _config.Blocks).Select(i => Own(new IdolBlock(weights, $"blocks.{i}", _config, modulated: true)))];
            _noiseRepo = [.. Enumerable.Range(0, _config.RefinerBlocks).Select(i => Own(new IdolRePo(weights, $"noise_repo_layers.{i}", _config)))];
            _contextRepo = [.. Enumerable.Range(0, _config.RefinerBlocks).Select(i => Own(new IdolRePo(weights, $"context_repo_layers.{i}", _config)))];
            _repo = [.. Enumerable.Range(0, _config.Blocks).Select(i => Own(new IdolRePo(weights, $"repo_layers.{i}", _config)))];
            _positionEmbedding = Own(IdolPositionEmbedding.Build(weights["pos_pe"], _config.ModelChannels));
        }
        catch
        {
            foreach (IDisposable item in built)
                item.Dispose();
            throw;
        }
    }

    public IdolConfig Config => _config;

    // The checkpoint keys the blocks and the rotary layers own; a forward pass reads them through the tensors those classes enumerate.
    private static readonly string[] BlockPrefixes = ["blocks.", "noise_refiner.", "context_refiner.", "repo_layers.", "noise_repo_layers.", "context_repo_layers."];

    /// <summary>Exactly the tensors a forward pass hands to the backend, for <c>IBackend.PreloadWeights</c> and <c>FreeWeights</c>: the model's own layers, the position embedding, and what each block and rotary layer reads
    /// (their q and k gains as the reshaped views the kernels get, the head norm's ones, the angle matrices).</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (string key in IdolWeights.ExpectedShapes(_config).Keys)
            if (!BlockPrefixes.Any(p => key.StartsWith(p, StringComparison.Ordinal)))
                yield return _weights[key];
        yield return _positionEmbedding;
        foreach (IdolBlock b in _noiseRefiner.Concat(_contextRefiner).Concat(_blocks))
            foreach (Tensor t in b.EnumerateWeights())
                yield return t;
        foreach (IdolRePo r in _noiseRepo.Concat(_contextRepo).Concat(_repo))
            foreach (Tensor t in r.EnumerateWeights())
                yield return t;
    }

    /// <summary>The image stream <c>[1, condTokens, C]</c> from the DINOv3 tokens <paramref name="feature1"/> <c>[1, L, 1280]</c> and the FLUX.2
    /// VAE tokens <paramref name="feature2"/> <c>[1, L, 128]</c>: both embedded, summed and refined by the unmodulated blocks. No time
    /// dependence. The caller owns the result.</summary>
    public Tensor EmbedContext(IBackend backend, Tensor feature1, Tensor feature2, Action<string, Tensor>? tap = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        if (feature1.Shape.Rank != 3 || feature2.Shape.Rank != 3 || feature1.Shape[0] != 1 || feature2.Shape[0] != 1
            || feature1.Shape[2] != _config.CondChannels || feature2.Shape[2] != _config.Cond2Channels || feature1.Shape[1] != feature2.Shape[1])
            throw new ArgumentException($"Idol's image tokens are a [1, L, {_config.CondChannels}] DINOv3 stream and a [1, L, {_config.Cond2Channels}] VAE stream of the same L; got {feature1.Shape} and {feature2.Shape}.");
        using Tensor a = Linear(backend, feature1, "cond_embedder", _config.ModelChannels);
        using Tensor b = Linear(backend, feature2, "cond_embedder2", _config.ModelChannels);
        Tensor h = new(a.Shape, DType.F32);
        backend.Add(h, a, b);
        for (int i = 0; i < _contextRefiner.Length; i++)
        {
            Tensor next;
            try
            {
                next = _contextRefiner[i].Forward(backend, h, null, _contextRepo[i]);
            }
            catch
            {
                h.Dispose();
                throw;
            }

            h.Dispose();
            h = next;
        }

        tap?.Invoke("after_context_refiner", h);
        return h;
    }

    /// <summary>One velocity: <paramref name="latent"/> <c>[1, 8192, 16]</c> and <paramref name="camera"/> <c>[1, 1, 5]</c> at the model time
    /// <paramref name="t"/> (1000 times the flow time), conditioned on <paramref name="context"/> from <see cref="EmbedContext"/>. Returns the
    /// velocities; the caller owns both. <paramref name="tap"/> sees the named intermediate tensors (borrowed) for parity checks.</summary>
    public (Tensor Latent, Tensor Camera) Forward(IBackend backend, Tensor latent, Tensor camera, float t, Tensor context, Action<string, Tensor>? tap = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        int d = _config.ModelChannels, n = _config.Tokens;
        if (latent.Shape.Rank != 3 || camera.Shape.Rank != 3 || latent.Shape[0] != 1 || camera.Shape[0] != 1 || latent.Shape[1] != n || camera.Shape[1] != 1
            || latent.Shape[2] != _config.LatentChannels || camera.Shape[2] != _config.CameraChannels)
            throw new ArgumentException($"Idol takes a [1, {n}, {_config.LatentChannels}] latent and a [1, 1, {_config.CameraChannels}] camera; got {latent.Shape} and {camera.Shape}.");
        if (context.Shape.Rank != 3 || context.Shape[0] != 1 || context.Shape[2] != d)
            throw new ArgumentException($"Idol's context is [1, L, {d}] from EmbedContext; got {context.Shape}.");

        using Tensor timeEmbedding = TimeEmbedding(backend, t);
        using Tensor timeModulation = TimeModulation(backend, timeEmbedding);

        Tensor hx = Embed(backend, latent);
        tap?.Invoke("embedded_tokens", hx);
        for (int i = 0; i < _noiseRefiner.Length; i++)
        {
            Tensor next;
            try
            {
                next = _noiseRefiner[i].Forward(backend, hx, timeModulation, _noiseRepo[i]);
            }
            catch
            {
                hx.Dispose();
                throw;
            }

            hx.Dispose();
            hx = next;
        }

        tap?.Invoke("after_noise_refiner", hx);

        using Tensor camHidden = CameraStream(backend, camera);
        int total = n + (int)context.Shape[1] + 1;
        Tensor h = new(new TensorShape(1, total, d), DType.F32);
        backend.Concat(h, [hx, context, camHidden], 1);
        hx.Dispose();
        for (int i = 0; i < _blocks.Length; i++)
        {
            Action<Tensor, Tensor>? rotationTap = i == 0 && tap is not null
                ? (cos, sin) => { tap("repo0_cos", cos); tap("repo0_sin", sin); }
                : null;
            Tensor next;
            try
            {
                next = _blocks[i].Forward(backend, h, timeModulation, _repo[i], rotationTap);
            }
            catch
            {
                h.Dispose();
                throw;
            }

            h.Dispose();
            h = next;
            if (i == 0)
                tap?.Invoke("after_block0", h);
        }

        tap?.Invoke("after_block23", h);
        try
        {
            return Head(backend, h, timeEmbedding, n, total);
        }
        finally
        {
            h.Dispose();
        }
    }

    // Idol's t_embedder: the sinusoidal embedding of the model time (cosines first), then Linear, SiLU, Linear.
    private Tensor TimeEmbedding(IBackend backend, float t)
    {
        using Tensor sinusoid = new(new TensorShape(1, TimestepChannels), DType.F32);
        DiTUtils.SinusoidalTimestepEmbedding(sinusoid, t, 1, TimestepChannels);
        using Tensor hidden = Linear(backend, sinusoid, "t_embedder.mlp.0", _config.ModelChannels);
        using Tensor act = new(hidden.Shape, DType.F32);
        backend.Silu(act, hidden);
        return Linear(backend, act, "t_embedder.mlp.2", _config.ModelChannels);
    }

    // adaLN_modulation: SiLU then Linear to the six shared modulation vectors.
    private Tensor TimeModulation(IBackend backend, Tensor timeEmbedding)
    {
        using Tensor act = new(timeEmbedding.Shape, DType.F32);
        backend.Silu(act, timeEmbedding);
        return Linear(backend, act, "adaLN_modulation.1", 6L * _config.ModelChannels);
    }

    // input_layer(z) + the fixed position embedding of the Sobol tokens.
    private Tensor Embed(IBackend backend, Tensor latent)
    {
        using Tensor projected = Linear(backend, latent, "input_layer", _config.ModelChannels);
        Tensor h = new(projected.Shape, DType.F32);
        backend.Add(h, projected, _positionEmbedding);
        return h;
    }

    // cam_refiner: MLP(5 -> width -> width) with tanh GELU between.
    private Tensor CameraStream(IBackend backend, Tensor camera)
    {
        using Tensor hidden = Linear(backend, camera, "cam_refiner.mlp.0", _config.ModelChannels);
        using Tensor act = new(hidden.Shape, DType.F32);
        backend.Gelu(act, hidden);
        return Linear(backend, act, "cam_refiner.mlp.2", _config.ModelChannels);
    }

    // The output head: an affine-free LayerNorm per stream, (shift, scale) = shift_table + the time embedding, then the two projections.
    private (Tensor Latent, Tensor Camera) Head(IBackend backend, Tensor h, Tensor timeEmbedding, int latentTokens, int total)
    {
        int d = _config.ModelChannels;
        using Tensor embedding = timeEmbedding.Reshape(new TensorShape(1, 1, d));
        using Tensor timeRows = new(new TensorShape(1, 2, d), DType.F32);
        backend.Concat(timeRows, [embedding, embedding], 1);
        using Tensor shiftScale = new(timeRows.Shape, DType.F32);
        backend.Add(shiftScale, _weights["shift_table"], timeRows);
        using Tensor shift = new(new TensorShape(1, d), DType.F32);
        using Tensor scale = new(new TensorShape(1, d), DType.F32);
        backend.SliceRows(shift, shiftScale, 0);
        backend.SliceRows(scale, shiftScale, 1);

        Tensor Stream(int rowOffset, int tokens, string outLayer, long outChannels)
        {
            using Tensor rows = new(new TensorShape(1, tokens, d), DType.F32);
            backend.SliceRows(rows, h, rowOffset);
            using Tensor modulated = new(rows.Shape, DType.F32);
            backend.NormModulate(NormModulateCall.Rows(modulated, rows, NormKind.LayerNorm, FinalNormEps)
                .WithModulation(ModulationSpec.Broadcast(rows, scale, shift, ScaleMode.OnePlus)));
            return Linear(backend, modulated, outLayer, outChannels);
        }

        Tensor latent = Stream(0, latentTokens, "out_layer", _config.LatentChannels);
        try
        {
            return (latent, Stream(total - 1, 1, "cam_out_layer", _config.CameraChannels));
        }
        catch
        {
            latent.Dispose();
            throw;
        }
    }

    private Tensor Linear(IBackend backend, Tensor x, string layer, long outDim)
    {
        Tensor o = new(new TensorShape(1, x.Shape[x.Shape.Rank - 2], outDim), DType.F32);
        backend.Linear(o, x, _weights[$"{layer}.weight"], _weights[$"{layer}.bias"]);
        return o;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        foreach (IdolBlock b in _noiseRefiner.Concat(_contextRefiner).Concat(_blocks))
            b.Dispose();
        foreach (IdolRePo r in _noiseRepo.Concat(_contextRepo).Concat(_repo))
            r.Dispose();
        _positionEmbedding.Dispose();
    }
}
