using HartsyInference.Core.Backends;
using HartsyInference.Core.Memory.Vram.Streaming;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>HunyuanVideo-1.5 DiT (<c>HunyuanVideo_1_5_DiffusionTransformer</c>, also MovieRoll's backbone): 54 dual-stream
/// <see cref="HunyuanImageBlock"/>s with a 3-axis <see cref="HunyuanImageRope"/>, no single-stream blocks. Image tokens come from a
/// 1x1x1 patch embedding of the 65-channel input (latent, conditioning copy, mask); text tokens are a Qwen2.5-VL stream refined by
/// <see cref="HunyuanVideoTokenRefiner"/> plus a ByT5 stream through <c>byt5_in</c>, each tagged with a <c>cond_type_embedding</c> row,
/// byT5 tokens first. One timestep per forward: <c>temb = time_in(t)</c> modulates both streams and the final layer.
/// <para>Scope of this class: one clip-wide timestep and valid (unpadded) text tokens. The per-frame modulation, per-window text
/// timestep and KV-cache forwards MovieRoll adds are a later phase. Weights come through
/// <see cref="HartsyInference.ModelAssets.CheckpointConverters.HunyuanVideo15CheckpointConverter"/>.</para></summary>
public sealed unsafe class HunyuanVideo15Dit : IDisposable, IStreamableDenoiser
{
    private const int TimestepEmbedDim = 256;

    private readonly HunyuanVideo15Config _config;
    private readonly HunyuanImageBlock[] _blocks;
    private readonly HunyuanImageRope _rope;
    private readonly HunyuanVideoTokenRefiner _refiner;

    private Tensor? _imgInW, _imgInB;
    private Tensor? _timeW0, _timeB0, _timeW1, _timeB1;
    private Tensor? _byt5NormW, _byt5NormB, _byt5W1, _byt5B1, _byt5W2, _byt5B2, _byt5W3, _byt5B3;
    private Tensor? _condType;
    private Tensor? _finalModW, _finalModB, _finalW, _finalB;

    /// <summary>Creates the DiT; call <see cref="LoadWeights"/> before <see cref="Forward"/>.</summary>
    public HunyuanVideo15Dit(HunyuanVideo15Config config)
    {
        _config = config;
        _blocks = new HunyuanImageBlock[config.NumBlocks];
        for (int i = 0; i < _blocks.Length; i++)
            _blocks[i] = new HunyuanImageBlock(config.HiddenSize, config.NumHeads, config.HeadDim, config.MlpDim);
        _rope = new HunyuanImageRope(config.RopeAxesDim, config.RopeTheta);
        _refiner = new HunyuanVideoTokenRefiner(config.TextDim, config.HiddenSize, config.NumHeads, numBlocks: 2, config.MlpRatio);
    }

    /// <summary>The shape this instance was built for.</summary>
    public HunyuanVideo15Config Config => _config;

    /// <inheritdoc/>
    public int BlockCount => _blocks.Length;

    /// <inheritdoc/>
    public Action<int>? BeforeBlockForward { get; set; }

    /// <inheritdoc/>
    public IStreamingBlock GetBlock(int idx) => _blocks[idx];

    /// <summary>Loads converter output (engine names, see <c>HunyuanVideo15CheckpointConverter</c>). Small tensors are held as F32;
    /// block weights keep their dtype.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _imgInW = F32(w, "img_in.proj.weight"); _imgInB = F32(w, "img_in.proj.bias");
        _timeW0 = F32(w, "time_in.mlp.0.weight"); _timeB0 = F32(w, "time_in.mlp.0.bias");
        _timeW1 = F32(w, "time_in.mlp.2.weight"); _timeB1 = F32(w, "time_in.mlp.2.bias");
        _byt5NormW = F32(w, "byt5_in.layernorm.weight"); _byt5NormB = F32(w, "byt5_in.layernorm.bias");
        _byt5W1 = F32(w, "byt5_in.fc1.weight"); _byt5B1 = F32(w, "byt5_in.fc1.bias");
        _byt5W2 = F32(w, "byt5_in.fc2.weight"); _byt5B2 = F32(w, "byt5_in.fc2.bias");
        _byt5W3 = F32(w, "byt5_in.fc3.weight"); _byt5B3 = F32(w, "byt5_in.fc3.bias");
        _condType = F32(w, "cond_type_embedding.weight");
        _finalModW = F32(w, "final_layer.adaLN_modulation.1.weight"); _finalModB = F32(w, "final_layer.adaLN_modulation.1.bias");
        _finalW = F32(w, "final_layer.linear.weight"); _finalB = F32(w, "final_layer.linear.bias");
        _refiner.LoadWeights(w, "txt_in");
        for (int i = 0; i < _blocks.Length; i++) _blocks[i].LoadWeights(w, $"transformer_blocks.{i}");
    }

    /// <inheritdoc/>
    public IEnumerable<Tensor> EnumerateSharedWeights()
    {
        Tensor?[] shared = [_imgInW, _imgInB, _timeW0, _timeB0, _timeW1, _timeB1, _byt5NormW, _byt5NormB, _byt5W1, _byt5B1,
            _byt5W2, _byt5B2, _byt5W3, _byt5B3, _condType, _finalModW, _finalModB, _finalW, _finalB];
        foreach (Tensor? t in shared) if (t is not null) yield return t;
        foreach (Tensor t in _refiner.EnumerateWeights()) yield return t;
    }

    /// <summary>Every weight: shared tensors, then each block.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in EnumerateSharedWeights()) yield return t;
        foreach (HunyuanImageBlock block in _blocks) foreach (Tensor t in block.EnumerateWeights()) yield return t;
    }

    /// <summary>Predicts the velocity <c>[1, 32, T, H, W]</c> of <paramref name="latent"/> (<c>[1, 32, T, H, W]</c>, F32).</summary>
    /// <param name="timestep">The scheduler timestep (0..1000), used for <c>time_in</c> and the token refiner.</param>
    /// <param name="qwen">Valid Qwen2.5-VL text tokens after the template crop, <c>[1, Lq, TextDim]</c> F32 (no padding rows).</param>
    /// <param name="byt5">Valid ByT5 glyph tokens <c>[1, Lb, ByT5Dim]</c> F32; null when the prompt has no quoted text.</param>
    /// <param name="conditioning">Optional <c>[1, 33, T, H, W]</c> (conditioning latent then mask); null feeds zeros (text-to-video).</param>
    public Tensor Forward(IBackend backend, Tensor latent, float timestep, Tensor qwen, Tensor? byt5 = null, Tensor? conditioning = null)
    {
        if (latent.Shape.Rank != 5 || latent.Shape[0] != 1 || latent.Shape[1] != _config.LatentChannels)
            throw new ArgumentException($"latent must be [1, {_config.LatentChannels}, T, H, W], got {latent.Shape}.", nameof(latent));
        int frames = (int)latent.Shape[2], height = (int)latent.Shape[3], width = (int)latent.Shape[4];
        int tokens = frames * height * width;
        int hidden = _config.HiddenSize;

        Tensor img = EmbedImage(backend, latent, conditioning, tokens);
        Tensor temb = TimestepMlp(backend, timestep);
        Tensor txt = BuildText(backend, qwen, byt5, timestep);

        for (int i = 0; i < _blocks.Length; i++)
        {
            BeforeBlockForward?.Invoke(i);
            (Tensor nextImg, Tensor nextTxt) = _blocks[i].Forward(backend, img, txt, temb, _rope, height, width, frames);
            img.Dispose(); txt.Dispose();
            img = nextImg; txt = nextTxt;
        }
        txt.Dispose();

        Tensor velocity = ApplyFinal(backend, img, temb, tokens, hidden, frames, height, width);
        temb.Dispose();
        return velocity;
    }

    /// <summary>Channel-concat the latent with the conditioning block (zeros when absent), move channels last and project.</summary>
    private Tensor EmbedImage(IBackend backend, Tensor latent, Tensor? conditioning, int tokens)
    {
        int latentChannels = _config.LatentChannels;
        int extraChannels = _config.ImgInChannels - latentChannels;
        Tensor channelsFirst = new(new TensorShape(_config.ImgInChannels, tokens), DType.F32);
        float* destination = (float*)channelsFirst.DataPointer;
        long latentCount = (long)latentChannels * tokens;
        new ReadOnlySpan<float>((void*)latent.DataPointer, (int)latentCount).CopyTo(new Span<float>(destination, (int)latentCount));
        long extraCount = (long)extraChannels * tokens;
        Span<float> extra = new(destination + latentCount, (int)extraCount);
        if (conditioning is null) extra.Clear();
        else
        {
            if (conditioning.Shape.ElementCount != extraCount)
                throw new ArgumentException($"conditioning must hold {extraChannels} channels over {tokens} tokens, got {conditioning.Shape}.", nameof(conditioning));
            new ReadOnlySpan<float>((void*)conditioning.DataPointer, (int)extraCount).CopyTo(extra);
        }

        Tensor tokenMajor = new(new TensorShape(tokens, _config.ImgInChannels), DType.F32);
        backend.Transpose2D(tokenMajor, channelsFirst, _config.ImgInChannels, tokens);
        channelsFirst.Dispose();
        Tensor tokenMajor3 = tokenMajor.Reshape(new TensorShape(1, tokens, _config.ImgInChannels));
        Tensor img = new(new TensorShape(1, tokens, _config.HiddenSize), DType.F32);
        backend.Linear(img, tokenMajor3, _imgInW!, _imgInB!);
        tokenMajor.Dispose();
        return img;
    }

    /// <summary>Sinusoid(256) then Linear, SiLU, Linear: <c>time_in.mlp</c>.</summary>
    private Tensor TimestepMlp(IBackend backend, float timestep)
    {
        int hidden = _config.HiddenSize;
        using Tensor sinusoid = new(new TensorShape(1, TimestepEmbedDim), DType.F32);
        DiTUtils.SinusoidalTimestepEmbedding(sinusoid, timestep, 1, TimestepEmbedDim);
        using Tensor first = new(new TensorShape(1, hidden), DType.F32);
        backend.Linear(first, sinusoid, _timeW0!, _timeB0!);
        using Tensor activated = new(first.Shape, DType.F32);
        backend.Silu(activated, first);
        Tensor temb = new(new TensorShape(1, hidden), DType.F32);
        backend.Linear(temb, activated, _timeW1!, _timeB1!);
        return temb;
    }

    /// <summary>byT5 rows first (<c>byt5_in</c> plus type row 1), then the refined Qwen rows (type row 0); valid tokens only.</summary>
    private Tensor BuildText(IBackend backend, Tensor qwen, Tensor? byt5, float timestep)
    {
        int hidden = _config.HiddenSize;
        Tensor refined = _refiner.Forward(backend, qwen, timestep);
        AddTypeRow(refined, 0);
        if (byt5 is null) return refined;

        Tensor mapped = MapByT5(backend, byt5);
        AddTypeRow(mapped, 1);
        Tensor joined = new(new TensorShape(1, mapped.Shape[1] + refined.Shape[1], hidden), DType.F32);
        backend.Concat(joined, [mapped, refined], 1);
        mapped.Dispose(); refined.Dispose();
        return joined;
    }

    /// <summary><c>ByT5Mapper</c>: LayerNorm, fc1, GELU(erf), fc2, GELU(erf), fc3 (no residual).</summary>
    private Tensor MapByT5(IBackend backend, Tensor byt5)
    {
        int length = (int)byt5.Shape[1];
        int hidden = _config.HiddenSize;
        using Tensor normed = new(byt5.Shape, DType.F32);
        backend.LayerNorm(normed, byt5, _byt5NormW!, _byt5NormB!, 1e-5f);
        using Tensor first = new(new TensorShape(1, length, _byt5W1!.Shape[0]), DType.F32);
        backend.Linear(first, normed, _byt5W1!, _byt5B1!);
        using Tensor firstActivated = new(first.Shape, DType.F32);
        backend.GeluErf(firstActivated, first);
        using Tensor second = new(new TensorShape(1, length, _byt5W2!.Shape[0]), DType.F32);
        backend.Linear(second, firstActivated, _byt5W2!, _byt5B2!);
        using Tensor secondActivated = new(second.Shape, DType.F32);
        backend.GeluErf(secondActivated, second);
        Tensor mapped = new(new TensorShape(1, length, hidden), DType.F32);
        backend.Linear(mapped, secondActivated, _byt5W3!, _byt5B3!);
        return mapped;
    }

    /// <summary>Adds <c>cond_type_embedding[row]</c> to every token in place.</summary>
    private void AddTypeRow(Tensor tokens, int row)
    {
        int hidden = _config.HiddenSize;
        long count = tokens.ElementCount / hidden;
        float* data = (float*)tokens.DataPointer;
        float* type = (float*)_condType!.DataPointer + (long)row * hidden;
        for (long i = 0; i < count; i++)
        {
            float* tokenRow = data + i * hidden;
            for (int c = 0; c < hidden; c++) tokenRow[c] += type[c];
        }
    }

    /// <summary><c>FinalLayer</c>: LayerNorm (no affine) modulated by <c>chunk(adaLN(SiLU(temb)))</c> = <c>[shift, scale]</c>, then Linear, and the
    /// 1x1x1 unpatchify (<c>[1, T*H*W, C]</c> to <c>[1, C, T, H, W]</c>). Consumes <paramref name="img"/>.</summary>
    private Tensor ApplyFinal(IBackend backend, Tensor img, Tensor temb, int tokens, int hidden, int frames, int height, int width)
    {
        using Tensor activated = new(temb.Shape, DType.F32);
        backend.Silu(activated, temb);
        using Tensor mod = new(new TensorShape(1, 2 * hidden), DType.F32);
        backend.Linear(mod, activated, _finalModW!, _finalModB!);

        Tensor normed = new(img.Shape, DType.F32);
        backend.LayerNormNoAffine(normed, img, 1e-6f);
        img.Dispose();
        using Tensor shift = new(new TensorShape(hidden), DType.F32);
        backend.SliceRows(shift, mod, 0);
        using Tensor scale = new(new TensorShape(hidden), DType.F32);
        backend.SliceRows(scale, mod, 1);
        using Tensor scalePlusOne = new(new TensorShape(hidden), DType.F32);
        backend.AddScalar(scalePlusOne, scale, 1f);
        using Tensor modulated = new(normed.Shape, DType.F32);
        backend.AffineBroadcastLastDim(modulated, normed, scalePlusOne, shift);
        normed.Dispose();

        int channels = _config.LatentChannels;
        using Tensor projected = new(new TensorShape(tokens, channels), DType.F32);
        using Tensor flat = modulated.Reshape(new TensorShape(tokens, hidden));
        backend.Linear(projected, flat, _finalW!, _finalB!);
        Tensor channelsFirst = new(new TensorShape(channels, tokens), DType.F32);
        backend.Transpose2D(channelsFirst, projected, tokens, channels);
        Tensor velocity = channelsFirst.Reshape(new TensorShape([1L, channels, frames, height, width]));
        return velocity;
    }

    private static Tensor F32(IReadOnlyDictionary<string, Tensor> w, string key) =>
        w.TryGetValue(key, out Tensor? tensor) ? TensorCasts.EnsureF32(tensor) : throw new KeyNotFoundException($"HunyuanVideo-1.5 DiT is missing '{key}'.");

    /// <inheritdoc/>
    public void Dispose() => _rope.Dispose();
}
