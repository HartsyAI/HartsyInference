using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Layout;
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
        if (config.RopeAxesDim.Sum() != config.HeadDim)
            throw new ArgumentException($"RopeAxesDim {string.Join('+', config.RopeAxesDim)} must sum to the head dim {config.HeadDim}.", nameof(config));
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
        (int frames, int height, int width) = ValidateInputs(latent, qwen, byt5, conditioning);
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

    /// <summary>Velocity with per-frame timesteps (MovieRoll's streaming path): latent frame f is denoised at frameTimesteps[f], so its image
    /// tokens take that frame's adaLN rows in every block and in the final layer. The text takes windowTimestep. Single window, no cache.
    /// With every frame at the same timestep this equals Forward.</summary>
    public Tensor ForwardPerFrame(IBackend backend, Tensor latent, float[] frameTimesteps, float windowTimestep, Tensor qwen, Tensor? byt5 = null, Tensor? conditioning = null)
    {
        ArgumentNullException.ThrowIfNull(frameTimesteps);
        (int frames, int height, int width) = ValidateInputs(latent, qwen, byt5, conditioning);
        if (frameTimesteps.Length != frames)
            throw new ArgumentException($"{frameTimesteps.Length} frame timesteps for {frames} latent frames.", nameof(frameTimesteps));
        int tokens = frames * height * width;
        int hidden = _config.HiddenSize;

        using TokenSegmentLayout layout = new(Enumerable.Repeat(height * width, frames).ToArray(), Enumerable.Range(0, frames).ToArray());
        Tensor? img = null, txt = null, txtTemb = null, frameTemb = null;
        try
        {
            img = EmbedImage(backend, latent, conditioning, tokens);
            txtTemb = TimestepMlp(backend, windowTimestep);
            txt = BuildText(backend, qwen, byt5, windowTimestep);
            frameTemb = TimestepMlpRows(backend, frameTimesteps);
            for (int i = 0; i < _blocks.Length; i++)
            {
                BeforeBlockForward?.Invoke(i);
                (Tensor nextImg, Tensor nextTxt) = _blocks[i].ForwardPerFrame(backend, img, txt, txtTemb, frameTemb, layout, _rope, height, width, frames);
                img.Dispose(); txt.Dispose();
                img = nextImg; txt = nextTxt;
            }
            txt.Dispose();
            txt = null;
            Tensor finalInput = img;
            img = null;
            return ApplyFinalPerFrame(backend, finalInput, frameTemb, layout, tokens, hidden, frames, height, width);
        }
        finally
        {
            img?.Dispose();
            txt?.Dispose();
            txtTemb?.Dispose();
            frameTemb?.Dispose();
        }
    }

    /// <summary>MovieRoll's streaming config fixes the sink at one chunk (<c>sink_chunks == 1</c>, enforced by its <c>validate</c>).</summary>
    private const int MovieRollSinkChunks = 1;

    /// <summary>The cache layer name of block <paramref name="block"/> in a <see cref="MovieRollKvCache"/> of this DiT, the MovieRoll
    /// <c>double_block_{i+1}</c> naming. Initialise the cache with <see cref="StreamLayerNames"/>.</summary>
    public static string StreamLayerName(int block) => $"double_block_{block + 1}";

    /// <summary>The cache layer names of every block, in block order.</summary>
    public IReadOnlyList<string> StreamLayerNames() => Enumerable.Range(0, _blocks.Length).Select(StreamLayerName).ToArray();

    /// <summary>MovieRoll's streaming denoise of one window (<c>forward_inference</c> with text): each block attends over its cached history
    /// and the window's own tokens, text included, and the window's velocity is returned. Blocks below <paramref name="fullHistoryBlocks"/>
    /// read their whole history; the rest (sink-recent) read what the last <see cref="StreamEncode"/> left them: the sink and the previous
    /// window. The cache is not changed here, so the caller encodes the window once it is clean.</summary>
    public Tensor StreamDenoise(IBackend backend, Tensor latent, float[] frameTimesteps, float windowTimestep, Tensor qwen, Tensor? byt5,
        MovieRollKvCache cache, int fullHistoryBlocks)
    {
        return StreamWindow(backend, latent, frameTimesteps, windowTimestep, qwen, byt5, cache, fullHistoryBlocks, encode: false, startFrame: 0)!;
    }

    /// <summary>MovieRoll's encode pass of one clean window: no text, the window's own K and V are appended to each block's cache at frames
    /// <c>[startFrame, startFrame + frames)</c>. The sink-recent blocks first drop their previous window, keeping only the sink, so the window
    /// is encoded over the sink alone, as MovieRoll's <c>trim_to_sink</c> does before every encode. <paramref name="qwen"/> and
    /// <paramref name="byt5"/> are validated but not read.</summary>
    public void StreamEncode(IBackend backend, Tensor latent, float[] frameTimesteps, Tensor qwen, Tensor? byt5, MovieRollKvCache cache,
        int fullHistoryBlocks, int startFrame)
    {
        if (startFrame < 0) throw new ArgumentOutOfRangeException(nameof(startFrame), startFrame, "The window must start at frame 0 or later.");
        Tensor? discarded = StreamWindow(backend, latent, frameTimesteps, 0f, qwen, byt5, cache, fullHistoryBlocks, encode: true, startFrame);
        discarded?.Dispose();
    }

    private Tensor? StreamWindow(IBackend backend, Tensor latent, float[] frameTimesteps, float windowTimestep, Tensor qwen, Tensor? byt5,
        MovieRollKvCache cache, int fullHistoryBlocks, bool encode, int startFrame)
    {
        ArgumentNullException.ThrowIfNull(frameTimesteps);
        ArgumentNullException.ThrowIfNull(cache);
        if (!cache.IsInit) throw new InvalidOperationException("Init the cache with StreamLayerNames before streaming.");
        if (fullHistoryBlocks < 0 || fullHistoryBlocks > _blocks.Length)
            throw new ArgumentOutOfRangeException(nameof(fullHistoryBlocks), fullHistoryBlocks, $"Full-history blocks must be in [0, {_blocks.Length}].");
        (int frames, int height, int width) = ValidateInputs(latent, qwen, byt5, null);
        if (frameTimesteps.Length != frames)
            throw new ArgumentException($"{frameTimesteps.Length} frame timesteps for {frames} latent frames.", nameof(frameTimesteps));
        int tokens = frames * height * width;
        int hidden = _config.HiddenSize;

        using TokenSegmentLayout layout = new(Enumerable.Repeat(height * width, frames).ToArray(), Enumerable.Range(0, frames).ToArray());
        Tensor? img = null, txt = null, txtTemb = null, frameTemb = null;
        try
        {
            img = EmbedImage(backend, latent, null, tokens);
            frameTemb = TimestepMlpRows(backend, frameTimesteps);
            if (!encode)
            {
                txtTemb = TimestepMlp(backend, windowTimestep);
                txt = BuildText(backend, qwen, byt5, windowTimestep);
            }

            for (int i = 0; i < _blocks.Length; i++)
            {
                string layer = StreamLayerName(i);
                if (encode && i >= fullHistoryBlocks) cache.TrimToSink(layer, MovieRollSinkChunks);
                (Tensor Keys, Tensor Values)? history = cache.Get(layer);
                int cachedFrames = cache.FrameRanges(layer).Sum(range => range.End - range.Start);
                float[] positions = HunyuanImagePosEmbed.GridRows([cachedFrames + frames, height, width]);

                MovieRollWindowOutput output = _blocks[i].ForwardWindow(backend, img!, txt, txtTemb, frameTemb!, layout, _rope, positions,
                    history?.Keys, history?.Values, returnKv: encode);
                try
                {
                    if (encode) cache.Update(layer, output.Keys!, output.Values!, startFrame, startFrame + frames);
                }
                finally
                {
                    output.Keys?.Dispose();
                    output.Values?.Dispose();
                }
                img.Dispose();
                img = output.Image;
                if (!encode)
                {
                    txt!.Dispose();
                    txt = output.Text;
                }
            }

            if (encode)
            {
                img.Dispose();
                img = null;
                return null;
            }
            txt!.Dispose();
            txt = null;
            Tensor finalInput = img!;
            img = null;
            return ApplyFinalPerFrame(backend, finalInput, frameTemb!, layout, tokens, hidden, frames, height, width);
        }
        finally
        {
            img?.Dispose();
            txt?.Dispose();
            txtTemb?.Dispose();
            frameTemb?.Dispose();
        }
    }

    /// <summary>The final layer with each latent frame's adaLN shift and scale (1 + scale, applied by the segment op). The projection is one batched
    /// Linear over all frames, and the shift and scale columns are split on the device. Consumes <paramref name="img"/>: it is disposed on every path.</summary>
    private Tensor ApplyFinalPerFrame(IBackend backend, Tensor img, Tensor frameTemb, TokenSegmentLayout layout,
        int tokens, int hidden, int frames, int height, int width)
    {
        SegmentModulation? modulation = null;
        Tensor? modulated = null;
        try
        {
            using Tensor activated = new(frameTemb.Shape, DType.F32);
            backend.Silu(activated, frameTemb);
            using Tensor mod = new(new TensorShape(frames, 2 * hidden), DType.F32);
            backend.Linear(mod, activated, _finalModW!, _finalModB!);
            Tensor shift = new(new TensorShape(frames, hidden), DType.F32);
            StridedCopies.SliceAxis(backend, shift, mod, 1, 0, hidden);
            Tensor scale = new(new TensorShape(frames, hidden), DType.F32);
            StridedCopies.SliceAxis(backend, scale, mod, 1, hidden, hidden);
            // NormAffine never reads the gate. The table is never touched, so Tensor never allocates its memory.
            Tensor gate = new(new TensorShape(frames, hidden), DType.F32);
            modulation = SegmentModulation.FromTables(shift, scale, gate, layout);

            modulated = modulation.NormAffine(backend, img, 1e-6f);
            int channels = _config.LatentChannels;
            using Tensor projected = new(new TensorShape(tokens, channels), DType.F32);
            using Tensor flat = modulated.Reshape(new TensorShape(tokens, hidden));
            backend.Linear(projected, flat, _finalW!, _finalB!);
            Tensor velocity = new(new TensorShape([1L, channels, frames, height, width]), DType.F32);
            backend.Transpose2D(velocity, projected, tokens, channels);
            return velocity;
        }
        finally
        {
            modulation?.Dispose();
            modulated?.Dispose();
            img.Dispose();
        }
    }

    /// <summary>Checks the latent, conditioning and text inputs and returns the latent's frame count and spatial size.</summary>
    private (int Frames, int Height, int Width) ValidateInputs(Tensor latent, Tensor qwen, Tensor? byt5, Tensor? conditioning)
    {
        RequireF32(latent, nameof(latent));
        if (latent.Shape.Rank != 5 || latent.Shape[0] != 1 || latent.Shape[1] != _config.LatentChannels)
            throw new ArgumentException($"latent must be [1, {_config.LatentChannels}, T, H, W], got {latent.Shape}.", nameof(latent));
        int frames = (int)latent.Shape[2], height = (int)latent.Shape[3], width = (int)latent.Shape[4];
        ValidateText(qwen, byt5);
        if (conditioning is not null)
        {
            RequireF32(conditioning, nameof(conditioning));
            int extra = _config.ImgInChannels - _config.LatentChannels;
            if (conditioning.Shape.Rank != 5 || conditioning.Shape[0] != 1 || conditioning.Shape[1] != extra
                || conditioning.Shape[2] != frames || conditioning.Shape[3] != height || conditioning.Shape[4] != width)
                throw new ArgumentException($"conditioning must be [1, {extra}, {frames}, {height}, {width}], got {conditioning.Shape}.", nameof(conditioning));
        }
        return (frames, height, width);
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
        using Tensor tokenMajor3 = tokenMajor.Reshape(new TensorShape(1, tokens, _config.ImgInChannels));
        Tensor img = new(new TensorShape(1, tokens, _config.HiddenSize), DType.F32);
        backend.Linear(img, tokenMajor3, _imgInW!, _imgInB!);
        tokenMajor.Dispose();
        return img;
    }

    /// <summary>Sinusoid(256) then Linear, SiLU, Linear: <c>time_in.mlp</c>, for one clip-wide timestep. Returns <c>[1, hidden]</c>.</summary>
    private Tensor TimestepMlp(IBackend backend, float timestep) => TimestepMlpRows(backend, [timestep]);

    /// <summary>The <c>time_in.mlp</c> of each timestep in one batched pass: <c>[timesteps.Length, hidden]</c>, row <c>f</c> for <c>timesteps[f]</c>.</summary>
    private Tensor TimestepMlpRows(IBackend backend, float[] timesteps)
    {
        int rows = timesteps.Length;
        int hidden = _config.HiddenSize;
        using Tensor sinusoid = new(new TensorShape(rows, TimestepEmbedDim), DType.F32);
        using (Tensor row = new(new TensorShape(1, TimestepEmbedDim), DType.F32))
        {
            Span<float> dst = sinusoid.AsSpan<float>();
            for (int r = 0; r < rows; r++)
            {
                DiTUtils.SinusoidalTimestepEmbedding(row, timesteps[r], 1, TimestepEmbedDim);
                row.AsReadOnlySpan<float>().CopyTo(dst.Slice(r * TimestepEmbedDim, TimestepEmbedDim));
            }
        }
        using Tensor first = new(new TensorShape(rows, hidden), DType.F32);
        backend.Linear(first, sinusoid, _timeW0!, _timeB0!);
        using Tensor activated = new(first.Shape, DType.F32);
        backend.Silu(activated, first);
        Tensor temb = new(new TensorShape(rows, hidden), DType.F32);
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
        // An owned [1, C, T, H, W] tensor (not a Reshape view, which would leave its parent to the finalizer once per step).
        Tensor velocity = new(new TensorShape([1L, channels, frames, height, width]), DType.F32);
        backend.Transpose2D(velocity, projected, tokens, channels);
        return velocity;
    }

    /// <summary>The token tensors are read as raw floats (the refiner's mean pool, the channel copy), so they must be F32 and have the expected layout.</summary>
    private void ValidateText(Tensor qwen, Tensor? byt5)
    {
        RequireF32(qwen, nameof(qwen));
        if (qwen.Shape.Rank != 3 || qwen.Shape[0] != 1 || qwen.Shape[1] < 1 || qwen.Shape[2] != _config.TextDim)
            throw new ArgumentException($"qwen must be [1, L > 0, {_config.TextDim}], got {qwen.Shape}.", nameof(qwen));
        if (byt5 is null) return;
        RequireF32(byt5, nameof(byt5));
        if (byt5.Shape.Rank != 3 || byt5.Shape[0] != 1 || byt5.Shape[1] < 1 || byt5.Shape[2] != _config.ByT5Dim)
            throw new ArgumentException($"byt5 must be [1, L > 0, {_config.ByT5Dim}], got {byt5.Shape}.", nameof(byt5));
    }

    private static void RequireF32(Tensor tensor, string name)
    {
        if (tensor.DType != DType.F32)
            throw new ArgumentException($"{name} must be F32, got {tensor.DType}.", name);
    }

    private static Tensor F32(IReadOnlyDictionary<string, Tensor> w, string key) =>
        w.TryGetValue(key, out Tensor? tensor) ? TensorCasts.EnsureF32(tensor) : throw new KeyNotFoundException($"HunyuanVideo-1.5 DiT is missing '{key}'.");

    /// <inheritdoc/>
    public void Dispose() => _rope.Dispose();
}
