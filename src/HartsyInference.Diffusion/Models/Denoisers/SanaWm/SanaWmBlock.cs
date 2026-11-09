using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Backends.Norm;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.SanaWm;

/// <summary>One SANA-WM DiT block in upstream order: gated attention (Gated-DeltaNet or softmax, summed with the camera
/// branch, output gate, projection), plucker residual, ungated cross-attention over the text, then gated GLUMBConvTemp FFN.
/// Every adaLN modulation is per frame.</summary>
/// <remarks>Linear and conv weights are borrowed from the checkpoint in their stored dtype (BF16), so the GEMMs read them
/// directly; vectors and tables are F32. The residual stream and the attention are F32. The FFN runs in BF16 on a backend
/// with BF16 support, with the checkpoint's BF16 kernels, and in F32 elsewhere, with host F32 copies of its kernels (the CPU
/// convolutions read F32 only). The attention runs one batch item at a time.</remarks>
public sealed class SanaWmBlock : IDisposable
{
    private const float LayerNormEpsilon = 1e-6f;
    private const float CrossNormEpsilon = 1e-6f;
    private const int ShiftMsa = 0, ScaleMsa = 1, GateMsa = 2, ShiftMlp = 3, ScaleMlp = 4, GateMlp = 5;
    private const int ModulationSlots = SanaWmConditioning.ModulationSlots;

    /// <summary>Cross-attention queries per SDPA call, bounding its score buffer when the backend materializes one.</summary>
    private const int CrossQueryChunk = 8192;

    private readonly IBackend _backend;
    private readonly List<Tensor> _owned = [];
    private readonly int _heads;
    private readonly int _headDim;
    private readonly int _channels;
    private readonly SanaWmGdnBidi? _gdn;
    private readonly SanaWmSoftmaxAttention? _softmaxAttention;
    private readonly Tensor _outputGateWeight;
    private readonly Tensor _outputGateBias;
    private readonly Tensor _projWeight;
    private readonly Tensor _projBias;
    private readonly Tensor? _pluckerWeight;
    private readonly Tensor? _pluckerBias;
    private readonly Tensor _crossQWeight;
    private readonly Tensor _crossQBias;
    private readonly Tensor _crossKvWeight;
    private readonly Tensor _crossKvBias;
    private readonly Tensor _crossProjWeight;
    private readonly Tensor _crossProjBias;
    private readonly Tensor _crossQNorm;
    private readonly Tensor _crossKNorm;
    private readonly Tensor _scaleShift;
    private readonly Tensor _scaleShiftOnes;
    private readonly SanaWmGlumbTempFfn _ffn;
    private readonly string _profileTag;
    private SanaWmCamBranch? _gdnCamera;
    private SanaWmSoftmaxCamBranch? _softmaxCamera;
    private int _disposed;

    /// <param name="backend">Backend the block runs on.</param>
    /// <param name="config">Checkpoint dimensions; the head dim must be 112.</param>
    /// <param name="index">Block index, selecting the softmax or GDN attention and the <c>blocks.{index}.</c> keys.</param>
    /// <param name="weights">Checkpoint tensors keyed by upstream name. They must outlive the block.</param>
    public SanaWmBlock(IBackend backend, SanaWmConfig config, int index, IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(weights);
        if (index < 0 || index >= config.Depth)
            throw new ArgumentOutOfRangeException(nameof(index), index, $"Block index must be in [0, {config.Depth}).");

        _backend = backend;
        _heads = config.NumHeads;
        _headDim = config.HeadDim;
        _channels = config.HiddenSize;
        bool softmax = config.IsSoftmaxBlock(index);
        _profileTag = $"b{index:D2}.{(softmax ? "sm" : "gdn")}";
        string prefix = $"blocks.{index}.";
        Func<string, Tensor> get = key => weights[prefix + key];
        bool built = false;
        try
        {
            if (softmax)
            {
                _softmaxAttention = new SanaWmSoftmaxAttention(backend, get("attn.qkv.weight"), Vector(get("attn.q_norm.weight")),
                    Vector(get("attn.k_norm.weight")), _heads, _headDim);
            }
            else
            {
                _gdn = new SanaWmGdnBidi(backend, new SanaWmGdnWeights
                {
                    Qkv = get("attn.qkv.weight"),
                    ConvK = get("attn.conv_k.weight"),
                    BetaWeight = get("attn.beta_proj.weight"),
                    BetaBias = get("attn.beta_proj.bias"),
                    GateWeight = get("attn.gate_proj.weight"),
                    GateBias = get("attn.gate_proj.bias"),
                    ALog = get("attn.A_log"),
                    DtBias = get("attn.dt_bias"),
                    QNorm = get("attn.q_norm.weight"),
                    KNorm = get("attn.k_norm.weight"),
                }, _heads, _headDim);
            }

            if (config.HasCameraBranch)
                BuildCamera(prefix, weights, softmax);

            _outputGateWeight = get("attn.output_gate.weight");
            _outputGateBias = Vector(get("attn.output_gate.bias"));
            _projWeight = get("attn.proj.weight");
            _projBias = Vector(get("attn.proj.bias"));
            if (config.HasPluckerBranch)
            {
                _pluckerWeight = get("plucker_proj.weight");
                _pluckerBias = Vector(get("plucker_proj.bias"));
            }

            _crossQWeight = get("cross_attn.q_linear.weight");
            _crossQBias = Vector(get("cross_attn.q_linear.bias"));
            _crossKvWeight = get("cross_attn.kv_linear.weight");
            _crossKvBias = Vector(get("cross_attn.kv_linear.bias"));
            _crossProjWeight = get("cross_attn.proj.weight");
            _crossProjBias = Vector(get("cross_attn.proj.bias"));
            _crossQNorm = Vector(get("cross_attn.q_norm.weight"));
            _crossKNorm = Vector(get("cross_attn.k_norm.weight"));
            _scaleShift = Vector(get("scale_shift_table"));
            _scaleShiftOnes = Own(SanaWmTensors.Filled(new TensorShape((long)ModulationSlots * _channels), 1f));

            DType ffnDtype = backend.Capabilities.SupportsBF16 ? DType.BF16 : DType.F32;
            _ffn = new SanaWmGlumbTempFfn(
                FfnKernel(get("mlp.inverted_conv.conv.weight"), ffnDtype),
                Vector(get("mlp.inverted_conv.conv.bias")),
                FfnKernel(get("mlp.depth_conv.conv.weight"), ffnDtype),
                Vector(get("mlp.depth_conv.conv.bias")),
                FfnKernel(get("mlp.point_conv.conv.weight"), ffnDtype),
                FfnKernel(get("mlp.t_conv.weight"), ffnDtype));
            built = true;
        }
        finally
        {
            if (!built)
                Dispose();
        }
    }

    /// <summary>The tensors the block reads on every forward, for preloading on a device backend.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        IEnumerable<Tensor> attention = _gdn?.EnumerateWeights() ?? _softmaxAttention!.EnumerateWeights();
        IEnumerable<Tensor> camera = _gdnCamera?.EnumerateWeights() ?? _softmaxCamera?.EnumerateWeights() ?? [];
        Tensor?[] own = [_outputGateWeight, _outputGateBias, _projWeight, _projBias, _pluckerWeight, _pluckerBias, _crossQWeight,
            _crossQBias, _crossKvWeight, _crossKvBias, _crossProjWeight, _crossProjBias, _crossQNorm, _crossKNorm, _scaleShift,
            _scaleShiftOnes];
        foreach (Tensor tensor in attention.Concat(camera).Concat(_ffn.EnumerateWeights()))
            yield return tensor;
        foreach (Tensor? tensor in own)
        {
            if (tensor is not null)
                yield return tensor;
        }
    }

    /// <summary>Runs the block on F32 tokens <c>[B, N, C]</c>, frame-major with <c>N = frames·latentHeight·latentWidth</c>, and
    /// returns a new owned F32 <c>[B, N, C]</c> tensor. <paramref name="x"/> is left untouched.</summary>
    public Tensor Forward(Tensor x, SanaWmForwardContext context) => Run(x, context, ownsInput: false);

    /// <summary><see cref="Forward"/> that takes ownership of <paramref name="x"/> and releases it once the attention residual no
    /// longer needs it, so a block stack holds one residual stream at a time. <paramref name="x"/> is disposed even on failure.</summary>
    internal Tensor ForwardOwned(Tensor x, SanaWmForwardContext context) => Run(x, context, ownsInput: true);

    /// <inheritdoc />
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        _gdn?.Dispose();
        _gdnCamera?.Dispose();
        _softmaxCamera?.Dispose();
        SanaWmTensors.DisposeAll(_owned);
    }

    private Tensor Run(Tensor x, SanaWmForwardContext context, bool ownsInput)
    {
        using SanaWmScratch scratch = new();
        if (ownsInput)
            scratch.Track(x);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        using StageProfiler.Scope blockScope = StageProfiler.Begin(_backend, _profileTag, "block");
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(context);
        int batch = context.Batch;
        int frames = context.Frames;
        int spatial = context.Spatial;
        int tokens = context.Tokens;
        SanaWmTensors.Require(x, nameof(x), DType.F32, [batch, tokens, _channels]);
        if ((_gdnCamera is not null || _softmaxCamera is not null) && context.Camera is null)
            throw new ArgumentException("The block has a camera branch, but the context carries no camera inputs.", nameof(context));

        long modulationWidth = (long)ModulationSlots * _channels;
        Tensor modulation = scratch.New((long)batch * frames, modulationWidth);
        _backend.AffineBroadcastLastDim(modulation, context.BlockTime, _scaleShiftOnes, _scaleShift);
        Tensor gateMsa = Gate(scratch, modulation, GateMsa, batch * frames);
        Tensor gateMlp = Gate(scratch, modulation, GateMlp, batch * frames);

        Tensor attentionInput = scratch.New((long)batch * frames, spatial, _channels);
        Modulated(attentionInput, x, modulation, ShiftMsa, ScaleMsa, spatial);
        Tensor attention;
        using (StageProfiler.Begin(_backend, _profileTag, "attn"))
            attention = scratch.Track(Attention(attentionInput, context));
        scratch.Release(attentionInput);
        Tensor afterAttention = scratch.New(x.Shape);
        _backend.GatedResidualRowIndexed(afterAttention, x, attention, gateMsa, context.RowIndex);
        scratch.Release(attention);
        if (ownsInput)
            scratch.Release(x);

        Tensor afterPlucker = afterAttention;
        if (_pluckerWeight is not null && context.Plucker is not null)
        {
            using StageProfiler.Scope pluckerScope = StageProfiler.Begin(_backend, _profileTag, "plucker_proj");
            Tensor pluckerTokens = scratch.New(x.Shape);
            _backend.Linear(pluckerTokens, context.Plucker, _pluckerWeight, _pluckerBias);
            afterPlucker = scratch.New(x.Shape);
            _backend.Add(afterPlucker, afterAttention, pluckerTokens);
            scratch.Release(pluckerTokens);
            scratch.Release(afterAttention);
        }

        Tensor cross;
        using (StageProfiler.Begin(_backend, _profileTag, "cross_attn"))
            cross = scratch.Track(CrossAttention(afterPlucker, context));
        Tensor afterCross = scratch.New(x.Shape);
        _backend.Add(afterCross, afterPlucker, cross);
        scratch.Release(cross);
        scratch.Release(afterPlucker);

        Tensor ffnInput = scratch.New(x.Shape);
        Modulated(ffnInput, afterCross, modulation, ShiftMlp, ScaleMlp, spatial);
        Tensor ffn;
        using (StageProfiler.Begin(_backend, _profileTag, "mlp"))
            ffn = scratch.Track(FeedForward(ffnInput, context));
        scratch.Release(ffnInput);
        Tensor output = scratch.New(x.Shape);
        _backend.GatedResidualRowIndexed(output, afterCross, ffn, gateMlp, context.RowIndex);
        return scratch.Detach(output);
    }

    private void BuildCamera(string prefix, IReadOnlyDictionary<string, Tensor> weights, bool softmax)
    {
        Dictionary<string, Tensor> camera = new(StringComparer.Ordinal);
        string[] keys = ["q_proj_cam.weight", "q_proj_cam.bias", "k_proj_cam.weight", "k_proj_cam.bias", "v_proj_cam.weight",
            "v_proj_cam.bias", "q_norm_cam.weight", "k_norm_cam.weight", "out_proj_cam.weight", "out_proj_cam.bias"];
        foreach (string key in keys)
            camera[key] = weights[prefix + "attn." + key];

        if (softmax)
        {
            _softmaxCamera = new SanaWmSoftmaxCamBranch(_heads, _headDim);
            _softmaxCamera.LoadWeights(camera);
            return;
        }

        camera["conv_k_cam.weight"] = weights[prefix + "attn.conv_k_cam.weight"];
        _gdnCamera = new SanaWmCamBranch(_heads, _headDim);
        _gdnCamera.LoadWeights(camera);
    }

    // LayerNorm (no affine) of every token, then x·(1 + scale) + shift with the token's frame's slots of the packed modulation.
    private void Modulated(Tensor output, Tensor input, Tensor modulation, int shiftSlot, int scaleSlot, int spatial)
    {
        long width = (long)ModulationSlots * _channels;
        ModulationSpec perFrame = ModulationSpec.Packed(modulation, width, (long)scaleSlot * _channels, (long)shiftSlot * _channels,
            spatial, ScaleMode.OnePlus);
        _backend.NormModulate(NormModulateCall.Rows(output, input, NormKind.LayerNorm, LayerNormEpsilon).WithModulation(perFrame));
    }

    // One slot of the packed [B·F, 6C] modulation as its own [B·F, C] table, the form the row-indexed gated residual reads.
    private Tensor Gate(SanaWmScratch scratch, Tensor modulation, int slot, int rows)
    {
        Tensor gate = scratch.New(rows, _channels);
        ReadOnlySpan<long> dims = [rows, _channels];
        ReadOnlySpan<long> strides = [(long)ModulationSlots * _channels, 1];
        _backend.CopyStrided(gate, modulation, StridedCopySpec.Gather(new StridedView(dims, strides, (long)slot * _channels)));
        return gate;
    }

    // Attention on [B·F, S, C] tokens, one batch item at a time: the main branch, the camera branch summed in, then the output
    // gate and projection. The output gate reads the attention input, as upstream's _apply_output_gate(combined, x) does.
    private Tensor Attention(Tensor input, SanaWmForwardContext context)
    {
        int batch = context.Batch;
        int tokens = context.Tokens;
        if (batch == 1)
            return AttentionItem(input, context, 0);

        long itemElements = (long)tokens * _channels;
        using SanaWmScratch scratch = new();
        Tensor output = scratch.New(new TensorShape(batch, tokens, _channels));
        for (int b = 0; b < batch; b++)
        {
            using SanaWmScratch item = new();
            Tensor a = item.Track(SanaWmTensors.Slice(_backend, input, b * itemElements,
                new TensorShape(context.Frames, context.Spatial, _channels)));
            Tensor projected = item.Track(AttentionItem(a, context, b));
            SanaWmTensors.Write(_backend, output, projected, b * itemElements);
        }

        return scratch.Detach(output);
    }

    // One batch item: a is F32 [F, S, C]; returns proj(branch · SiLU(output_gate(a))) as [N, C].
    private Tensor AttentionItem(Tensor a, SanaWmForwardContext context, int item)
    {
        int frames = context.Frames;
        int spatial = context.Spatial;
        int tokens = context.Tokens;
        using SanaWmScratch scratch = new();
        // Every projection of `a` (q, k, v, the camera's q, k, v and the output gate) reads it in the GEMM dtype; converting it
        // once here replaces the conversion each GEMM would otherwise make. Identical arithmetic: the GEMM rounds it the same way.
        Tensor? projection = _projWeight.DType == DType.BF16 && _backend.Capabilities.SupportsBF16
            ? scratch.Track(SanaWmTensors.ToBf16(_backend, a))
            : null;
        Tensor branch;
        if (_gdn is not null)
        {
            Tensor main, beta, decay;
            using (StageProfiler.Begin(_backend, _profileTag, "attn.main"))
                (main, beta, decay) = _gdn.ForwardItem(a, frames, spatial, context.RopeCos, context.RopeSin, _profileTag, projection);
            branch = scratch.Track(main);
            scratch.Track(beta);
            scratch.Track(decay);
            if (_gdnCamera is not null)
            {
                Tensor camera;
                using (StageProfiler.Begin(_backend, _profileTag, "attn.forward_cam_branch"))
                    camera = scratch.Track(_gdnCamera.ForwardItem(_backend, a, context.Camera!, item, beta, decay, frames, spatial, _profileTag,
                        projection));
                branch = Sum(scratch, main, camera);
            }

            scratch.Release(beta);
            scratch.Release(decay);
        }
        else
        {
            Tensor main;
            using (StageProfiler.Begin(_backend, _profileTag, "attn.main"))
                main = scratch.Track(_softmaxAttention!.ForwardItem(a, tokens, context.RopeCos, context.RopeSin, projection));
            branch = main;
            if (_softmaxCamera is not null)
            {
                Tensor camera;
                using (StageProfiler.Begin(_backend, _profileTag, "attn.forward_cam_branch"))
                    camera = scratch.Track(_softmaxCamera.ForwardItem(_backend, a, context.Camera!, item, tokens, projection));
                branch = Sum(scratch, main, camera);
            }
        }

        Tensor gated = scratch.New(tokens, _channels);
        using (StageProfiler.Begin(_backend, _profileTag, "attn.apply_output_gate"))
        {
            Tensor gateLinear = scratch.New(tokens, _channels);
            _backend.Linear(gateLinear, projection ?? a, _outputGateWeight, _outputGateBias);
            Tensor gateActivated = scratch.New(tokens, _channels);
            _backend.Silu(gateActivated, gateLinear);
            scratch.Release(gateLinear);
            _backend.Mul(gated, branch, gateActivated);
            scratch.Release(branch);
            scratch.Release(gateActivated);
        }

        Tensor projected = scratch.New(tokens, _channels);
        using (StageProfiler.Begin(_backend, _profileTag, "attn.proj"))
            _backend.Linear(projected, gated, _projWeight, _projBias);
        return scratch.Detach(projected);
    }

    // Text cross-attention: q from the tokens, k and v from the caption, RMSNorm over the full width before the head split,
    // no RoPE. The mask is additive on padded caption keys. Queries run per batch item in chunks; each row is independent.
    private Tensor CrossAttention(Tensor x, SanaWmForwardContext context)
    {
        int batch = context.Batch;
        int tokens = context.Tokens;
        long textTokens = context.TextTokens;
        float scale = MathF.Pow(_headDim, -0.5f);
        using SanaWmScratch scratch = new();
        Tensor keys = TextHeads(scratch, context.Memory, batch, textTokens, 0, _crossKNorm);
        Tensor values = TextHeads(scratch, context.Memory, batch, textTokens, _channels, null);

        int chunk = Math.Min(tokens, CrossQueryChunk);
        if (batch == 1 && chunk == tokens)
            return CrossChunk(x, keys, values, context.MemoryBias, tokens, scale);

        long headElements = (long)_heads * textTokens * _headDim;
        TensorShape headShape = new(1, _heads, textTokens, _headDim);
        Tensor output = scratch.New(x.Shape);
        for (int b = 0; b < batch; b++)
        {
            using SanaWmScratch item = new();
            Tensor itemKeys = batch == 1 ? keys : item.Track(SanaWmTensors.Slice(_backend, keys, b * headElements, headShape));
            Tensor itemValues = batch == 1 ? values : item.Track(SanaWmTensors.Slice(_backend, values, b * headElements, headShape));
            Tensor? itemBias = context.MemoryBias is null || batch == 1 ? context.MemoryBias
                : item.Track(SanaWmTensors.Slice(_backend, context.MemoryBias, b * textTokens, new TensorShape(1, 1, 1, textTokens)));
            for (int start = 0; start < tokens; start += chunk)
            {
                int rows = Math.Min(chunk, tokens - start);
                long offset = ((long)b * tokens + start) * _channels;
                using SanaWmScratch step = new();
                Tensor queries = step.Track(SanaWmTensors.Slice(_backend, x, offset, new TensorShape(rows, _channels)));
                Tensor attended = step.Track(CrossChunk(queries, itemKeys, itemValues, itemBias, rows, scale));
                SanaWmTensors.Write(_backend, output, attended, offset);
            }
        }

        return scratch.Detach(output);
    }

    // Keys (normed) or values of the caption, head-major [B, H, L, D], from the fused kv projection.
    private Tensor TextHeads(SanaWmScratch scratch, Tensor memory, int batch, long textTokens, int rowOffset, Tensor? norm)
    {
        long rows = batch * textTokens;
        Tensor projected = scratch.New(rows, _channels);
        _backend.LinearWeightRows(projected, memory, _crossKvWeight, _crossKvBias, rowOffset, _channels);
        if (norm is not null)
        {
            Tensor normed = scratch.New(rows, _channels);
            _backend.RmsNorm(normed, projected, norm, CrossNormEpsilon);
            scratch.Release(projected);
            projected = normed;
        }

        Tensor heads = scratch.New(batch, _heads, textTokens, _headDim);
        _backend.Permute0213(heads, projected, (int)textTokens, _heads, _headDim);
        scratch.Release(projected);
        return heads;
    }

    // proj(SDPA(RMSNorm(q(x)), k, v)) for one batch item's rows [rows, C]; returns [rows, C].
    private Tensor CrossChunk(Tensor x, Tensor keys, Tensor values, Tensor? bias, int rows, float scale)
    {
        using SanaWmScratch scratch = new();
        Tensor query = scratch.New(rows, _channels);
        _backend.Linear(query, x, _crossQWeight, _crossQBias);
        Tensor queryNormed = scratch.New(rows, _channels);
        _backend.RmsNorm(queryNormed, query, _crossQNorm, CrossNormEpsilon);
        scratch.Release(query);
        Tensor queryHeads = scratch.New(1, _heads, rows, _headDim);
        _backend.Permute0213(queryHeads, queryNormed, rows, _heads, _headDim);
        scratch.Release(queryNormed);
        Tensor attended = scratch.New(1, _heads, rows, _headDim);
        // BF16 operands with F32 softmax and accumulation, as upstream's BF16 cross-attention.
        _backend.Attention(AttentionCall.FromScaledDotProductAttention(attended, queryHeads, keys, values, bias, scale, allowF16: false)
            with { Numerics = AttentionNumerics.Bf16Storage });
        scratch.Release(queryHeads);
        Tensor merged = scratch.New(rows, _channels);
        _backend.Permute0213(merged, attended, _heads, rows, _headDim);
        scratch.Release(attended);
        Tensor projected = scratch.New(rows, _channels);
        _backend.Linear(projected, merged, _crossProjWeight, _crossProjBias);
        return scratch.Detach(projected);
    }

    // The modulated FFN input [B, N, C] F32 through the FFN in its dtype; returns F32 [B, N, C].
    private Tensor FeedForward(Tensor input, SanaWmForwardContext context)
    {
        using SanaWmScratch scratch = new();
        Tensor ffnInput = input.DType == _ffn.DType ? input : scratch.Track(SanaWmTensors.ToBf16(_backend, input));
        Tensor output = scratch.Track(_ffn.Forward(_backend, ffnInput, context.Frames, context.Height, context.Width, _profileTag));
        return output.DType == DType.F32 ? scratch.Detach(output) : SanaWmTensors.ToF32(_backend, output);
    }

    private Tensor Sum(SanaWmScratch scratch, Tensor a, Tensor b)
    {
        Tensor sum = scratch.New(a.Shape);
        _backend.Add(sum, a, b);
        scratch.Release(a);
        scratch.Release(b);
        return sum;
    }

    private Tensor Vector(Tensor source) => SanaWmTensors.HostF32(source, _owned);

    // The FFN's conv kernels in its dtype: borrowed when the checkpoint already stores them so, else a host copy the block owns.
    private Tensor FfnKernel(Tensor source, DType dtype)
    {
        if (source.DType == dtype)
            return source;
        return Own(source.CastTo(dtype));
    }

    private Tensor Own(Tensor tensor)
    {
        _owned.Add(tensor);
        return tensor;
    }
}
