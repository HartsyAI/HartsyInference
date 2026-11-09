using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.SanaWm;

/// <summary>One SANA-WM DiT block in upstream order: gated attention (Gated-DeltaNet or softmax, summed with the camera
/// branch, output gate, projection), plucker residual, ungated cross-attention over the text, then gated GLUMBConvTemp FFN.
/// Every adaLN modulation is per frame. Weights are borrowed from the checkpoint; F32 copies are owned by the block.</summary>
public sealed class SanaWmBlock : IDisposable
{
    private const float LayerNormEpsilon = 1e-6f;
    private const float CrossNormEpsilon = 1e-6f;
    private const int ShiftMsa = 0, ScaleMsa = 1, GateMsa = 2, ShiftMlp = 3, ScaleMlp = 4, GateMlp = 5;
    private const int ModulationSlots = 6;
    private const int CameraChannels = 20;

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
    private readonly SanaWmGlumbTempFfn _ffn;
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
        string prefix = $"blocks.{index}.";
        Func<string, Tensor> get = key => weights[prefix + key];
        bool built = false;
        try
        {
            if (softmax)
            {
                _softmaxAttention = new SanaWmSoftmaxAttention(backend, Owned(get("attn.qkv.weight")), get("attn.q_norm.weight"),
                    get("attn.k_norm.weight"), _heads, _headDim);
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

            _outputGateWeight = Owned(get("attn.output_gate.weight"));
            _outputGateBias = Owned(get("attn.output_gate.bias"));
            _projWeight = Owned(get("attn.proj.weight"));
            _projBias = Owned(get("attn.proj.bias"));
            if (config.HasPluckerBranch)
            {
                _pluckerWeight = Owned(get("plucker_proj.weight"));
                _pluckerBias = Owned(get("plucker_proj.bias"));
            }

            _crossQWeight = Owned(get("cross_attn.q_linear.weight"));
            _crossQBias = Owned(get("cross_attn.q_linear.bias"));
            _crossKvWeight = Owned(get("cross_attn.kv_linear.weight"));
            _crossKvBias = Owned(get("cross_attn.kv_linear.bias"));
            _crossProjWeight = Owned(get("cross_attn.proj.weight"));
            _crossProjBias = Owned(get("cross_attn.proj.bias"));
            _crossQNorm = get("cross_attn.q_norm.weight");
            _crossKNorm = get("cross_attn.k_norm.weight");
            _scaleShift = get("scale_shift_table");
            _ffn = new SanaWmGlumbTempFfn(
                Owned(get("mlp.inverted_conv.conv.weight")),
                Owned(get("mlp.inverted_conv.conv.bias")),
                Owned(get("mlp.depth_conv.conv.weight")),
                Owned(get("mlp.depth_conv.conv.bias")),
                Owned(get("mlp.point_conv.conv.weight")),
                Owned(get("mlp.t_conv.weight")));
            built = true;
        }
        finally
        {
            if (!built)
                Dispose();
        }
    }

    /// <summary>Runs the block on F32 tokens <c>[B, N, C]</c>, frame-major with <c>N = frames·latentHeight·latentWidth</c>.
    /// Returns a new owned F32 <c>[B, N, C]</c> tensor.</summary>
    /// <param name="x">Residual stream input.</param>
    /// <param name="memory">Text tokens after the caption projection, F32 <c>[B, L, C]</c>.</param>
    /// <param name="memoryBias">Additive cross-attention mask <c>[B, 1, 1, L]</c>, F32, or null for no padding.</param>
    /// <param name="timeModulation">F32 <c>[B·F, 6·C]</c> from the time embedding's <c>t_block</c>, one row per frame.</param>
    /// <param name="plucker">Projected plucker tokens <c>[B, N, C]</c> from the DiT, or null when the checkpoint has no plucker branch.</param>
    /// <param name="cameraConditions">Per-frame camera conditions, <c>B·F·20</c> floats; required with a camera branch.</param>
    /// <param name="ropeCos">Main 3D RoPE cosine table <c>[N, headDim]</c>, interleaved pairs.</param>
    /// <param name="ropeSin">Main 3D RoPE sine table in the same layout.</param>
    /// <param name="cameraCos">Per-complex RoPE cosine table <c>[N, headDim/2]</c> for the camera branch.</param>
    /// <param name="cameraSin">Per-complex RoPE sine table <c>[N, headDim/2]</c>.</param>
    public Tensor Forward(Tensor x, Tensor memory, Tensor? memoryBias, Tensor timeModulation, Tensor? plucker,
        ReadOnlySpan<float> cameraConditions, int frames, int latentHeight, int latentWidth, Tensor ropeCos, Tensor ropeSin,
        Tensor cameraCos, Tensor cameraSin)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(memory);
        ArgumentNullException.ThrowIfNull(timeModulation);
        if (frames <= 0 || latentHeight <= 0 || latentWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(frames), "Frames and latent grid must be positive.");
        if (x.DType != DType.F32 || x.Shape.Rank != 3 || x.Shape[2] != _channels)
            throw new ArgumentException($"x must be F32 [B, N, {_channels}]; got {x.DType} {x.Shape}.", nameof(x));

        long batch = x.Shape[0];
        int spatial = checked(latentHeight * latentWidth);
        int tokens = checked(frames * spatial);
        if (x.Shape[1] != tokens)
            throw new ArgumentException($"x must hold {tokens} tokens per batch item; got {x.Shape}.", nameof(x));
        long textTokens = memory.Shape.Rank == 3 ? memory.Shape[1] : -1;
        SanaWmTensors.Require(memory, nameof(memory), DType.F32, [batch, textTokens, _channels]);
        SanaWmTensors.Require(timeModulation, nameof(timeModulation), DType.F32, [batch * frames, ModulationSlots * _channels]);
        if (memoryBias is not null)
            SanaWmTensors.Require(memoryBias, nameof(memoryBias), DType.F32, [batch, 1, 1, textTokens]);
        if (plucker is not null)
            SanaWmTensors.Require(plucker, nameof(plucker), DType.F32, [batch, tokens, _channels]);
        if ((_gdnCamera is not null || _softmaxCamera is not null)
            && cameraConditions.Length != batch * frames * CameraChannels)
            throw new ArgumentException($"Camera conditions must hold {batch * frames * CameraChannels} floats.", nameof(cameraConditions));

        TensorShape rows = new([batch * tokens, _channels]);
        List<Tensor> scratch = [];
        Tensor output = new(x.Shape, DType.F32, _backend.Device);
        bool succeeded = false;
        try
        {
            Tensor modulation = ComposeModulation(scratch, timeModulation, batch * frames);
            Tensor xRows = x.Reshape(rows);

            Tensor attnInput = Modulated(scratch, LayerNorm(scratch, xRows), modulation, ShiftMsa, ScaleMsa, batch, frames, spatial);
            Tensor attention = Attention(scratch, attnInput.Reshape(x.Shape), cameraConditions, frames, latentHeight, latentWidth,
                ropeCos, ropeSin, cameraCos, cameraSin);
            Tensor afterAttention = Gated(scratch, xRows, attention, modulation, GateMsa, batch, frames, spatial);

            Tensor afterPlucker = afterAttention;
            if (plucker is not null)
            {
                Tensor pluckerTokens = NewTensor(scratch, rows);
                _backend.Linear(pluckerTokens, plucker.Reshape(rows), _pluckerWeight!, _pluckerBias);
                afterPlucker = Sum(scratch, afterAttention, pluckerTokens);
            }

            Tensor cross = CrossAttention(scratch, afterPlucker, memory, memoryBias, batch, tokens, textTokens);
            Tensor afterCross = Sum(scratch, afterPlucker, cross);

            Tensor ffnInput = Modulated(scratch, LayerNorm(scratch, afterCross), modulation, ShiftMlp, ScaleMlp, batch, frames, spatial);
            Tensor ffn = _ffn.Forward(_backend, ffnInput.Reshape(x.Shape), frames, latentHeight, latentWidth);
            scratch.Add(ffn);
            Tensor result = Gated(scratch, afterCross, ffn.Reshape(rows), modulation, GateMlp, batch, frames, spatial);

            _backend.CopyTo(output, result.Reshape(x.Shape));
            succeeded = true;
            return output;
        }
        finally
        {
            if (!succeeded)
                output.Dispose();
            SanaWmTensors.DisposeAll(scratch);
        }
    }

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

    private Tensor Owned(Tensor source) => SanaWmTensors.ToF32(_backend, source, _owned);

    private void BuildCamera(string prefix, IReadOnlyDictionary<string, Tensor> weights, bool softmax)
    {
        Dictionary<string, Tensor> camera = new(StringComparer.Ordinal);
        string[] keys = ["q_proj_cam.weight", "q_proj_cam.bias", "k_proj_cam.weight", "k_proj_cam.bias", "v_proj_cam.weight",
            "v_proj_cam.bias", "q_norm_cam.weight", "k_norm_cam.weight", "out_proj_cam.weight", "out_proj_cam.bias"];
        foreach (string key in keys)
            camera[key] = Owned(weights[prefix + "attn." + key]);

        if (softmax)
        {
            _softmaxCamera = new SanaWmSoftmaxCamBranch(_heads, _headDim);
            _softmaxCamera.LoadWeights(camera);
            return;
        }

        camera["conv_k_cam.weight"] = Owned(weights[prefix + "attn.conv_k_cam.weight"]);
        _gdnCamera = new SanaWmCamBranch(_heads, _headDim);
        _gdnCamera.LoadWeights(camera);
    }

    // Attention on [B, N, C] tokens: the main branch, the camera branch summed in, then the output gate and projection.
    // The output gate reads the attention input, as upstream's _apply_output_gate(combined, x) does.
    private Tensor Attention(List<Tensor> scratch, Tensor input, ReadOnlySpan<float> cameraConditions, int frames, int latentHeight,
        int latentWidth, Tensor ropeCos, Tensor ropeSin, Tensor cameraCos, Tensor cameraSin)
    {
        TensorShape rows = new([input.Shape[0] * input.Shape[1], _channels]);
        Tensor main;
        if (_gdn is not null)
        {
            (Tensor gdnOutput, Tensor beta, Tensor decay) = _gdn.ForwardWithGates(input, frames, latentHeight, latentWidth, ropeCos, ropeSin);
            scratch.Add(gdnOutput);
            scratch.Add(beta);
            scratch.Add(decay);
            main = gdnOutput;
            if (_gdnCamera is not null)
            {
                Tensor camera = _gdnCamera.Forward(_backend, input, cameraConditions, frames, latentHeight, latentWidth,
                    cameraCos, cameraSin, beta, decay);
                scratch.Add(camera);
                main = Sum(scratch, main, camera);
            }
        }
        else
        {
            main = _softmaxAttention!.Forward(input, frames, latentHeight, latentWidth, ropeCos, ropeSin);
            scratch.Add(main);
            if (_softmaxCamera is not null)
            {
                Tensor camera = _softmaxCamera.Forward(_backend, input, cameraConditions, frames, latentHeight, latentWidth,
                    cameraCos, cameraSin);
                scratch.Add(camera);
                main = Sum(scratch, main, camera);
            }
        }

        Tensor gateLinear = NewTensor(scratch, rows);
        _backend.Linear(gateLinear, input.Reshape(rows), _outputGateWeight, _outputGateBias);
        Tensor gateActivated = NewTensor(scratch, rows);
        _backend.Silu(gateActivated, gateLinear);
        Tensor gated = NewTensor(scratch, rows);
        _backend.Mul(gated, main.Reshape(rows), gateActivated);
        Tensor projected = NewTensor(scratch, rows);
        _backend.Linear(projected, gated, _projWeight, _projBias);
        return projected;
    }

    // Text cross-attention: q from the tokens, k and v from the caption, RMSNorm over the full width before the head split,
    // no RoPE. The mask is additive on padded caption keys and broadcasts over heads and queries.
    private Tensor CrossAttention(List<Tensor> scratch, Tensor xRows, Tensor memory, Tensor? memoryBias, long batch, int tokens,
        long textTokens)
    {
        TensorShape rows = xRows.Shape;
        TensorShape textRows = new([batch * textTokens, _channels]);
        TensorShape queryMajor = new([batch, tokens, _heads, _headDim]);
        TensorShape textMajor = new([batch, textTokens, _heads, _headDim]);
        TensorShape headMajorQuery = new([batch, _heads, tokens, _headDim]);
        TensorShape headMajorText = new([batch, _heads, textTokens, _headDim]);
        Tensor memoryRows = memory.Reshape(textRows);

        Tensor query = NewTensor(scratch, rows);
        _backend.Linear(query, xRows, _crossQWeight, _crossQBias);
        Tensor queryNormed = NewTensor(scratch, rows);
        _backend.RmsNorm(queryNormed, query, _crossQNorm, CrossNormEpsilon);

        Tensor key = NewTensor(scratch, textRows);
        _backend.LinearWeightRows(key, memoryRows, _crossKvWeight, _crossKvBias, 0, _channels);
        Tensor keyNormed = NewTensor(scratch, textRows);
        _backend.RmsNorm(keyNormed, key, _crossKNorm, CrossNormEpsilon);
        Tensor value = NewTensor(scratch, textRows);
        _backend.LinearWeightRows(value, memoryRows, _crossKvWeight, _crossKvBias, _channels, _channels);

        Tensor qh = NewTensor(scratch, headMajorQuery);
        _backend.Permute0213(qh, queryNormed.Reshape(queryMajor), tokens, _heads, _headDim);
        Tensor kh = NewTensor(scratch, headMajorText);
        _backend.Permute0213(kh, keyNormed.Reshape(textMajor), (int)textTokens, _heads, _headDim);
        Tensor vh = NewTensor(scratch, headMajorText);
        _backend.Permute0213(vh, value.Reshape(textMajor), (int)textTokens, _heads, _headDim);

        Tensor attended = NewTensor(scratch, headMajorQuery);
        _backend.ScaledDotProductAttention(attended, qh, kh, vh, memoryBias, MathF.Pow(_headDim, -0.5f));
        Tensor merged = NewTensor(scratch, rows);
        _backend.Permute0213(merged.Reshape(queryMajor), attended, _heads, tokens, _headDim);

        Tensor projected = NewTensor(scratch, rows);
        _backend.Linear(projected, merged, _crossProjWeight, _crossProjBias);
        return projected;
    }

    // [B·F·6, C] rows: slot k of frame f in batch item b is row (b·F + f)·6 + k. Scale slots hold 1 + scale, the modulate factor.
    private Tensor ComposeModulation(List<Tensor> scratch, Tensor timeModulation, long batchFrames)
    {
        using Tensor timeHost = SanaWmTensors.Readback(_backend, timeModulation);
        using Tensor tableHost = SanaWmTensors.Readback(_backend, _scaleShift);
        using Tensor composed = new(new TensorShape([batchFrames * ModulationSlots, _channels]), DType.F32, DeviceKind.Cpu);
        ReadOnlySpan<float> time = timeHost.AsReadOnlySpan<float>();
        ReadOnlySpan<float> table = tableHost.AsReadOnlySpan<float>();
        Span<float> destination = composed.AsSpan<float>();
        for (long row = 0; row < batchFrames; row++)
        {
            for (int slot = 0; slot < ModulationSlots; slot++)
            {
                float factor = slot == ScaleMsa || slot == ScaleMlp ? 1f : 0f;
                long target = (row * ModulationSlots + slot) * _channels;
                long source = row * ModulationSlots * _channels + slot * _channels;
                for (int c = 0; c < _channels; c++)
                    destination[(int)target + c] = table[slot * _channels + c] + time[(int)source + c] + factor;
            }
        }

        Tensor device = SanaWmTensors.Upload(_backend, composed);
        scratch.Add(device);
        return device;
    }

    // Modulate: out = in · scale + shift, per frame, with one shift and one scale vector per (batch item, frame).
    private Tensor Modulated(List<Tensor> scratch, Tensor normed, Tensor modulation, int shiftSlot, int scaleSlot, long batch,
        int frames, int spatial)
    {
        Tensor output = NewTensor(scratch, normed.Shape);
        long tokens = (long)frames * spatial;
        for (long b = 0; b < batch; b++)
        {
            for (int f = 0; f < frames; f++)
            {
                long row = b * tokens + (long)f * spatial;
                long modulationRow = (b * frames + f) * ModulationSlots;
                Tensor input = Track(scratch, normed.SliceRows(row, spatial));
                Tensor target = Track(scratch, output.SliceRows(row, spatial));
                Tensor shift = Track(scratch, modulation.SliceRows(modulationRow + shiftSlot, 1).Reshape(new TensorShape([_channels])));
                Tensor scale = Track(scratch, modulation.SliceRows(modulationRow + scaleSlot, 1).Reshape(new TensorShape([_channels])));
                _backend.AffineBroadcastLastDim(target, input, scale, shift);
            }
        }

        return output;
    }

    // Gated residual: out = residual + gate · delta, with the gate constant over each frame.
    private Tensor Gated(List<Tensor> scratch, Tensor residual, Tensor delta, Tensor modulation, int gateSlot, long batch,
        int frames, int spatial)
    {
        Tensor output = NewTensor(scratch, residual.Shape);
        Tensor scaled = NewTensor(scratch, residual.Shape);
        long tokens = (long)frames * spatial;
        for (long b = 0; b < batch; b++)
        {
            for (int f = 0; f < frames; f++)
            {
                long row = b * tokens + (long)f * spatial;
                long modulationRow = (b * frames + f) * ModulationSlots;
                Tensor gate = Track(scratch, modulation.SliceRows(modulationRow + gateSlot, 1).Reshape(new TensorShape([_channels])));
                Tensor deltaRows = Track(scratch, delta.SliceRows(row, spatial));
                Tensor scaledRows = Track(scratch, scaled.SliceRows(row, spatial));
                _backend.AffineBroadcastLastDim(scaledRows, deltaRows, gate, null);
                Tensor residualRows = Track(scratch, residual.SliceRows(row, spatial));
                Tensor outputRows = Track(scratch, output.SliceRows(row, spatial));
                _backend.Add(outputRows, residualRows, scaledRows);
            }
        }

        return output;
    }

    private Tensor LayerNorm(List<Tensor> scratch, Tensor input)
    {
        Tensor normed = NewTensor(scratch, input.Shape);
        _backend.LayerNormNoAffine(normed, input, LayerNormEpsilon);
        return normed;
    }

    private Tensor Sum(List<Tensor> scratch, Tensor a, Tensor b)
    {
        Tensor sum = NewTensor(scratch, a.Shape);
        _backend.Add(sum, a, b);
        return sum;
    }

    private Tensor NewTensor(List<Tensor> scratch, TensorShape shape)
    {
        Tensor tensor = new(shape, DType.F32, _backend.Device);
        scratch.Add(tensor);
        return tensor;
    }

    private static Tensor Track(List<Tensor> scratch, Tensor view)
    {
        scratch.Add(view);
        return view;
    }
}
