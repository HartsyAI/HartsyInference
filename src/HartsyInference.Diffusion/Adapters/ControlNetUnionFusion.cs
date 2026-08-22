using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.UNetBlocks;

namespace HartsyInference.Diffusion.Adapters;

/// <summary>The union-specific modules of a xinsir-style union SDXL ControlNet (diffusers <c>ControlNetUnionModel</c>): a learned per-control-type <c>task_embedding</c> table, a small pre-LN transformer (<c>transformer_layes</c> — sic, the checkpoint key carries the typo) that fuses pooled condition features with the pooled latent feature, a zero-initialized <c>spatial_ch_projs</c> Linear that turns the fused condition token into a per-channel bias, and a <c>control_add_embedding</c> MLP that injects a sinusoid-encoded multi-hot control-type vector into the timestep embedding. Replaces the plain ControlNet's <c>hidden = conv_in(latent) + cond_hint</c> injection with <c>hidden = conv_in(latent) + cond_hint + alpha</c> where <c>alpha</c> is the transformer-fused channel bias, and adds <c>control_emb</c> onto the timestep embedding before the ADM addition embedding — everything downstream (down blocks, mid block, zero convs) is the standard SDXL ControlNet path. The fusion transformer is a faithful port of the reference: diffusers feeds the <c>[B, tokens, C]</c> token stack straight into <c>nn.MultiheadAttention</c> with the default <c>batch_first=False</c>, so attention runs over the diffusion-batch dim independently per token — the tokens NEVER interact (LayerNorm/MLP are per-token too). Two consequences we exploit, both bit-identical to running the full stack (verified against the diffusers oracle at B=1 and B=2): the pooled-latent token the reference appends is dead compute for the single-condition case (its transformed value is discarded — <c>spatial_ch_projs</c> only consumes the condition tokens), and the condition token can be run through the layers alone as a <c>[1, B, C]</c> stack with B as the SDPA sequence dim. This also keeps the forward off <see cref="IBackend.Split"/>, whose CUDA implementation is a host fallback (mid-forward sync trap).</summary>
public sealed class ControlNetUnionFusion : IDisposable
{
    private readonly int _numControlTypes;
    private readonly int _channels;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _additionTimeEmbedDim;
    private int _disposed;

    // ── Owned tensors (allocated at LoadWeights from checkpoint data) ──
    private Tensor?[] _taskEmbeddingRows = [];
    private TransformerLayer[] _layers = [];

    // ── Borrowed tensors (owned by the safetensors loader) ────────────
    private Tensor? _spatialProjWeight, _spatialProjBias;
    private readonly TimestepEmbedding _controlAddEmbedding;

    // Cached ones/(H·W) weight for the spatial-mean Linear, keyed on the spatial size it was
    // built for. Field-stable reference so GPU backends upload it once and cache-hit afterwards.
    private Tensor? _meanWeight;
    private int _meanWeightSpatial = -1;

    /// <summary>Creates the union fusion modules. <paramref name="channels"/> is the transformer/task-embedding width (<c>num_trans_channel</c>, equal to the base UNet's model channels: 320). <paramref name="timeDim"/> is the timestep-embedding width the control-type MLP projects to (1280 for SDXL).</summary>
    public ControlNetUnionFusion(int numControlTypes, int channels, int timeDim, int additionTimeEmbedDim, int numHeads = 8)
    {
        if (numControlTypes <= 0)
            throw new ArgumentException($"Union ControlNet needs a positive control-type count; got {numControlTypes}.", nameof(numControlTypes));
        _numControlTypes = numControlTypes;
        _channels = channels;
        _numHeads = numHeads;
        _headDim = channels / numHeads;
        _additionTimeEmbedDim = additionTimeEmbedDim;
        _controlAddEmbedding = new TimestepEmbedding(additionTimeEmbedDim * numControlTypes, timeDim);
    }

    /// <summary>Number of rows in the checkpoint's task-embedding table (6 for the standard union revision, 8 for ProMax).</summary>
    public int NumControlTypes => _numControlTypes;

    /// <summary>Loads the union modules from a diffusers-layout weight dictionary: <c>task_embedding</c>, <c>transformer_layes.{i}.*</c>, <c>spatial_ch_projs.*</c>, <c>control_add_embedding.*</c>. The fused <c>in_proj</c> QKV weights and the task-embedding rows are split into owned per-part tensors here (row-contiguous copies) so the forward pass can use plain <see cref="IBackend.Linear"/>/<see cref="IBackend.Add"/> calls.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        Tensor taskTable = weights["task_embedding"];
        if ((int)taskTable.Shape[0] != _numControlTypes || (int)taskTable.Shape[1] != _channels)
        {
            throw new HartsyInferenceException(
                $"Union ControlNet task_embedding shape {taskTable.Shape} doesn't match the expected [{_numControlTypes}, {_channels}].");
        }
        _taskEmbeddingRows = new Tensor?[_numControlTypes];
        for (int i = 0; i < _numControlTypes; i++)
        {
            _taskEmbeddingRows[i] = CopyRows(taskTable, rowStart: i, rowCount: 1, rowWidth: _channels, new TensorShape(1, _channels));
        }

        int layerCount = 0;
        while (weights.ContainsKey($"transformer_layes.{layerCount}.ln_1.weight")) layerCount++;
        if (layerCount == 0)
        {
            throw new HartsyInferenceException("Union ControlNet checkpoint has no transformer_layes.* fusion layers.");
        }
        _layers = new TransformerLayer[layerCount];
        for (int i = 0; i < layerCount; i++)
        {
            string p = $"transformer_layes.{i}";
            Tensor inProjW = weights[$"{p}.attn.in_proj_weight"];
            Tensor inProjB = weights[$"{p}.attn.in_proj_bias"];
            if ((int)inProjW.Shape[0] != 3 * _channels || (int)inProjW.Shape[1] != _channels)
            {
                throw new HartsyInferenceException(
                    $"Union fusion layer {i} in_proj_weight shape {inProjW.Shape} doesn't match the expected [{3 * _channels}, {_channels}].");
            }
            TensorShape wShape = new TensorShape(_channels, _channels);
            TensorShape bShape = new TensorShape(_channels);
            _layers[i] = new TransformerLayer
            {
                QWeight = CopyRows(inProjW, 0, _channels, _channels, wShape),
                KWeight = CopyRows(inProjW, _channels, _channels, _channels, wShape),
                VWeight = CopyRows(inProjW, 2 * _channels, _channels, _channels, wShape),
                QBias = CopyRows(inProjB, 0, _channels, 1, bShape),
                KBias = CopyRows(inProjB, _channels, _channels, 1, bShape),
                VBias = CopyRows(inProjB, 2 * _channels, _channels, 1, bShape),
                OutWeight = weights[$"{p}.attn.out_proj.weight"],
                OutBias = weights[$"{p}.attn.out_proj.bias"],
                Ln1Weight = weights[$"{p}.ln_1.weight"],
                Ln1Bias = weights[$"{p}.ln_1.bias"],
                Ln2Weight = weights[$"{p}.ln_2.weight"],
                Ln2Bias = weights[$"{p}.ln_2.bias"],
                FcWeight = weights[$"{p}.mlp.c_fc.weight"],
                FcBias = weights[$"{p}.mlp.c_fc.bias"],
                ProjWeight = weights[$"{p}.mlp.c_proj.weight"],
                ProjBias = weights[$"{p}.mlp.c_proj.bias"],
            };
        }

        _spatialProjWeight = weights["spatial_ch_projs.weight"];
        _spatialProjBias = weights["spatial_ch_projs.bias"];
        _controlAddEmbedding.LoadWeights(weights, "control_add_embedding");
    }

    /// <summary>Computes the control-type embedding added onto the timestep embedding: each entry of the multi-hot <c>[numControlTypes]</c> vector (1 at <paramref name="controlTypeIndex"/>, 0 elsewhere) is sinusoid-encoded exactly like an SDXL micro-conditioning scalar, the encodings are concatenated to <c>[B, 256·numControlTypes]</c>, and the <c>control_add_embedding</c> MLP projects to <c>[B, timeDim]</c> F32.</summary>
    public unsafe Tensor ComputeControlEmbedding(IBackend backend, int controlTypeIndex, int batch)
    {
        ThrowIfDisposed();
        ValidateControlTypeIndex(controlTypeIndex);
        int totalDim = _additionTimeEmbedDim * _numControlTypes;
        Tensor controlEmbeds = new Tensor(new TensorShape(batch, totalDim), DType.F32);
        float* ptr = (float*)controlEmbeds.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            for (int j = 0; j < _numControlTypes; j++)
            {
                float value = j == controlTypeIndex ? 1f : 0f;
                AdditionEmbedding.EmbedScalar(ptr, b * totalDim + j * _additionTimeEmbedDim, value, _additionTimeEmbedDim);
            }
        }
        Tensor output = _controlAddEmbedding.ForwardEmbedding(backend, controlEmbeds);
        controlEmbeds.Dispose();
        return output;
    }

    /// <summary>Fuses the conditioning hint into the conv_in output: pools the hint to a per-channel feature token, adds the task embedding, runs the token through the fusion transformer, and returns <c>convInSample + condHint + spatial_ch_projs(fusedCondToken)</c> as a new tensor. Inputs are not disposed.</summary>
    public Tensor FuseConditionHint(IBackend backend, Tensor convInSample, Tensor condHint, int controlTypeIndex)
    {
        ThrowIfDisposed();
        ValidateControlTypeIndex(controlTypeIndex);
        int batch = (int)convInSample.Shape[0];
        int spatial = (int)(convInSample.ElementCount / (batch * _channels));
        DType dtype = convInSample.DType;

        // 1. Pooled condition token [1, B, C] + task embedding (the reference's pooled-latent
        //    token is dead compute for the single-condition case — see class doc).
        Tensor condToken = SpatialMean(backend, condHint, batch, spatial, dtype);
        Tensor taskRow = TaskRowForBatch(backend, controlTypeIndex, batch, dtype);
        Tensor tokens = new Tensor(condToken.Shape, dtype);
        backend.Add(tokens, condToken, taskRow);
        condToken.Dispose();
        if (!ReferenceEquals(taskRow, _taskEmbeddingRows[controlTypeIndex])) taskRow.Dispose();

        // 2. Fusion transformer on the [1, B, C] token stack.
        for (int i = 0; i < _layers.Length; i++)
        {
            Tensor next = ForwardTransformerLayer(backend, tokens, _layers[i], batch, dtype);
            tokens.Dispose();
            tokens = next;
        }

        // 3. alpha = spatial_ch_projs(fused condition token).
        Tensor alpha = new Tensor(new TensorShape(batch, _channels), dtype);
        backend.Linear(alpha, tokens, _spatialProjWeight!, _spatialProjBias);
        tokens.Dispose();

        // 4. fused = convInSample + condHint + alpha (per-batch per-channel broadcast over spatial).
        Tensor fused = new Tensor(convInSample.Shape, dtype);
        backend.Add(fused, convInSample, condHint);
        backend.BroadcastAdd(fused, alpha, _channels, spatial);
        alpha.Dispose();
        return fused;
    }

    /// <summary>Yields all weight tensors (owned splits + borrowed checkpoint tensors) for GPU preloading / freeing.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? row in _taskEmbeddingRows) if (row is not null) yield return row;
        foreach (TransformerLayer layer in _layers)
        {
            foreach (Tensor w in layer.EnumerateWeights()) yield return w;
        }
        if (_spatialProjWeight is not null) yield return _spatialProjWeight;
        if (_spatialProjBias is not null) yield return _spatialProjBias;
        foreach (Tensor w in _controlAddEmbedding.EnumerateWeights()) yield return w;
    }

    /// <summary>One pre-LN residual attention block: <c>x += attn(ln_1(x)); x += c_proj(quickgelu(c_fc(ln_2(x))))</c>. Operates on the [tokens, B, C] stack with tokens as the SDPA batch dim and B as the sequence dim — the exact semantic of the reference's batch_first=False MultiheadAttention (see class doc).</summary>
    private Tensor ForwardTransformerLayer(IBackend backend, Tensor x, TransformerLayer layer, int batch, DType dtype)
    {
        int tokens = (int)x.Shape[0];
        TensorShape stackShape = new TensorShape(tokens, batch, _channels);
        TensorShape mhShape = new TensorShape(tokens, _numHeads, batch, _headDim);

        Tensor normed = new Tensor(stackShape, dtype);
        backend.LayerNorm(normed, x, layer.Ln1Weight!, layer.Ln1Bias!, 1e-5f);

        Tensor query = new Tensor(stackShape, dtype);
        backend.Linear(query, normed, layer.QWeight!, layer.QBias);
        Tensor key = new Tensor(stackShape, dtype);
        backend.Linear(key, normed, layer.KWeight!, layer.KBias);
        Tensor value = new Tensor(stackShape, dtype);
        backend.Linear(value, normed, layer.VWeight!, layer.VBias);
        normed.Dispose();

        Tensor queryMh = new Tensor(mhShape, dtype);
        backend.Permute0213(queryMh, query, batch, _numHeads, _headDim);
        query.Dispose();
        Tensor keyMh = new Tensor(mhShape, dtype);
        backend.Permute0213(keyMh, key, batch, _numHeads, _headDim);
        key.Dispose();
        Tensor valueMh = new Tensor(mhShape, dtype);
        backend.Permute0213(valueMh, value, batch, _numHeads, _headDim);
        value.Dispose();

        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor attnOut = new Tensor(mhShape, dtype);
        backend.ScaledDotProductAttention(attnOut, queryMh, keyMh, valueMh, null, scale);
        queryMh.Dispose();
        keyMh.Dispose();
        valueMh.Dispose();

        Tensor merged = new Tensor(stackShape, dtype);
        backend.Permute0213(merged, attnOut, _numHeads, batch, _headDim);
        attnOut.Dispose();
        Tensor projected = new Tensor(stackShape, dtype);
        backend.Linear(projected, merged, layer.OutWeight!, layer.OutBias);
        merged.Dispose();
        Tensor afterAttn = new Tensor(stackShape, dtype);
        backend.Add(afterAttn, x, projected);
        projected.Dispose();

        Tensor normed2 = new Tensor(stackShape, dtype);
        backend.LayerNorm(normed2, afterAttn, layer.Ln2Weight!, layer.Ln2Bias!, 1e-5f);
        int mlpDim = (int)layer.FcWeight!.Shape[0];
        TensorShape mlpShape = new TensorShape(tokens, batch, mlpDim);
        Tensor fc = new Tensor(mlpShape, dtype);
        backend.Linear(fc, normed2, layer.FcWeight!, layer.FcBias);
        normed2.Dispose();

        // QuickGELU: x · sigmoid(1.702·x) — the reference's fast GELU approximation, kept
        // rather than the exact-erf Gelu op so activations match bit-for-bit in spirit.
        Tensor scaled = new Tensor(mlpShape, dtype);
        backend.Scale(scaled, fc, 1.702f);
        backend.Sigmoid(scaled, scaled);
        Tensor gated = new Tensor(mlpShape, dtype);
        backend.Mul(gated, fc, scaled);
        fc.Dispose();
        scaled.Dispose();

        Tensor mlpOut = new Tensor(stackShape, dtype);
        backend.Linear(mlpOut, gated, layer.ProjWeight!, layer.ProjBias);
        gated.Dispose();
        Tensor output = new Tensor(stackShape, dtype);
        backend.Add(output, afterAttn, mlpOut);
        afterAttn.Dispose();
        mlpOut.Dispose();
        return output;
    }

    /// <summary>Mean over the spatial dims of a <c>[B, C, H, W]</c> tensor via a ones/(H·W) Linear (single flat GEMV — both backends derive rows from element count, so no reshape view is needed). Returns <c>[1, B, C]</c>, which is one token row of the [tokens, B, C] fusion stack.</summary>
    private Tensor SpatialMean(IBackend backend, Tensor input, int batch, int spatial, DType dtype)
    {
        if (_meanWeight is null || _meanWeightSpatial != spatial)
        {
            _meanWeight?.Dispose();
            Tensor weight = new Tensor(new TensorShape(1, spatial), DType.F32);
            weight.AsSpan<float>().Fill(1f / spatial);
            _meanWeight = weight;
            _meanWeightSpatial = spatial;
        }
        Tensor mean = new Tensor(new TensorShape(1, batch, _channels), dtype);
        backend.Linear(mean, input, _meanWeight, null);
        return mean;
    }

    /// <summary>The task-embedding row for the control type, cast to the activation dtype and replicated along the batch dim when B &gt; 1 so it can be added to a <c>[1, B, C]</c> token with a plain element-wise Add. Returns the stored row itself when no cast/replication is needed — the caller only disposes the result when it differs from the stored row.</summary>
    private Tensor TaskRowForBatch(IBackend backend, int controlTypeIndex, int batch, DType dtype)
    {
        Tensor row = _taskEmbeddingRows[controlTypeIndex]!;
        Tensor typed = Utilities.DtypeCastHelper.EnsureDtype(backend, row, dtype, disposeSourceOnCast: false);
        if (batch == 1)
        {
            return typed;
        }
        Tensor replicated = new Tensor(new TensorShape(1, batch, _channels), dtype);
        Tensor[] copies = new Tensor[batch];
        for (int b = 0; b < batch; b++) copies[b] = typed;
        backend.Concat(replicated, copies, 0);
        if (!ReferenceEquals(typed, row)) typed.Dispose();
        return replicated;
    }

    /// <summary>Copies <paramref name="rowCount"/> contiguous rows starting at <paramref name="rowStart"/> out of a row-major 2-D (or 1-D, with <paramref name="rowWidth"/> = 1) checkpoint tensor into a new owned tensor of the same dtype. Raw byte copy — dtype-agnostic, load-time only (the source is still host-resident).</summary>
    private static Tensor CopyRows(Tensor source, int rowStart, int rowCount, int rowWidth, TensorShape shape)
    {
        Tensor dest = new Tensor(shape, source.DType);
        int elemBytes = source.DType.SizeInBytes;
        long byteOffset = (long)rowStart * rowWidth * elemBytes;
        long byteCount = (long)rowCount * rowWidth * elemBytes;
        ReadOnlySpan<byte> src = source.AsReadOnlySpan<byte>().Slice((int)byteOffset, (int)byteCount);
        src.CopyTo(dest.AsSpan<byte>());
        return dest;
    }

    private void ValidateControlTypeIndex(int controlTypeIndex)
    {
        if (controlTypeIndex < 0 || controlTypeIndex >= _numControlTypes)
        {
            throw new HartsyInferenceException(
                $"Union control-type index {controlTypeIndex} is out of range for this checkpoint's {_numControlTypes}-row task embedding " +
                $"(the standard union revision has 6 types; Tile/Repaint need the 8-type ProMax revision).");
        }
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
    }

    /// <summary>Disposes the owned split tensors (task rows, QKV splits, mean weight); borrowed checkpoint tensors are only unreferenced.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            foreach (Tensor? row in _taskEmbeddingRows) row?.Dispose();
            Array.Clear(_taskEmbeddingRows);
            foreach (TransformerLayer layer in _layers) layer.DisposeOwned();
            _layers = [];
            _spatialProjWeight = null;
            _spatialProjBias = null;
            _meanWeight?.Dispose();
            _meanWeight = null;
        }
        GC.SuppressFinalize(this);
    }

    /// <summary>One fusion transformer layer's weights. Q/K/V weights and biases are owned splits of the checkpoint's fused <c>in_proj</c>; the rest are borrowed from the loader's dictionary.</summary>
    private sealed class TransformerLayer
    {
        public Tensor? QWeight, KWeight, VWeight;
        public Tensor? QBias, KBias, VBias;
        public Tensor? OutWeight, OutBias;
        public Tensor? Ln1Weight, Ln1Bias;
        public Tensor? Ln2Weight, Ln2Bias;
        public Tensor? FcWeight, FcBias;
        public Tensor? ProjWeight, ProjBias;

        public IEnumerable<Tensor> EnumerateWeights()
        {
            Tensor?[] all = [QWeight, KWeight, VWeight, QBias, KBias, VBias, OutWeight, OutBias,
                Ln1Weight, Ln1Bias, Ln2Weight, Ln2Bias, FcWeight, FcBias, ProjWeight, ProjBias];
            foreach (Tensor? t in all) if (t is not null) yield return t;
        }

        public void DisposeOwned()
        {
            QWeight?.Dispose(); KWeight?.Dispose(); VWeight?.Dispose();
            QBias?.Dispose(); KBias?.Dispose(); VBias?.Dispose();
            QWeight = null; KWeight = null; VWeight = null;
            QBias = null; KBias = null; VBias = null;
            OutWeight = null; OutBias = null; Ln1Weight = null; Ln1Bias = null;
            Ln2Weight = null; Ln2Bias = null; FcWeight = null; FcBias = null;
            ProjWeight = null; ProjBias = null;
        }
    }
}
