using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Swin;

namespace HartsyInference.Vision.Detection.GroundingDino;

/// <summary>Faithful Swin-Transformer-Tiny backbone for Grounding DINO (<c>model.backbone.conv_encoder.model.*</c>):
/// 4x4 patch embed → 96-dim, four stages (depths [2,2,6,2], heads [3,6,12,24]) of shifted-window multi-head
/// self-attention with relative-position bias and patch merging between stages. Exports the stage2/3/4
/// pre-downsampling feature maps (channels 192/384/768), each passed through its <c>hidden_states_norms</c> LayerNorm
/// and returned as <c>[1, C, H, W]</c>. The window attention, MLP and patch-merging math is <see cref="SwinOps"/>,
/// shared with the BiRefNet Swin encoder; this class owns the HF key layout and the export points.</summary>
public sealed unsafe class SwinBackbone : IDisposable
{
    private readonly GroundingDinoConfig _cfg;
    private readonly int _numStages;
    private Tensor? _patchProjW, _patchProjB, _embNormW, _embNormB;
    private readonly SwinStageWeights[] _stages;
    private readonly Dictionary<string, Tensor> _outNorms = new();   // "stage2/3/4" -> {W,B} via suffix
    private int _disposed;

    public SwinBackbone(GroundingDinoConfig cfg)
    {
        _cfg = cfg;
        _numStages = cfg.SwinDepths.Length;
        _stages = new SwinStageWeights[_numStages];
        for (int s = 0; s < _numStages; s++)
            _stages[s] = new SwinStageWeights();
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "model.backbone.conv_encoder.model")
    {
        _patchProjW = w[$"{prefix}.embeddings.patch_embeddings.projection.weight"];
        _patchProjB = w[$"{prefix}.embeddings.patch_embeddings.projection.bias"];
        _embNormW = w[$"{prefix}.embeddings.norm.weight"];
        _embNormB = w[$"{prefix}.embeddings.norm.bias"];
        for (int s = 0; s < _numStages; s++)
        {
            string sp = $"{prefix}.encoder.layers.{s}";
            SwinStageWeights sw = _stages[s];
            int depth = _cfg.SwinDepths[s];
            sw.Blocks = new SwinOps.BlockWeights[depth];
            for (int b = 0; b < depth; b++)
            {
                string bp = $"{sp}.blocks.{b}";
                sw.Blocks[b] = new SwinOps.BlockWeights
                {
                    LnBeforeW = w[$"{bp}.layernorm_before.weight"], LnBeforeB = w[$"{bp}.layernorm_before.bias"],
                    QW = w[$"{bp}.attention.self.query.weight"], QB = w[$"{bp}.attention.self.query.bias"],
                    KW = w[$"{bp}.attention.self.key.weight"], KB = w[$"{bp}.attention.self.key.bias"],
                    VW = w[$"{bp}.attention.self.value.weight"], VB = w[$"{bp}.attention.self.value.bias"],
                    RelBiasTable = w[$"{bp}.attention.self.relative_position_bias_table"],
                    OW = w[$"{bp}.attention.output.dense.weight"], OB = w[$"{bp}.attention.output.dense.bias"],
                    LnAfterW = w[$"{bp}.layernorm_after.weight"], LnAfterB = w[$"{bp}.layernorm_after.bias"],
                    Fc1W = w[$"{bp}.intermediate.dense.weight"], Fc1B = w[$"{bp}.intermediate.dense.bias"],
                    Fc2W = w[$"{bp}.output.dense.weight"], Fc2B = w[$"{bp}.output.dense.bias"],
                };
            }
            if (s < _numStages - 1)
            {
                sw.DownReductionW = w[$"{sp}.downsample.reduction.weight"];
                sw.DownNormW = w[$"{sp}.downsample.norm.weight"];
                sw.DownNormB = w[$"{sp}.downsample.norm.bias"];
            }
        }
        for (int i = 0; i < _cfg.BackboneOutChannels.Length; i++)
        {
            string stageName = $"stage{i + 2}";
            _outNorms[$"{stageName}.weight"] = w[$"{prefix}.hidden_states_norms.{stageName}.weight"];
            _outNorms[$"{stageName}.bias"] = w[$"{prefix}.hidden_states_norms.{stageName}.bias"];
        }
    }

    /// <summary>Runs the backbone. Returns the three exported feature maps as <c>[1, C, H, W]</c> (stage2/3/4).</summary>
    public Tensor[] Forward(IBackend backend, Tensor pixelValues)
    {
        int ps = _cfg.SwinPatchSize;
        // maybe_pad pixel_values so H,W divisible by patch size (bottom/right, zeros).
        Tensor padded = SwinOps.PadToMultiple(pixelValues, ps, out int pH, out int pW);
        int gH = pH / ps, gW = pW / ps, embed = _cfg.SwinEmbedDim;

        Tensor conv = new(new TensorShape(1, embed, gH, gW), DType.F32);
        backend.Conv2D(conv, padded, _patchProjW!, _patchProjB, ps, ps, 0, 0);
        if (!ReferenceEquals(padded, pixelValues)) padded.Dispose();

        // [1,embed,gH,gW] -> [1, gH*gW, embed]
        int tokens = gH * gW;
        Tensor x = SwinOps.ChannelsToTokens(conv, embed, tokens);
        conv.Dispose();
        Tensor xn = new(x.Shape, DType.F32);
        backend.LayerNorm(xn, x, _embNormW!, _embNormB!, _cfg.SwinLayerNormEps);
        x.Dispose();
        x = xn;

        int curH = gH, curW = gW, dim = embed;
        List<Tensor> outputs = new();
        for (int s = 0; s < _numStages; s++)
        {
            int heads = _cfg.SwinNumHeads[s];
            SwinOps.BlockWeights[] blocks = _stages[s].Blocks;
            for (int b = 0; b < blocks.Length; b++)
            {
                int shift = (b % 2 == 1) ? _cfg.SwinWindowSize / 2 : 0;
                Tensor next = SwinOps.Block(backend, x, curH, curW, dim, heads, _cfg.SwinWindowSize, shift, _cfg.SwinLayerNormEps, blocks[b]);
                x.Dispose();
                x = next;
            }

            // Export before-downsampling output for stages 1,2,3 (named stage2/3/4).
            if (s >= 1)
            {
                string stageName = $"stage{s + 1}";
                outputs.Add(SwinOps.LayerNormToNchw(backend, x, curH, curW, dim, _outNorms[$"{stageName}.weight"], _outNorms[$"{stageName}.bias"], 1e-5f));
            }

            if (s < _numStages - 1)
            {
                Tensor merged = SwinOps.PatchMerging(backend, x, curH, curW, dim, _stages[s].DownNormW!, _stages[s].DownNormB!, _stages[s].DownReductionW!, _cfg.SwinLayerNormEps);
                x.Dispose();
                x = merged;
                curH = (curH + 1) / 2;
                curW = (curW + 1) / 2;
                dim *= 2;
            }
        }
        x.Dispose();
        return outputs.ToArray();
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }

    private sealed class SwinStageWeights
    {
        public SwinOps.BlockWeights[] Blocks = Array.Empty<SwinOps.BlockWeights>();
        public Tensor? DownReductionW, DownNormW, DownNormB;
    }
}
