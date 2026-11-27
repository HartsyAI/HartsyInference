using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Swin;

namespace HartsyInference.Vision.Matting.BiRefNet;

/// <summary>The BiRefNet encoder: a Swin-L transformer (embed 192, depths [2,2,18,2], heads [6,12,24,48], window 12)
/// returning the four stage maps (1/4…1/32, 192/384/768/1536 channels), each passed through its own output LayerNorm.
/// Weights use the upstream <c>bb.*</c> names with fused qkv already split by the converter.</summary>
internal sealed unsafe class BiRefNetSwinBackbone
{
    private const int Embed = 192;
    private const int WindowSize = 12;
    private const float LayerNormEps = 1e-5f;
    private static readonly int[] Depths = [2, 2, 18, 2];
    private static readonly int[] Heads = [6, 12, 24, 48];

    private Tensor? _patchW, _patchB, _patchNormW, _patchNormB;
    private readonly SwinOps.BlockWeights[][] _blocks = new SwinOps.BlockWeights[4][];
    private readonly Tensor?[] _downW = new Tensor?[3], _downNormW = new Tensor?[3], _downNormB = new Tensor?[3];
    private readonly Tensor?[] _outNormW = new Tensor?[4], _outNormB = new Tensor?[4];

    /// <summary>Loads every <c>bb.*</c> tensor; <paramref name="owned"/> collects tensors cast to F32.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, List<Tensor> owned)
    {
        _patchW = BiRefNetOps.Own(w["bb.patch_embed.proj.weight"], owned);
        _patchB = BiRefNetOps.Own(w["bb.patch_embed.proj.bias"], owned);
        _patchNormW = BiRefNetOps.Own(w["bb.patch_embed.norm.weight"], owned);
        _patchNormB = BiRefNetOps.Own(w["bb.patch_embed.norm.bias"], owned);
        for (int s = 0; s < 4; s++)
        {
            _blocks[s] = new SwinOps.BlockWeights[Depths[s]];
            for (int b = 0; b < Depths[s]; b++)
                _blocks[s][b] = LoadBlock(w, $"bb.layers.{s}.blocks.{b}", owned);
            if (s < 3)
            {
                _downW[s] = BiRefNetOps.Own(w[$"bb.layers.{s}.downsample.reduction.weight"], owned);
                _downNormW[s] = BiRefNetOps.Own(w[$"bb.layers.{s}.downsample.norm.weight"], owned);
                _downNormB[s] = BiRefNetOps.Own(w[$"bb.layers.{s}.downsample.norm.bias"], owned);
            }
            _outNormW[s] = BiRefNetOps.Own(w[$"bb.norm{s}.weight"], owned);
            _outNormB[s] = BiRefNetOps.Own(w[$"bb.norm{s}.bias"], owned);
        }
    }

    /// <summary>Runs the encoder on <c>[1,3,H,W]</c> (H, W multiples of 32). Returns the stage maps <c>[1,C,H/2^(s+2),W/2^(s+2)]</c>.</summary>
    public Tensor[] Forward(IBackend backend, Tensor image)
    {
        Tensor padded = SwinOps.PadToMultiple(image, 4, out int pH, out int pW);
        int gh = pH / 4, gw = pW / 4;
        Tensor conv = new(new TensorShape(1, Embed, gh, gw), DType.F32);
        backend.Conv2D(conv, padded, _patchW!, _patchB, 4, 4, 0, 0);
        if (!ReferenceEquals(padded, image)) padded.Dispose();

        Tensor tokens = SwinOps.ChannelsToTokens(conv, Embed, gh * gw);
        conv.Dispose();
        Tensor x = new(tokens.Shape, DType.F32);
        backend.LayerNorm(x, tokens, _patchNormW!, _patchNormB!, LayerNormEps);
        tokens.Dispose();

        Tensor[] outputs = new Tensor[4];
        int curH = gh, curW = gw, dim = Embed;
        for (int s = 0; s < 4; s++)
        {
            int heads = Heads[s];
            for (int b = 0; b < Depths[s]; b++)
            {
                int shift = b % 2 == 0 ? 0 : WindowSize / 2;
                Tensor next = SwinOps.Block(backend, x, curH, curW, dim, heads, WindowSize, shift, LayerNormEps, _blocks[s][b]);
                x.Dispose();
                x = next;
            }
            outputs[s] = SwinOps.LayerNormToNchw(backend, x, curH, curW, dim, _outNormW[s]!, _outNormB[s]!, LayerNormEps);
            if (s < 3)
            {
                Tensor merged = SwinOps.PatchMerging(backend, x, curH, curW, dim, _downNormW[s]!, _downNormB[s]!, _downW[s]!, LayerNormEps);
                x.Dispose();
                x = merged;
                curH = (curH + 1) / 2;
                curW = (curW + 1) / 2;
                dim *= 2;
            }
        }
        x.Dispose();
        return outputs;
    }

    private static SwinOps.BlockWeights LoadBlock(IReadOnlyDictionary<string, Tensor> w, string bp, List<Tensor> owned) => new()
    {
        LnBeforeW = BiRefNetOps.Own(w[$"{bp}.norm1.weight"], owned), LnBeforeB = BiRefNetOps.Own(w[$"{bp}.norm1.bias"], owned),
        QW = BiRefNetOps.Own(w[$"{bp}.attn.q.weight"], owned), QB = BiRefNetOps.Own(w[$"{bp}.attn.q.bias"], owned),
        KW = BiRefNetOps.Own(w[$"{bp}.attn.k.weight"], owned), KB = BiRefNetOps.Own(w[$"{bp}.attn.k.bias"], owned),
        VW = BiRefNetOps.Own(w[$"{bp}.attn.v.weight"], owned), VB = BiRefNetOps.Own(w[$"{bp}.attn.v.bias"], owned),
        RelBiasTable = BiRefNetOps.Own(w[$"{bp}.attn.relative_position_bias_table"], owned),
        OW = BiRefNetOps.Own(w[$"{bp}.attn.proj.weight"], owned), OB = BiRefNetOps.Own(w[$"{bp}.attn.proj.bias"], owned),
        LnAfterW = BiRefNetOps.Own(w[$"{bp}.norm2.weight"], owned), LnAfterB = BiRefNetOps.Own(w[$"{bp}.norm2.bias"], owned),
        Fc1W = BiRefNetOps.Own(w[$"{bp}.mlp.fc1.weight"], owned), Fc1B = BiRefNetOps.Own(w[$"{bp}.mlp.fc1.bias"], owned),
        Fc2W = BiRefNetOps.Own(w[$"{bp}.mlp.fc2.weight"], owned), Fc2B = BiRefNetOps.Own(w[$"{bp}.mlp.fc2.bias"], owned),
    };
}
