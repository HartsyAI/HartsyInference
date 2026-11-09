using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Matting.BiRefNet;

/// <summary>Upstream BiRefNet <c>Decoder</c> in eval mode. Four levels run from the 1/32 squeezed feature to 1/4, each
/// fusing a 1×1 lateral projection of the encoder skip with the image re-tiled into a grid of patches (<c>dec_ipt</c>),
/// then a basic decoder block and, at levels 4–2, a gradient gate. The 1/4 result is upsampled to full size and fused
/// with the full-resolution patch injection into the single-channel logit map.</summary>
internal sealed class BiRefNetDecoder
{
    /// <summary>Patch grid per injection site: the 1/32, 1/16, 1/8, 1/4 and full-resolution maps.</summary>
    private static readonly int[] PatchGrids = [32, 16, 8, 4, 1];

    private readonly BiRefNetSimpleConvs[] _ipt = [new(), new(), new(), new(), new()];
    private readonly BiRefNetBasicDecBlk[] _blocks = [new(), new(), new(), new()];
    private readonly BiRefNetGdtGate[] _gates = [new(), new(), new()];
    private readonly Tensor?[] _latW = new Tensor?[3], _latB = new Tensor?[3];
    private Tensor? _outW, _outB;

    /// <summary>Loads the decoder; <paramref name="owned"/> collects every F32 or folded tensor the model must free.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, List<Tensor> owned)
    {
        string[] iptNames = ["ipt_blk5", "ipt_blk4", "ipt_blk3", "ipt_blk2", "ipt_blk1"];
        for (int i = 0; i < _ipt.Length; i++)
            _ipt[i].LoadWeights(w, $"decoder.{iptNames[i]}", owned);

        string[] blockNames = ["decoder_block4", "decoder_block3", "decoder_block2", "decoder_block1"];
        for (int i = 0; i < _blocks.Length; i++)
            _blocks[i].LoadWeights(w, $"decoder.{blockNames[i]}", owned);

        string[] lateralNames = ["lateral_block4", "lateral_block3", "lateral_block2"];
        for (int i = 0; i < 3; i++)
        {
            _latW[i] = BiRefNetOps.Own(w[$"decoder.{lateralNames[i]}.conv.weight"], owned);
            _latB[i] = BiRefNetOps.Own(w[$"decoder.{lateralNames[i]}.conv.bias"], owned);
        }

        string[] gateLevels = ["4", "3", "2"];
        for (int i = 0; i < _gates.Length; i++)
            _gates[i].LoadWeights(w, $"decoder.gdt_convs_{gateLevels[i]}", $"decoder.gdt_convs_attn_{gateLevels[i]}", owned);

        _outW = BiRefNetOps.Own(w["decoder.conv_out1.0.weight"], owned);
        _outB = BiRefNetOps.Own(w["decoder.conv_out1.0.bias"], owned);
    }

    /// <summary>Decodes to the logit map <c>[1,1,H,W]</c>. <paramref name="image"/> is the normalized <c>[1,3,H,W]</c>
    /// network input; <paramref name="x1"/>…<paramref name="x4"/> are the 1/4…1/32 encoder features, x4 already squeezed.</summary>
    public Tensor Forward(IBackend backend, Tensor image, Tensor x1, Tensor x2, Tensor x3, Tensor x4)
    {
        int h = (int)image.Shape[2], w = (int)image.Shape[3];

        Tensor injected4 = Inject(backend, image, 0);
        Tensor deep = BiRefNetOps.Concat(backend, x4, injected4);
        injected4.Dispose();
        Tensor p4 = _blocks[0].Forward(backend, deep);
        deep.Dispose();
        Tensor p4g = _gates[0].Forward(backend, p4);
        p4.Dispose();

        Tensor p3 = Level(backend, image, p4g, x3, level: 1);
        p4g.Dispose();
        Tensor p3g = _gates[1].Forward(backend, p3);
        p3.Dispose();

        Tensor p2 = Level(backend, image, p3g, x2, level: 2);
        p3g.Dispose();
        Tensor p2g = _gates[2].Forward(backend, p2);
        p2.Dispose();

        Tensor p1 = Level(backend, image, p2g, x1, level: 3);
        p2g.Dispose();

        Tensor up = BiRefNetOps.Resize(backend, p1, h, w);
        p1.Dispose();
        Tensor injectedFull = Inject(backend, image, 4);
        Tensor full = BiRefNetOps.Concat(backend, up, injectedFull);
        up.Dispose(); injectedFull.Dispose();
        Tensor logits = BiRefNetOps.Conv(backend, full, _outW!, _outB, pad: 0);
        full.Dispose();
        return logits;
    }

    /// <summary>One intermediate level: upsample the coarser map to the skip, add the lateral projection, concatenate the
    /// level's patch injection, and run the level's decoder block. Returns the block output before any gate.</summary>
    private Tensor Level(IBackend backend, Tensor image, Tensor coarse, Tensor skip, int level)
    {
        int sh = (int)skip.Shape[2], sw = (int)skip.Shape[3];
        Tensor up = BiRefNetOps.Resize(backend, coarse, sh, sw);
        Tensor lateral = BiRefNetOps.Conv(backend, skip, _latW[level - 1]!, _latB[level - 1], pad: 0);
        Tensor summed = BiRefNetOps.Add(backend, up, lateral);
        up.Dispose(); lateral.Dispose();
        Tensor injected = Inject(backend, image, level);
        Tensor withPatches = BiRefNetOps.Concat(backend, summed, injected);
        summed.Dispose(); injected.Dispose();
        Tensor block = _blocks[level].Forward(backend, withPatches);
        withPatches.Dispose();
        return block;
    }

    /// <summary>Patch injection at a site: the image tiled into the site's grid, run through its SimpleConvs.</summary>
    private Tensor Inject(IBackend backend, Tensor image, int site)
    {
        Tensor patches = BiRefNetOps.Patchify(image, PatchGrids[site]);
        Tensor injected = _ipt[site].Forward(backend, patches);
        patches.Dispose();
        return injected;
    }
}
