using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Matting.BiRefNet;

/// <summary>The BiRefNet matting network in eval mode (Swin-L encoder, squeeze block, decoder). The encoder runs twice:
/// once on the image and once on its half-resolution copy, and each stage pair is concatenated on channels (the
/// upstream <c>mul_scl_ipt='cat'</c>). The three shallower stages are then resized to the 1/32 map and prepended to it
/// (<c>cxt_num=3</c>) before the squeeze block. Output is the raw logit map; the caller applies the sigmoid.</summary>
internal sealed class BiRefNetModel : IDisposable
{
    private const int SqueezeGrid = 32;

    private readonly BiRefNetSwinBackbone _backbone = new();
    private readonly BiRefNetBasicDecBlk _squeeze = new();
    private readonly BiRefNetDecoder _decoder = new();
    private readonly List<Tensor> _owned = new();
    private int _disposed;

    /// <summary>Loads the converted checkpoint (see <c>BiRefNetCheckpointConverter</c>).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> converted)
    {
        ArgumentNullException.ThrowIfNull(converted);
        _backbone.LoadWeights(converted, _owned);
        _squeeze.LoadWeights(converted, "squeeze_module.0", _owned);
        _decoder.LoadWeights(converted, _owned);
    }

    /// <summary>Maps a normalized <c>[1,3,H,W]</c> image (H and W multiples of 32) to logits <c>[1,1,H,W]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor image)
    {
        int h = (int)image.Shape[2], w = (int)image.Shape[3];
        if (h % 32 != 0 || w % 32 != 0)
            throw new ArgumentException($"BiRefNet input must have height and width divisible by 32; got {h}x{w}.", nameof(image));

        Tensor[] full = _backbone.Forward(backend, image);
        Tensor half = BiRefNetOps.Resize(backend, image, h / 2, w / 2);
        Tensor[] small = _backbone.Forward(backend, half);
        half.Dispose();

        Tensor[] stages = new Tensor[4];
        for (int s = 0; s < 4; s++)
        {
            Tensor up = BiRefNetOps.Resize(backend, small[s], (int)full[s].Shape[2], (int)full[s].Shape[3]);
            stages[s] = BiRefNetOps.Concat(backend, full[s], up);
            up.Dispose(); full[s].Dispose(); small[s].Dispose();
        }

        Tensor c1 = BiRefNetOps.Resize(backend, stages[0], SqueezeGrid, SqueezeGrid);
        Tensor c2 = BiRefNetOps.Resize(backend, stages[1], SqueezeGrid, SqueezeGrid);
        Tensor c3 = BiRefNetOps.Resize(backend, stages[2], SqueezeGrid, SqueezeGrid);
        Tensor deepest = stages[3];
        Tensor context = BiRefNetOps.Concat(backend, c1, c2, c3, deepest);
        c1.Dispose(); c2.Dispose(); c3.Dispose(); deepest.Dispose();

        Tensor squeezed = _squeeze.Forward(backend, context);
        context.Dispose();

        Tensor logits = _decoder.Forward(backend, image, stages[0], stages[1], stages[2], squeezed);
        squeezed.Dispose();
        stages[0].Dispose(); stages[1].Dispose(); stages[2].Dispose();
        return logits;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        foreach (Tensor t in _owned) t.Dispose();
        _owned.Clear();
        GC.SuppressFinalize(this);
    }
}
