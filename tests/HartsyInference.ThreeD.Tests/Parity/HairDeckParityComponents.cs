using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ThreeD.Models.HairDeck;

namespace HartsyInference.ThreeD.Tests.Parity;

/// <summary>The HairDeck components loaded once per test class from the released checkpoints, with the reference dump.</summary>
public sealed class HairDeckParityComponents : IDisposable
{
    private readonly SafeTensorsLoader _imageLoader = new();
    private readonly SafeTensorsLoader _colorLoader = new();
    private readonly SafeTensorsLoader _chainLoader = new();
    private readonly SafeTensorsLoader _vaeLoader = new();

    public HairDeckParityComponents()
    {
        _imageLoader.Load(HairDeckReferenceLayout.ImageModel);
        _colorLoader.Load(HairDeckReferenceLayout.ColorModel);
        _chainLoader.Load(HairDeckReferenceLayout.ChainModel);
        _vaeLoader.Load(HairDeckReferenceLayout.CardVae);

        Image = new HairDeckImageDit(HairDeckImageCheckpointConverter.Convert(_imageLoader.GetAllTensors()));
        Color = new HairDeckColorHead(HairDeckColorCheckpointConverter.Convert(_colorLoader.GetAllTensors()));
        Chain = new HairDeckChainDit(HairDeckChainCheckpointConverter.Convert(_chainLoader.GetAllTensors()));
        HairDeckStatsReader.Vectors output = HairDeckStatsReader.ReadOutput(HairDeckReferenceLayout.CardVaeStats, 5);
        Vae = new HairDeckCardVae(HairDeckCardVaeCheckpointConverter.Convert(_vaeLoader.GetAllTensors(), integrate: false), output.Mu, output.Sd);

        (float[] rootMu, float[] rootSd) = HairDeckStatsReader.ReadRoot(HairDeckReferenceLayout.TokenStats);
        HairDeckStatsReader.Vectors latent = HairDeckStatsReader.ReadLatent(HairDeckReferenceLayout.LatentStats, 32);
        Tokenizer = new HairDeckTokenizer(rootMu, rootSd, latent.Mu, latent.Sd);

        HairDeckStatsReader.Vectors chain = HairDeckStatsReader.ReadChain(HairDeckReferenceLayout.ChainStats, 26);
        ChainMu = chain.Mu;
        ChainSd = chain.Sd;
        Dump = HairDeckReferenceDump.Open(HairDeckReferenceLayout.Reference);
    }

    public HairDeckImageDit Image { get; }

    public HairDeckColorHead Color { get; }

    public HairDeckChainDit Chain { get; }

    public HairDeckCardVae Vae { get; }

    public HairDeckTokenizer Tokenizer { get; }

    public float[] ChainMu { get; }

    public float[] ChainSd { get; }

    public HairDeckReferenceDump Dump { get; }

    /// <summary>A new F32 tensor of <paramref name="values"/> with the given shape; the caller disposes it.</summary>
    public static Tensor Make(ReadOnlySpan<float> values, params int[] shape)
    {
        TensorShape s = shape.Length switch
        {
            1 => new TensorShape(shape[0]),
            2 => new TensorShape(shape[0], shape[1]),
            3 => new TensorShape(shape[0], shape[1], shape[2]),
            _ => throw new ArgumentException("Rank 1 to 3 only."),
        };
        Tensor t = new(s, DType.F32);
        if (values.Length != (int)t.ElementCount) throw new ArgumentException($"{values.Length} values for shape {s}.");
        unsafe
        {
            values.CopyTo(new Span<float>(t.DataPointer, values.Length));
        }
        return t;
    }

    public void Dispose()
    {
        _imageLoader.Dispose();
        _colorLoader.Dispose();
        _chainLoader.Dispose();
        _vaeLoader.Dispose();
    }
}
