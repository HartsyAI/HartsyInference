using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>Typed weights of the card VAE decoder and its point-count head (<c>hairdeck/cards/vae.py</c>, decoder <c>ada</c>).
/// The encoder (<c>enc.*</c>, <c>inp</c>, <c>to_z</c>, <c>cls</c>, <c>pe_enc</c>) is not read: sampling only decodes.</summary>
public static class HairDeckCardVaeCheckpointConverter
{
    /// <summary>Attention heads of the released decoder (<c>LoadedVAE</c> builds <c>CardVAE(..., 8, ...)</c>).</summary>
    public const int Heads = 8;

    /// <summary>Channels per control point of the decoder output: xyz, width, thickness.</summary>
    public const int OutputChannels = 5;

    /// <summary>One adaLN decoder block: 6-way modulation, fused QKV attention and the exact-GELU MLP.</summary>
    public sealed record Block(
        HairDeckSetDitCheckpointConverter.Linear Ada,
        HairDeckSetDitCheckpointConverter.Linear Qkv,
        HairDeckSetDitCheckpointConverter.Linear Output,
        HairDeckSetDitCheckpointConverter.Linear MlpIn,
        HairDeckSetDitCheckpointConverter.Linear MlpOut);

    /// <summary>The decoder: latent MLP, Fourier query projection, blocks, output norm and head, and the count head.</summary>
    public sealed record Weights(
        int Dim,
        int LatentDim,
        int Frequencies,
        int MaxPoints,
        bool Integrate,
        HairDeckSetDitCheckpointConverter.Linear FromLatentIn,
        HairDeckSetDitCheckpointConverter.Linear FromLatentOut,
        HairDeckSetDitCheckpointConverter.Linear QueryIn,
        IReadOnlyList<Block> Blocks,
        Tensor NormWeight,
        Tensor NormBias,
        HairDeckSetDitCheckpointConverter.Linear Output,
        HairDeckSetDitCheckpointConverter.Linear CountIn,
        HairDeckSetDitCheckpointConverter.Linear CountOut);

    /// <summary>Reads the decoder. <paramref name="integrate"/> is the checkpoint's <c>integ</c> flag (positions are the
    /// trapezoid integral of the xyz channels when set).</summary>
    public static Weights Convert(IReadOnlyDictionary<string, Tensor> all, bool integrate)
    {
        ArgumentNullException.ThrowIfNull(all);
        Tensor norm = HairDeckTensorReader.TakeRank(all, "norm.weight", 1);
        long dim = norm.Shape[0];
        if (dim % Heads != 0)
            throw new InvalidDataException($"HairDeck card VAE width {dim} is not a multiple of {Heads} heads.");
        Tensor fromZ = HairDeckTensorReader.TakeRank(all, "from_z.0.weight", 2);
        long zdim = fromZ.Shape[1];
        Tensor queryIn = HairDeckTensorReader.TakeRank(all, "q_in.weight", 2);
        long fourierWidth = queryIn.Shape[1];
        if (fourierWidth % 2 == 0)
            throw new InvalidDataException($"HairDeck card VAE query width {fourierWidth} is not 1 + 2 * frequencies.");
        Tensor countOut = HairDeckTensorReader.TakeRank(all, "n_head.2.weight", 2);
        long maxPoints = countOut.Shape[0];

        int depth = 0;
        while (HairDeckTensorReader.Exists(all, $"dec.blocks.{depth}.ada.weight")) depth++;
        if (depth == 0)
            throw new InvalidDataException("HairDeck card VAE checkpoint has no decoder blocks.");

        List<Block> blocks = new(depth);
        for (int i = 0; i < depth; i++)
        {
            string b = $"dec.blocks.{i}.";
            blocks.Add(new Block(
                Ada: HairDeckSetDitCheckpointConverter.Lin(all, b + "ada", 6 * dim, dim),
                Qkv: HairDeckSetDitCheckpointConverter.Lin(all, b + "qkv", 3 * dim, dim),
                Output: HairDeckSetDitCheckpointConverter.Lin(all, b + "o", dim, dim),
                MlpIn: HairDeckSetDitCheckpointConverter.Lin(all, b + "mlp.0", 4 * dim, dim),
                MlpOut: HairDeckSetDitCheckpointConverter.Lin(all, b + "mlp.2", dim, 4 * dim)));
        }

        return new Weights(
            Dim: (int)dim,
            LatentDim: (int)zdim,
            Frequencies: (int)((fourierWidth - 1) / 2),
            MaxPoints: (int)maxPoints,
            Integrate: integrate,
            FromLatentIn: HairDeckSetDitCheckpointConverter.Lin(all, "from_z.0", dim, zdim),
            FromLatentOut: HairDeckSetDitCheckpointConverter.Lin(all, "from_z.2", dim, dim),
            QueryIn: HairDeckSetDitCheckpointConverter.Lin(all, "q_in", dim, fourierWidth),
            Blocks: blocks,
            NormWeight: norm,
            NormBias: HairDeckTensorReader.Take(all, "norm.bias", dim),
            Output: HairDeckSetDitCheckpointConverter.Lin(all, "out", OutputChannels, dim),
            CountIn: HairDeckSetDitCheckpointConverter.Lin(all, "n_head.0", dim, zdim),
            CountOut: HairDeckSetDitCheckpointConverter.Lin(all, "n_head.2", maxPoints, dim));
    }
}
