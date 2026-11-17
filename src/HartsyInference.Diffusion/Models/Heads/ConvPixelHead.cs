using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;

namespace HartsyInference.Diffusion.Models.Heads;

/// <summary>The weights of one up block: a 1x2x2 transposed convolution and a 3x3x3 causal convolution. Shapes follow the
/// PyTorch layouts, <c>[Cin, Cmid, 2, 2]</c> for the transposed convolution and <c>[Cout, Cmid, 3, 3, 3]</c> for the other.</summary>
public sealed record ConvPixelUpBlockWeights(Tensor TransposeWeight, Tensor? TransposeBias, Tensor ConvWeight, Tensor? ConvBias);

/// <summary>The weights of a <see cref="ConvPixelHead"/>: a 3x1x1 bottleneck convolution, the up blocks in the order they run,
/// and a 3x1x1 output convolution. The caller keeps ownership of the tensors.</summary>
public sealed record ConvPixelHeadWeights(Tensor BottleneckWeight, Tensor? BottleneckBias,
    IReadOnlyList<ConvPixelUpBlockWeights> UpBlocks, Tensor OutputWeight, Tensor? OutputBias);

/// <summary>A convolutional head that renders a DiT's token grid to pixels, one frame at a time: the timestep embedding is added
/// to the tokens and a SiLU applied, a 3x1x1 causal convolution mixes the grid, each up block doubles the height and width
/// (a 1x2x2 transposed convolution, a 3x3x3 causal convolution, SiLU), and a 3x1x1 causal convolution writes the output
/// channels. Every temporal kernel has size 3 and pads the past side only, so frame t reads frames t-2..t.
/// <para>Built from the paper "Back to Pixels" (arXiv 2611.17336, App. A.2). Not compared with the authors' code: none is
/// released. Where the paper is silent this code takes the plain reading and says so in the notes below.</para>
/// <list type="bullet">
/// <item>Spatial padding of the 3x3x3 convolution is zero, and the temporal past is zero-filled.</item>
/// <item>There is no activation after the bottleneck and none after the output convolution; the text states a SiLU only
/// before the bottleneck and at the end of each up block.</item>
/// <item>Biases are optional, as the weights carry them; the paper does not say.</item>
/// <item>The timestep embedding added to the tokens is whatever the caller passes (the DiT's own).</item>
/// </list>
/// Runs in F32 on the existing <see cref="CausalConv3d"/> and the backend's <c>ConvTranspose2d</c>.</summary>
public sealed class ConvPixelHead : IDitOutputHead
{
    private readonly int _inputDim;
    private readonly CausalConv3d _bottleneck;
    private readonly List<(Tensor Weight, Tensor? Bias, CausalConv3d Conv, int InChannels, int MidChannels)> _blocks = [];
    private readonly CausalConv3d _output;

    /// <summary>Validates the weight shapes against <paramref name="inputDim"/> and builds the head.</summary>
    /// <param name="inputDim">The DiT width D.</param>
    public ConvPixelHead(int inputDim, ConvPixelHeadWeights weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        if (weights.UpBlocks.Count == 0) throw new ArgumentException("A pixel head needs at least one up block.", nameof(weights));
        _inputDim = inputDim;
        Expect(weights.BottleneckWeight, "bottleneck", inputDim, inputDim, 3, 1, 1);
        _bottleneck = new CausalConv3d(weights.BottleneckWeight, weights.BottleneckBias, padT: 1);
        int channels = inputDim;
        for (int i = 0; i < weights.UpBlocks.Count; i++)
        {
            ConvPixelUpBlockWeights block = weights.UpBlocks[i];
            if (block.TransposeWeight.Shape.Rank != 4 || block.TransposeWeight.Shape[0] != channels
                || block.TransposeWeight.Shape[2] != 2 || block.TransposeWeight.Shape[3] != 2)
                throw new ArgumentException($"Up block {i}: the transposed convolution weight must be [{channels}, C, 2, 2]; got {block.TransposeWeight.Shape}.");
            int mid = (int)block.TransposeWeight.Shape[1];
            if (block.ConvWeight.Shape.Rank != 5 || block.ConvWeight.Shape[1] != mid
                || block.ConvWeight.Shape[2] != 3 || block.ConvWeight.Shape[3] != 3 || block.ConvWeight.Shape[4] != 3)
                throw new ArgumentException($"Up block {i}: the convolution weight must be [C, {mid}, 3, 3, 3]; got {block.ConvWeight.Shape}.");
            _blocks.Add((block.TransposeWeight, block.TransposeBias,
                new CausalConv3d(block.ConvWeight, block.ConvBias, padT: 1, padH: 1, padW: 1), channels, mid));
            channels = (int)block.ConvWeight.Shape[0];
        }
        if (weights.OutputWeight.Shape.Rank != 5 || weights.OutputWeight.Shape[1] != channels
            || weights.OutputWeight.Shape[2] != 3 || weights.OutputWeight.Shape[3] != 1 || weights.OutputWeight.Shape[4] != 1)
            throw new ArgumentException($"The output convolution weight must be [C, {channels}, 3, 1, 1]; got {weights.OutputWeight.Shape}.");
        _output = new CausalConv3d(weights.OutputWeight, weights.OutputBias, padT: 1);
        OutputChannels = (int)weights.OutputWeight.Shape[0];
        UpBlockCount = weights.UpBlocks.Count;
    }

    /// <summary>Output channels (3 for an RGB-like space).</summary>
    public int OutputChannels { get; }

    /// <summary>How many times the head doubles the height and width: a patch of 2^N pixels needs N up blocks.</summary>
    public int UpBlockCount { get; }

    /// <inheritdoc/>
    public Tensor Forward(IBackend backend, Tensor hidden, Tensor timestepEmbedding, (int T, int H, int W) grid)
    {
        ArgumentNullException.ThrowIfNull(backend);
        (int t, int h, int w) = grid;
        int tokens = t * h * w;
        if (hidden.Shape.Rank != 2 || hidden.Shape[0] != tokens || hidden.Shape[1] != _inputDim)
            throw new ArgumentException($"hidden must be [{tokens}, {_inputDim}] for a {t}x{h}x{w} grid; got {hidden.Shape}.", nameof(hidden));
        if (timestepEmbedding.ElementCount != _inputDim)
            throw new ArgumentException($"The timestep embedding must hold {_inputDim} values (one timestep for the whole clip); got {timestepEmbedding.Shape}.", nameof(timestepEmbedding));

        // [S, D] -> [D, S] is [1, D, T, H, W]; the embedding adds per channel.
        Tensor channelFirst = new(new TensorShape([1L, _inputDim, tokens]), DType.F32);
        backend.Transpose2D(channelFirst, hidden, tokens, _inputDim);
        backend.BroadcastAdd(channelFirst, timestepEmbedding, _inputDim, tokens);
        Tensor activated = new(channelFirst.Shape, DType.F32);
        backend.Silu(activated, channelFirst);
        channelFirst.Dispose();

        Tensor current;
        using (Tensor grid5 = activated.Reshape(new TensorShape([1L, _inputDim, t, h, w])))
        {
            current = _bottleneck.Forward(backend, grid5);
        }
        activated.Dispose();

        int height = h, width = w;
        foreach ((Tensor weight, Tensor? bias, CausalConv3d conv, int inChannels, int midChannels) in _blocks)
        {
            Tensor upsampled = TransposeFrames(backend, current, weight, bias, inChannels, midChannels, t, height, width);
            current.Dispose();
            height *= 2;
            width *= 2;
            Tensor mixed = conv.Forward(backend, upsampled);
            upsampled.Dispose();
            Tensor activation = new(mixed.Shape, DType.F32);
            backend.Silu(activation, mixed);
            mixed.Dispose();
            current = activation;
        }
        Tensor result = _output.Forward(backend, current);
        current.Dispose();
        return result;
    }

    /// <inheritdoc/>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor tensor in _bottleneck.EnumerateWeights()) yield return tensor;
        foreach ((Tensor weight, Tensor? bias, CausalConv3d conv, _, _) in _blocks)
        {
            yield return weight;
            if (bias is not null) yield return bias;
            foreach (Tensor tensor in conv.EnumerateWeights()) yield return tensor;
        }
        foreach (Tensor tensor in _output.EnumerateWeights()) yield return tensor;
    }

    // [1, C, T, h, w] -> a 2x2 stride-2 transposed convolution on every frame -> [1, Cmid, T, 2h, 2w]. The frames ride the
    // batch dimension of the 2-D op, so the layout is permuted to [T, C, h, w] and back.
    private static Tensor TransposeFrames(IBackend backend, Tensor input, Tensor weight, Tensor? bias, int inChannels,
        int midChannels, int frames, int height, int width)
    {
        Tensor perFrame = new(new TensorShape([(long)frames, inChannels, height, width]), DType.F32);
        backend.Permute0213(perFrame, input, inChannels, frames, height * width);
        Tensor convolved = new(new TensorShape([(long)frames, midChannels, 2 * height, 2 * width]), DType.F32);
        backend.ConvTranspose2d(convolved, perFrame, weight, bias, 2, 2, 0, 0);
        perFrame.Dispose();
        Tensor channelFirst = new(new TensorShape([1L, midChannels, frames, 2 * height, 2 * width]), DType.F32);
        backend.Permute0213(channelFirst, convolved, frames, midChannels, 4 * height * width);
        convolved.Dispose();
        return channelFirst;
    }

    private static void Expect(Tensor weight, string name, int cOut, int cIn, int kt, int kh, int kw)
    {
        if (weight.Shape.Rank != 5 || weight.Shape[0] != cOut || weight.Shape[1] != cIn || weight.Shape[2] != kt
            || weight.Shape[3] != kh || weight.Shape[4] != kw)
            throw new ArgumentException($"The {name} convolution weight must be [{cOut}, {cIn}, {kt}, {kh}, {kw}]; got {weight.Shape}.");
    }
}
