using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Diffusion.Models.Denoisers.SanaWm;

/// <summary>SANA-WM stage-1 DiT dimensions, read from the checkpoint header. Widths, depth, softmax layer indices and
/// branch flags come from tensor shapes and key presence; <see cref="RequiredHeadDim"/> is the one fixed width, because
/// the attention and RoPE kernels assume it.</summary>
public sealed record SanaWmConfig
{
    /// <summary>Per-head width the attention and RoPE kernels are built for; the derived width must equal it.</summary>
    public const int RequiredHeadDim = 112;

    /// <summary>Number of DiT blocks, counted contiguously from block 0.</summary>
    public required int Depth { get; init; }

    /// <summary>Hidden width C.</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Attention heads, taken from the per-head <c>A_log</c> length.</summary>
    public required int NumHeads { get; init; }

    /// <summary>Per-head width, <see cref="HiddenSize"/> divided by <see cref="NumHeads"/>.</summary>
    public required int HeadDim { get; init; }

    /// <summary>Latent channels into <c>x_embedder</c>.</summary>
    public required int InChannels { get; init; }

    /// <summary>Channels out of <c>final_layer.linear</c>.</summary>
    public required int OutChannels { get; init; }

    /// <summary>FFN hidden width: half of the inverted convolution's GLU output.</summary>
    public required int FfnHidden { get; init; }

    /// <summary>Text feature width into <c>y_embedder.y_proj.fc1</c>.</summary>
    public required int TextWidth { get; init; }

    /// <summary>Rows of <c>y_embedding</c>, the null caption length.</summary>
    public required int NullCaptionLength { get; init; }

    /// <summary>Sinusoidal timestep embedding width into <c>t_embedder</c>.</summary>
    public required int TimestepWidth { get; init; }

    /// <summary>Taps of the temporal depthwise convolution (<c>conv_k</c>) in GDN blocks.</summary>
    public required int ConvKernelSize { get; init; }

    /// <summary>Rows of <c>pos_embed</c>, the stored position table. Its use is decided by the DiT forward.</summary>
    public required int PosEmbedTokens { get; init; }

    /// <summary>Rows of each block's <c>scale_shift_table</c> (shift, scale and gate for attention and the FFN).</summary>
    public required int ModulationRows { get; init; }

    /// <summary>Indices of softmax-attention blocks, which carry no <c>conv_k</c> or <c>conv_k_cam</c>.</summary>
    public required IReadOnlyList<int> SoftmaxBlocks { get; init; }

    /// <summary>True when every block carries the camera (<c>*_cam</c>) branch.</summary>
    public required bool HasCameraBranch { get; init; }

    /// <summary>True when <c>plucker_embedder</c> and every block's <c>plucker_proj</c> are present.</summary>
    public required bool HasPluckerBranch { get; init; }

    /// <summary>Input channels of <c>plucker_embedder</c>, or 0 when absent.</summary>
    public required int PluckerChannels { get; init; }

    /// <summary>True when <c>raymap_embedder</c> is present.</summary>
    public required bool HasRaymapEmbedder { get; init; }

    /// <summary>Input channels of <c>raymap_embedder</c>, or 0 when absent.</summary>
    public required int RaymapChannels { get; init; }

    /// <summary>Longest caption, which is also the null caption length.</summary>
    public int CaptionMaxLength => NullCaptionLength;

    /// <summary>Whether block <paramref name="block"/> uses softmax attention rather than GDN.</summary>
    public bool IsSoftmaxBlock(int block) => SoftmaxBlocks.Contains(block);

    /// <summary>Derives the configuration from the header of a SANA-WM weight file. Only names, shapes and key presence are read.</summary>
    /// <exception cref="HartsyInferenceException">The header lacks a required tensor or its shapes are inconsistent.</exception>
    public static SanaWmConfig FromWeights(IReadOnlyDictionary<string, SafeTensorDescriptor> header)
    {
        ArgumentNullException.ThrowIfNull(header);
        Dictionary<string, TensorShape> shapes = new(header.Count, StringComparer.Ordinal);
        foreach (KeyValuePair<string, SafeTensorDescriptor> entry in header)
            shapes[entry.Key] = entry.Value.Shape;
        return Derive(shapes);
    }

    /// <summary>Derives the configuration from loaded tensors. Only their shapes are read.</summary>
    /// <exception cref="HartsyInferenceException">The tensors lack a required tensor or their shapes are inconsistent.</exception>
    public static SanaWmConfig FromWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        Dictionary<string, TensorShape> shapes = new(weights.Count, StringComparer.Ordinal);
        foreach (KeyValuePair<string, Tensor> entry in weights)
            shapes[entry.Key] = entry.Value.Shape;
        return Derive(shapes);
    }

    private static SanaWmConfig Derive(IReadOnlyDictionary<string, TensorShape> shapes)
    {
        int depth = CountBlocks(shapes);
        if (depth == 0)
            throw Invalid($"no {Block(0, "attn.qkv.weight")} tensor; the header has no DiT blocks");

        const string patchName = "x_embedder.proj.weight";
        TensorShape patch = Require(shapes, patchName, 5);
        int hidden = Dim(patchName, patch, 0);
        int inChannels = Dim(patchName, patch, 1);
        Expect(patchName, patch, hidden, inChannels, 1, 1, 1);

        string aLogName = Block(0, "attn.A_log");
        int heads = Dim(aLogName, Require(shapes, aLogName, 1), 0);
        if (hidden % heads != 0)
            throw Invalid($"{heads} heads do not divide the hidden width {hidden}");
        int headDim = hidden / heads;
        if (headDim != RequiredHeadDim)
            throw Invalid($"head width {headDim} is not {RequiredHeadDim}; the attention and RoPE kernels assume it");
        Expect(Block(0, "attn.qkv.weight"), Require(shapes, Block(0, "attn.qkv.weight"), 2), 3L * hidden, hidden);
        Expect(Block(0, "attn.beta_proj.weight"), Require(shapes, Block(0, "attn.beta_proj.weight"), 2), heads, hidden);

        List<int> softmax = [];
        int kernel = 0;
        for (int block = 0; block < depth; block++)
        {
            string convName = Block(block, "attn.conv_k.weight");
            if (!shapes.ContainsKey(convName))
            {
                softmax.Add(block);
                continue;
            }
            TensorShape conv = Require(shapes, convName, 3);
            int taps = Dim(convName, conv, 2);
            Expect(convName, conv, hidden, 1, taps);
            if (kernel == 0)
                kernel = taps;
            else if (taps != kernel)
                throw Invalid($"{convName} has {taps} taps, but an earlier GDN block has {kernel}");
        }
        if (kernel == 0)
            throw Invalid("no block has attn.conv_k.weight, so the GDN temporal convolution width is undefined");

        int camera = CountBlocksWith(shapes, depth, "attn.q_proj_cam.weight");
        if (camera != 0 && camera != depth)
            throw Invalid($"camera branch is present in {camera} of {depth} blocks; it must be in all or none");

        bool pluckerEmbedder = shapes.ContainsKey("plucker_embedder.proj.weight");
        int pluckerBlocks = CountBlocksWith(shapes, depth, "plucker_proj.weight");
        if (pluckerBlocks != 0 && pluckerBlocks != depth)
            throw Invalid($"plucker_proj is present in {pluckerBlocks} of {depth} blocks; it must be in all or none");
        if (pluckerEmbedder != (pluckerBlocks == depth))
            throw Invalid("plucker_embedder.proj.weight and the per-block plucker_proj disagree; "
                + "both or neither are required");
        int pluckerChannels = 0;
        if (pluckerEmbedder)
        {
            const string pluckerName = "plucker_embedder.proj.weight";
            TensorShape plucker = Require(shapes, pluckerName, 5);
            pluckerChannels = Dim(pluckerName, plucker, 1);
            Expect(pluckerName, plucker, hidden, pluckerChannels, 1, 1, 1);
        }

        bool raymap = shapes.ContainsKey("raymap_embedder.proj.weight");
        int raymapChannels = 0;
        if (raymap)
        {
            const string raymapName = "raymap_embedder.proj.weight";
            TensorShape raymapShape = Require(shapes, raymapName, 5);
            raymapChannels = Dim(raymapName, raymapShape, 1);
            Expect(raymapName, raymapShape, hidden, raymapChannels, 1, 1, 1);
        }

        string invName = Block(0, "mlp.inverted_conv.conv.weight");
        TensorShape inverted = Require(shapes, invName, 4);
        int glu = Dim(invName, inverted, 0);
        if (glu % 2 != 0)
            throw Invalid($"{invName} has {glu} rows; a GLU output must be even");
        int ffn = glu / 2;
        Expect(invName, inverted, glu, hidden, 1, 1);
        string pointName = Block(0, "mlp.point_conv.conv.weight");
        Expect(pointName, Require(shapes, pointName, 4), hidden, ffn, 1, 1);

        const string fc1Name = "y_embedder.y_proj.fc1.weight";
        TensorShape fc1 = Require(shapes, fc1Name, 2);
        int text = Dim(fc1Name, fc1, 1);
        Expect(fc1Name, fc1, hidden, text);
        const string embeddingName = "y_embedder.y_embedding";
        TensorShape embedding = Require(shapes, embeddingName, 2);
        int nullLength = Dim(embeddingName, embedding, 0);
        Expect(embeddingName, embedding, nullLength, text);

        const string timestepName = "t_embedder.mlp.0.weight";
        TensorShape timestep = Require(shapes, timestepName, 2);
        int timestepWidth = Dim(timestepName, timestep, 1);
        Expect(timestepName, timestep, hidden, timestepWidth);

        string tableName = Block(0, "scale_shift_table");
        TensorShape table = Require(shapes, tableName, 2);
        int modulationRows = Dim(tableName, table, 0);
        Expect(tableName, table, modulationRows, hidden);
        Expect("t_block.1.weight", Require(shapes, "t_block.1.weight", 2), (long)modulationRows * hidden, hidden);

        const string posName = "pos_embed";
        TensorShape pos = Require(shapes, posName, 3);
        int posTokens = Dim(posName, pos, 1);
        Expect(posName, pos, 1, posTokens, hidden);

        const string finalName = "final_layer.linear.weight";
        TensorShape finalLinear = Require(shapes, finalName, 2);
        int outChannels = Dim(finalName, finalLinear, 0);
        Expect(finalName, finalLinear, outChannels, hidden);
        if (outChannels != inChannels)
            throw Invalid($"final_layer.linear has {outChannels} output channels but x_embedder takes {inChannels}");
        Expect("final_layer.scale_shift_table", Require(shapes, "final_layer.scale_shift_table", 2), 2, hidden);

        return new SanaWmConfig
        {
            Depth = depth,
            HiddenSize = hidden,
            NumHeads = heads,
            HeadDim = headDim,
            InChannels = inChannels,
            OutChannels = outChannels,
            FfnHidden = ffn,
            TextWidth = text,
            NullCaptionLength = nullLength,
            TimestepWidth = timestepWidth,
            ConvKernelSize = kernel,
            PosEmbedTokens = posTokens,
            ModulationRows = modulationRows,
            SoftmaxBlocks = softmax.AsReadOnly(),
            HasCameraBranch = camera == depth,
            HasPluckerBranch = pluckerEmbedder,
            PluckerChannels = pluckerChannels,
            HasRaymapEmbedder = raymap,
            RaymapChannels = raymapChannels,
        };
    }

    private static string Block(int index, string suffix) => $"blocks.{index}.{suffix}";

    private static int CountBlocks(IReadOnlyDictionary<string, TensorShape> shapes)
    {
        int count = 0;
        while (shapes.ContainsKey(Block(count, "attn.qkv.weight")))
            count++;
        return count;
    }

    private static int CountBlocksWith(IReadOnlyDictionary<string, TensorShape> shapes, int depth, string suffix)
    {
        int count = 0;
        for (int block = 0; block < depth; block++)
        {
            if (shapes.ContainsKey(Block(block, suffix)))
                count++;
        }
        return count;
    }

    private static TensorShape Require(IReadOnlyDictionary<string, TensorShape> shapes, string name, int rank)
    {
        if (!shapes.TryGetValue(name, out TensorShape shape))
            throw Invalid($"missing {name}");
        if (shape.Rank != rank)
            throw Invalid($"{name} has rank {shape.Rank}, expected {rank}");
        return shape;
    }

    private static int Dim(string name, TensorShape shape, int index)
    {
        long value = shape[index];
        if (value <= 0 || value > int.MaxValue)
            throw Invalid($"{name} has dimension {index} = {value}");
        return (int)value;
    }

    private static void Expect(string name, TensorShape shape, params long[] expected)
    {
        bool match = shape.Rank == expected.Length;
        for (int i = 0; match && i < expected.Length; i++)
            match = shape[i] == expected[i];
        if (!match)
            throw Invalid($"{name} has shape {ShapeText(shape)}, expected {ShapeText(new TensorShape(expected))}");
    }

    internal static string ShapeText(TensorShape shape)
    {
        long[] dims = new long[shape.Rank];
        for (int i = 0; i < dims.Length; i++)
            dims[i] = shape[i];
        return "[" + string.Join(", ", dims) + "]";
    }

    private static HartsyInferenceException Invalid(string message) => new($"SANA-WM header: {message}");
}
