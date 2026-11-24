namespace HartsyInference.ModelAssets.Tests.Onnx;

/// <summary>A small graph with Kim_Vocal_2's exact op sequence: widths 2, 3 then bottleneck 4, frequency 8 halving per level, 4 spectrogram channels.
/// Convs and MatMuls carry anonymous numeric names as the real export does; BatchNorms carry readable ones.</summary>
internal sealed class TinyKimGraph
{
    public const float BnEpsilon = 1e-3f;
    public static readonly int[] Widths = [2, 3, 4];
    public const int DimF = 8;
    public const int InChannels = 4;

    private readonly OnnxFixtureBuilder _b = new();
    private readonly Random _rng;
    private readonly bool _omitDefaults;
    private int _id;
    private string _cur = "input";

    /// <summary>Every initializer written, by file name, so a test can compare against what the converter produced.</summary>
    public Dictionary<string, float[]> Values { get; } = [];

    /// <summary>The file-level initializer names of the first TDF MatMul and its BatchNorm in encoder 0.</summary>
    public string FirstMatMulWeight { get; private set; } = "";
    public string FirstBnPrefix { get; } = "encoding_blocks.0.tdf.1";
    public string FirstUpPrefix { get; private set; } = "";
    public string FirstUpWeight { get; private set; } = "";

    public TinyKimGraph(int seed, bool swapSkip = false, bool omitDefaults = false)
    {
        _omitDefaults = omitDefaults;
        _rng = new Random(seed);
        Conv("first", InChannels, Widths[0], 1);
        Relu();
        Transpose();
        List<string> skips = [];
        for (int level = 0; level < Widths.Length - 1; level++)
        {
            Block($"encoding_blocks.{level}", Widths[level], DimF >> level);
            skips.Add(_cur);
            Conv("down", Widths[level], Widths[level + 1], 2, stride: 2);
            Relu();
        }
        Block("bottleneck_block", Widths[^1], DimF >> (Widths.Length - 1));
        for (int i = 0; i < skips.Count; i++)
        {
            int level = skips.Count - 1 - i;
            string up = $"us.{i}";
            ConvTranspose(up, Widths[level + 1], Widths[level]);
            Bn($"{up}.1", Widths[level]);
            Relu();
            string skip = swapSkip && i == 0 ? skips[0] : skips[level];
            string mul = Next();
            _b.AddNode("Mul", $"Mul_{_id}", [_cur, skip], [mul]);
            _cur = mul;
            Block($"decoding_blocks.{i}", Widths[level], DimF >> level);
        }
        Transpose();
        Conv("final", Widths[0], InChannels, 1, output: "output");
    }

    public byte[] Build() => _b.Build();

    private void Block(string prefix, int width, int freq)
    {
        for (int j = 0; j < 3; j++)
        {
            Conv("tfc", width, width, 3);
            Relu();
        }
        string tfcOut = _cur;
        int mid = Math.Max(1, freq / 2);
        MatMul(freq, mid, prefix);
        Bn($"{prefix}.tdf.1", width);
        Relu();
        MatMul(mid, freq, prefix, second: true);
        Bn($"{prefix}.tdf.4", width);
        Relu();
        string add = Next();
        _b.AddNode("Add", $"Add_{_id}", [tfcOut, _cur], [add]);
        _cur = add;
    }

    private void Conv(string tag, int cin, int cout, int kernel, int stride = 1, string? output = null)
    {
        int pad = kernel == 3 ? 1 : 0;
        string w = Anonymous(), bias = Anonymous(), outName = output ?? Next();
        Add(w, [cout, cin, kernel, kernel]);
        Add(bias, [cout]);
        Dictionary<string, long[]> attributes = new()
        {
            ["kernel_shape"] = [kernel, kernel],
            ["strides"] = [stride, stride],
            ["pads"] = [pad, pad, pad, pad],
            ["dilations"] = [1, 1],
            ["group"] = [1],
        };
        if (_omitDefaults)
        {
            attributes.Remove("dilations");
            attributes.Remove("group");
            if (stride == 1) attributes.Remove("strides");
            if (pad == 0) attributes.Remove("pads");
        }
        _b.AddNode("Conv", $"Conv_{_id}", [_cur, w, bias], [outName], attributes);
        _cur = outName;
    }

    private void ConvTranspose(string prefix, int cin, int cout)
    {
        string w = $"{prefix}.0.weight", bias = $"{prefix}.0.bias", outName = Next();
        Add(w, [cin, cout, 2, 2]);
        Add(bias, [cout]);
        if (FirstUpPrefix.Length == 0) { FirstUpPrefix = prefix; FirstUpWeight = w; }
        Dictionary<string, long[]> attributes = new()
        {
            ["kernel_shape"] = [2, 2],
            ["strides"] = [2, 2],
            ["pads"] = [0, 0, 0, 0],
            ["dilations"] = [1, 1],
            ["group"] = [1],
        };
        if (_omitDefaults)
        {
            attributes.Remove("dilations");
            attributes.Remove("group");
            attributes.Remove("pads");
        }
        _b.AddNode("ConvTranspose", $"ConvTranspose_{_id}", [_cur, w, bias], [outName], attributes);
        _cur = outName;
    }

    private void MatMul(int inDim, int outDim, string prefix, bool second = false)
    {
        string w = Anonymous(), outName = Next();
        Add(w, [inDim, outDim]);
        if (prefix == "encoding_blocks.0" && !second) FirstMatMulWeight = w;
        _b.AddNode("MatMul", $"MatMul_{_id}", [_cur, w], [outName]);
        _cur = outName;
    }

    private void Bn(string prefix, int channels)
    {
        string outName = Next();
        string[] names = [$"{prefix}.weight", $"{prefix}.bias", $"{prefix}.running_mean", $"{prefix}.running_var"];
        for (int i = 0; i < 4; i++)
        {
            Add(names[i], [channels], positive: i == 3 || i == 0);
        }
        _b.AddNode("BatchNormalization", $"BatchNormalization_{_id}", [_cur, .. names], [outName], null,
            new Dictionary<string, float> { ["epsilon"] = BnEpsilon });
        _cur = outName;
    }

    private void Relu()
    {
        string o = Next();
        _b.AddNode("Relu", $"Relu_{_id}", [_cur], [o]);
        _cur = o;
    }

    private void Transpose()
    {
        string o = Next();
        _b.AddNode("Transpose", $"Transpose_{_id}", [_cur], [o], new Dictionary<string, long[]> { ["perm"] = [0, 1, 3, 2] });
        _cur = o;
    }

    private string Next() => $"{_id++ + 1000}";

    private string Anonymous() => $"onnx::W_{_id++ + 5000}";

    private void Add(string name, long[] dims, bool positive = false)
    {
        long n = 1;
        foreach (long d in dims) n *= d;
        float[] data = new float[n];
        for (long i = 0; i < n; i++) data[i] = positive ? 0.5f + (float)_rng.NextDouble() : (float)(_rng.NextDouble() - 0.5);
        Values[name] = data;
        _b.AddInitializer(name, data, dims);
    }
}
