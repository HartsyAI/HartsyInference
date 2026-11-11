using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Gating;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Hunyuan3D.Paint;

/// <summary>One <c>Basic2p5DTransformerBlock</c>: material-wise self attention, reference attention, multiview attention, text attention, DINO attention
/// and a GEGLU feed-forward. The activations of every material travel as separate tensors <c>[views, tokens, channels]</c> (upstream stacks them as
/// <c>(batch, material, view)</c>), so the per-material weights need no gather and the multiview attention sees one material's views as one sequence.</summary>
internal sealed unsafe class HunyuanPaintBlock
{
    private const float NormEps = 1e-5f;
    private readonly HunyuanPaintConfig _config;
    private readonly int _channels, _heads, _headDim;
    private readonly bool _extended;
    private Tensor? _n1W, _n1B, _n2W, _n2B, _n3W, _n3B, _ffInW, _ffInB, _ffOutW, _ffOutB;
    private HunyuanPaintAttention[] _self = [];
    private HunyuanPaintAttention? _cross, _multiview, _dino;
    private HunyuanPaintAttention[] _reference = [];
    private readonly List<Tensor> _owned = [];

    internal HunyuanPaintBlock(HunyuanPaintConfig config, int channels, int heads, bool extended, string name)
    {
        _config = config;
        _channels = channels;
        _heads = heads;
        _headDim = channels / heads;
        _extended = extended;
        Name = name;
    }

    /// <summary>Name under which the reference stream stores this block's features.</summary>
    internal string Name { get; }

    internal void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        string t = $"{prefix}.transformer";
        _n1W = Own(w[$"{t}.norm1.weight"]); _n1B = Own(w[$"{t}.norm1.bias"]);
        _n2W = Own(w[$"{t}.norm2.weight"]); _n2B = Own(w[$"{t}.norm2.bias"]);
        _n3W = Own(w[$"{t}.norm3.weight"]); _n3B = Own(w[$"{t}.norm3.bias"]);
        _ffInW = Own(w[$"{t}.ff.net.0.proj.weight"]); _ffInB = Own(w[$"{t}.ff.net.0.proj.bias"]);
        _ffOutW = Own(w[$"{t}.ff.net.2.weight"]); _ffOutB = Own(w[$"{t}.ff.net.2.bias"]);
        _cross = Attention(w, $"{t}.attn2", $"{t}.attn2");

        int materials = _extended && _config.MaterialCount > 1 ? _config.MaterialCount : 1;
        _self = new HunyuanPaintAttention[materials];
        for (int m = 0; m < materials; m++)
            _self[m] = m == 0 ? Attention(w, $"{t}.attn1", $"{t}.attn1")
                : Attention(w, $"{t}.attn1.processor", $"{t}.attn1.processor", $"_{_config.Materials[m]}");
        if (!_extended)
            return;
        _multiview = Attention(w, $"{prefix}.attn_multiview", $"{prefix}.attn_multiview");
        _reference = ReferenceAttentions(w, $"{prefix}.attn_refview");
        if (_config.UseDino)
            _dino = Attention(w, $"{prefix}.attn_dino", $"{prefix}.attn_dino");
    }

    internal IEnumerable<Tensor> EnumerateWeights() => _owned;

    /// <summary>Runs the block over every material's <paramref name="x"/> (consumed); returns the new activations.</summary>
    internal Tensor[] Forward(IBackend backend, Tensor[] x, HunyuanPaintBlockState state)
    {
        int views = (int)x[0].Shape[0], tokens = (int)x[0].Shape[1];
        TensorShape shape = new(views, tokens, _channels), joint = new(1, views * tokens, _channels);
        Tensor[] h1 = new Tensor[x.Length];
        for (int m = 0; m < x.Length; m++)
        {
            h1[m] = new Tensor(shape, DType.F32);
            backend.LayerNorm(h1[m], x[m], _n1W!, _n1B!, NormEps);
            x[m] = AddInto(backend, x[m], _self[Math.Min(m, _self.Length - 1)].Run(backend, h1[m], h1[m]));
        }

        if (state.Write)
        {
            Tensor copy = new(joint, DType.F32);
            backend.Scale(copy, h1[0], 1f);
            state.Embeds[Name] = copy;
        }
        else if (_extended)
        {
            if (_reference.Length > 0 && state.ReferenceScale != 0f && state.Embeds.TryGetValue(Name, out Tensor? condition))
            {
                Tensor query = h1[0].Reshape(joint);
                for (int m = 0; m < x.Length; m++)
                    x[m] = AddInto(backend, x[m], _reference[m].Run(backend, query, condition), state.ReferenceScale);
            }
            if (views > 1 && _multiview is not null)
            {
                (Tensor Cos, Tensor Sin)? rotation = state.Rotation is not null && state.Rotation.TryGetValue(views * tokens, out (Tensor Cos, Tensor Sin) table) ? table : null;
                for (int m = 0; m < x.Length; m++)
                {
                    Tensor joined = h1[m].Reshape(joint);
                    x[m] = AddInto(backend, x[m], _multiview.Run(backend, joined, joined, rotation), state.MultiviewScale);
                }
            }
        }
        foreach (Tensor h in h1)
            h.Dispose();

        for (int m = 0; m < x.Length; m++)
        {
            Tensor n2 = new(shape, DType.F32);
            backend.LayerNorm(n2, x[m], _n2W!, _n2B!, NormEps);
            using Tensor text = Repeat(backend, state.Text[Math.Min(m, state.Text.Length - 1)], views);
            x[m] = AddInto(backend, x[m], _cross!.Run(backend, n2, text));
            if (_dino is not null && state.Dino is not null && !state.Write)
            {
                using Tensor dino = Repeat(backend, state.Dino, views);
                x[m] = AddInto(backend, x[m], _dino.Run(backend, n2, dino));
            }
            n2.Dispose();

            Tensor n3 = new(shape, DType.F32);
            backend.LayerNorm(n3, x[m], _n3W!, _n3B!, NormEps);
            x[m] = AddInto(backend, x[m], FeedForward(backend, n3, views, tokens));
            n3.Dispose();
        }
        return x;
    }

    private Tensor FeedForward(IBackend backend, Tensor input, int views, int tokens)
    {
        int inner = (int)_ffOutW!.Shape[1];
        Tensor projected = new(new TensorShape(views, tokens, inner * 2), DType.F32);
        backend.Linear(projected, input, _ffInW!, _ffInB);
        Tensor gated = new(new TensorShape(views, tokens, inner), DType.F32);
        backend.Glu(gated, projected, GluSpec.GeGlu);
        projected.Dispose();
        Tensor output = new(new TensorShape(views, tokens, _channels), DType.F32);
        backend.Linear(output, gated, _ffOutW!, _ffOutB);
        gated.Dispose();
        return output;
    }

    // x + scale * delta; takes both inputs.
    private static Tensor AddInto(IBackend backend, Tensor x, Tensor delta, float scale = 1f)
    {
        Tensor scaled = delta;
        if (scale != 1f)
        {
            scaled = new Tensor(delta.Shape, DType.F32);
            backend.Scale(scaled, delta, scale);
            delta.Dispose();
        }
        Tensor sum = new(x.Shape, DType.F32);
        backend.Add(sum, x, scaled.Shape.Equals(x.Shape) ? scaled : scaled.Reshape(x.Shape));
        x.Dispose();
        scaled.Dispose();
        return sum;
    }

    // [1, S, D] repeated to [views, S, D].
    private static Tensor Repeat(IBackend backend, Tensor tokens, int views)
    {
        Tensor output = new(new TensorShape(views, tokens.Shape[1], tokens.Shape[2]), DType.F32);
        Tensor[] copies = new Tensor[views];
        Array.Fill(copies, tokens);
        backend.Concat(output, copies, 0);
        return output;
    }

    private HunyuanPaintAttention Attention(IReadOnlyDictionary<string, Tensor> w, string qkv, string output, string suffix = "")
    {
        Tensor q = Own(w[$"{qkv}.to_q{suffix}.weight"]), k = Own(w[$"{qkv}.to_k{suffix}.weight"]), v = Own(w[$"{qkv}.to_v{suffix}.weight"]);
        Tensor o = Own(w[$"{output}.to_out{suffix}.0.weight"]), b = Own(w[$"{output}.to_out{suffix}.0.bias"]);
        return new HunyuanPaintAttention(_heads, _headDim, q, k, v, o, b);
    }

    // Reference attention shares Q and K and gives every material its own value and output weights. Upstream concatenates the value
    // projections of all materials and views the result as (heads, materials * headDim), so head h reads channels [h * M * d, (h + 1) * M * d) of
    // the concatenation and the output is split into the same groups. Selecting those rows once gives each material an ordinary attention.
    private HunyuanPaintAttention[] ReferenceAttentions(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        int materials = _config.MaterialCount;
        Tensor q = Own(w[$"{prefix}.to_q.weight"]), k = Own(w[$"{prefix}.to_k.weight"]);
        Tensor[] values = new Tensor[materials];
        for (int m = 0; m < materials; m++)
            values[m] = Own(m == 0 ? w[$"{prefix}.to_v.weight"] : w[$"{prefix}.processor.to_v_{_config.Materials[m]}.weight"]);
        HunyuanPaintAttention[] result = new HunyuanPaintAttention[materials];
        for (int m = 0; m < materials; m++)
        {
            Tensor regrouped = RegroupValues(values, m);
            _owned.Add(regrouped);
            Tensor o = Own(m == 0 ? w[$"{prefix}.to_out.0.weight"] : w[$"{prefix}.processor.to_out_{_config.Materials[m]}.0.weight"]);
            Tensor b = Own(m == 0 ? w[$"{prefix}.to_out.0.bias"] : w[$"{prefix}.processor.to_out_{_config.Materials[m]}.0.bias"]);
            result[m] = new HunyuanPaintAttention(_heads, _headDim, q, k, regrouped, o, b);
        }
        return result;
    }

    private Tensor RegroupValues(Tensor[] values, int material)
    {
        int materials = values.Length, rows = (int)values[0].Shape[0], cols = (int)values[0].Shape[1];
        long rowBytes = cols * sizeof(float);
        Tensor regrouped = new(new TensorShape(rows, cols), DType.F32);
        for (int head = 0; head < _heads; head++)
            for (int j = 0; j < _headDim; j++)
            {
                int fused = head * materials * _headDim + material * _headDim + j;
                Buffer.MemoryCopy((byte*)values[fused / rows].DataPointer + (long)(fused % rows) * rowBytes,
                    (byte*)regrouped.DataPointer + ((long)head * _headDim + j) * rowBytes, rowBytes, rowBytes);
            }
        return regrouped;
    }

    private Tensor Own(Tensor source)
    {
        Tensor f32 = Hunyuan3DShapeVae.F32(source);
        if (!_owned.Contains(f32))
            _owned.Add(f32);
        return f32;
    }
}
