using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.UNetBlocks;

namespace HartsyInference.ThreeD.Models.Hunyuan3D.Paint;

/// <summary>The Stable Diffusion 2.x UNet of Hunyuan3D Paint with 2.5D transformer blocks. One instance is the generating stream (<c>unet</c>: twelve input
/// channels, multiview, reference and, for 2.1, material and DINO attention) and another, built with <c>generating: false</c>, is the reference stream
/// (<c>unet_dual</c>: four input channels, plain blocks that record their features). Every method takes one activation per material, each
/// <c>[views, channels, height, width]</c> F32.</summary>
public sealed class HunyuanPaintUnet : IDisposable
{
    private readonly HunyuanPaintConfig _config;
    private readonly bool _generating;
    private readonly int _levels;
    private readonly TimestepEmbedding _time;
    private readonly UNetResNetBlock[][] _downResnets, _upResnets;
    private readonly HunyuanPaintSpatialTransformer?[][] _downAttn, _upAttn;
    private readonly UNetResNetBlock[] _mid;
    private readonly HunyuanPaintSpatialTransformer _midAttn;
    private Tensor? _convInW, _convInB, _classEmbedding, _normOutW, _normOutB, _convOutW, _convOutB;
    private readonly Tensor?[] _downsampleW, _downsampleB, _upsampleW, _upsampleB;
    private HunyuanPaintF32Weights? _weights;

    /// <summary>Builds the UNet; <paramref name="generating"/> selects the stream.</summary>
    public HunyuanPaintUnet(HunyuanPaintConfig config, bool generating)
    {
        ArgumentNullException.ThrowIfNull(config);
        _config = config;
        _generating = generating;
        int[] ch = config.BlockOutChannels;
        _levels = ch.Length;
        int timeDim = config.TimeEmbeddingDim;
        _time = new TimestepEmbedding(ch[0], timeDim);
        _downResnets = new UNetResNetBlock[_levels][];
        _upResnets = new UNetResNetBlock[_levels][];
        _downAttn = new HunyuanPaintSpatialTransformer?[_levels][];
        _upAttn = new HunyuanPaintSpatialTransformer?[_levels][];
        _downsampleW = new Tensor?[_levels]; _downsampleB = new Tensor?[_levels];
        _upsampleW = new Tensor?[_levels]; _upsampleB = new Tensor?[_levels];
        int layers = config.LayersPerBlock;
        for (int i = 0; i < _levels; i++)
        {
            bool attention = i < _levels - 1;
            int inCh = i == 0 ? ch[0] : ch[i - 1];
            _downResnets[i] = new UNetResNetBlock[layers];
            _downAttn[i] = new HunyuanPaintSpatialTransformer?[layers];
            for (int j = 0; j < layers; j++)
            {
                _downResnets[i][j] = new UNetResNetBlock(j == 0 ? inCh : ch[i], ch[i], timeDim, config.NormGroups);
                if (attention)
                    _downAttn[i][j] = new HunyuanPaintSpatialTransformer(config, ch[i], config.Heads[i], generating, $"down_{i}_{j}_0");
            }
        }
        int mid = ch[^1];
        _mid = [new UNetResNetBlock(mid, mid, timeDim, config.NormGroups), new UNetResNetBlock(mid, mid, timeDim, config.NormGroups)];
        _midAttn = new HunyuanPaintSpatialTransformer(config, mid, config.Heads[^1], generating, "mid_0_0");
        for (int i = 0; i < _levels; i++)
        {
            int level = _levels - 1 - i;
            bool attention = i > 0;
            int outCh = ch[level], prevOut = i == 0 ? mid : ch[level + 1], inputCh = ch[Math.Max(level - 1, 0)];
            _upResnets[i] = new UNetResNetBlock[layers + 1];
            _upAttn[i] = new HunyuanPaintSpatialTransformer?[layers + 1];
            for (int j = 0; j <= layers; j++)
            {
                int skip = j == layers ? inputCh : outCh, main = j == 0 ? prevOut : outCh;
                _upResnets[i][j] = new UNetResNetBlock(main + skip, outCh, timeDim, config.NormGroups);
                if (attention)
                    _upAttn[i][j] = new HunyuanPaintSpatialTransformer(config, outCh, config.Heads[level], generating, $"up_{i}_{j}_0");
            }
        }
    }

    /// <summary>Reads the weights from <paramref name="source"/> (diffusers names under <paramref name="prefix"/>); tensors are converted to F32 once.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> source, string prefix = "")
    {
        string p = prefix.Length == 0 ? "" : prefix + ".";
        HunyuanPaintF32Weights w = _weights = new HunyuanPaintF32Weights(source);
        _convInW = w[$"{p}conv_in.weight"]; _convInB = w[$"{p}conv_in.bias"];
        int expectedIn = _generating ? _config.InChannels : _config.ReferenceInChannels;
        if (_convInW.Shape[1] != expectedIn)
            throw new InvalidDataException($"conv_in takes {_convInW.Shape[1]} channels, the configuration expects {expectedIn}.");
        _time.LoadWeights(w, $"{p}time_embedding");
        // Only the generating stream has the embedding: upstream attaches it after copying the UNet for the reference stream, which ignores class labels.
        if (_generating && _config.CameraClasses > 0)
            _classEmbedding = w[$"{p}class_embedding.weight"];
        for (int i = 0; i < _levels; i++)
        {
            for (int j = 0; j < _downResnets[i].Length; j++)
            {
                _downResnets[i][j].LoadWeights(w, $"{p}down_blocks.{i}.resnets.{j}");
                _downAttn[i][j]?.LoadWeights(w, $"{p}down_blocks.{i}.attentions.{j}");
            }
            if (i < _levels - 1)
            {
                _downsampleW[i] = w[$"{p}down_blocks.{i}.downsamplers.0.conv.weight"];
                _downsampleB[i] = w[$"{p}down_blocks.{i}.downsamplers.0.conv.bias"];
            }
        }
        _mid[0].LoadWeights(w, $"{p}mid_block.resnets.0");
        _midAttn.LoadWeights(w, $"{p}mid_block.attentions.0");
        _mid[1].LoadWeights(w, $"{p}mid_block.resnets.1");
        for (int i = 0; i < _levels; i++)
        {
            for (int j = 0; j < _upResnets[i].Length; j++)
            {
                _upResnets[i][j].LoadWeights(w, $"{p}up_blocks.{i}.resnets.{j}");
                _upAttn[i][j]?.LoadWeights(w, $"{p}up_blocks.{i}.attentions.{j}");
            }
            if (i < _levels - 1)
            {
                _upsampleW[i] = w[$"{p}up_blocks.{i}.upsamplers.0.conv.weight"];
                _upsampleB[i] = w[$"{p}up_blocks.{i}.upsamplers.0.conv.bias"];
            }
        }
        _normOutW = w[$"{p}conv_norm_out.weight"]; _normOutB = w[$"{p}conv_norm_out.bias"];
        _convOutW = w[$"{p}conv_out.weight"]; _convOutB = w[$"{p}conv_out.bias"];
    }

    /// <summary>Releases the F32 copies and regrouped tensors made by <see cref="LoadWeights"/> (and their GPU residency); the checkpoint is not touched.</summary>
    public void Dispose()
    {
        foreach (HunyuanPaintSpatialTransformer?[] level in _downAttn)
            foreach (HunyuanPaintSpatialTransformer? transformer in level)
                transformer?.Dispose();
        foreach (HunyuanPaintSpatialTransformer?[] level in _upAttn)
            foreach (HunyuanPaintSpatialTransformer? transformer in level)
                transformer?.Dispose();
        _midAttn.Dispose();
        _weights?.Dispose();
        _weights = null;
        _convInW = _convInB = _classEmbedding = _normOutW = _normOutB = _convOutW = _convOutB = null;
    }

    /// <summary>Runs one pass. <paramref name="samples"/> holds one <c>[views, inChannels, H, W]</c> tensor per material (borrowed);
    /// <paramref name="classLabels"/> gives each view's camera class when the model has a class embedding. Returns one <c>[views, 4, H, W]</c>
    /// prediction per material; the caller owns them.</summary>
    public Tensor[] Forward(IBackend backend, Tensor[] samples, float timestep, ReadOnlySpan<int> classLabels, HunyuanPaintBlockState state)
    {
        ArgumentNullException.ThrowIfNull(samples);
        if (_convInW is null)
            throw new InvalidOperationException("Load the weights first.");
        int materials = samples.Length, views = (int)samples[0].Shape[0], height = (int)samples[0].Shape[2], width = (int)samples[0].Shape[3];
        Tensor temb = TimeEmbedding(backend, timestep, views, classLabels);

        Tensor[] hidden = new Tensor[materials];
        List<Tensor[]> skips = [];
        for (int m = 0; m < materials; m++)
        {
            hidden[m] = new Tensor(new TensorShape(views, _config.BlockOutChannels[0], height, width), DType.F32);
            backend.Conv2D(hidden[m], samples[m], _convInW, _convInB, 1, 1, 1, 1);
        }
        skips.Add(Clone(backend, hidden));
        for (int i = 0; i < _levels; i++)
        {
            for (int j = 0; j < _downResnets[i].Length; j++)
            {
                hidden = Step(hidden, h => _downResnets[i][j].Forward(backend, h, temb));
                if (_downAttn[i][j] is { } attention)
                    hidden = Replace(hidden, attention.Forward(backend, hidden, state));
                skips.Add(Clone(backend, hidden));
            }
            if (i < _levels - 1)
            {
                hidden = Step(hidden, h =>
                {
                    int rows = (int)h.Shape[2], cols = (int)h.Shape[3];
                    Tensor down = new(new TensorShape(views, h.Shape[1], (rows + 1) / 2, (cols + 1) / 2), DType.F32);
                    backend.Conv2D(down, h, _downsampleW[i]!, _downsampleB[i], 2, 2, 1, 1);
                    return down;
                });
                skips.Add(Clone(backend, hidden));
            }
        }

        hidden = Step(hidden, h => _mid[0].Forward(backend, h, temb));
        hidden = Replace(hidden, _midAttn.Forward(backend, hidden, state));
        hidden = Step(hidden, h => _mid[1].Forward(backend, h, temb));

        for (int i = 0; i < _levels; i++)
        {
            for (int j = 0; j < _upResnets[i].Length; j++)
            {
                Tensor[] skip = skips[^1];
                skips.RemoveAt(skips.Count - 1);
                Tensor[] joined = new Tensor[materials];
                for (int m = 0; m < materials; m++)
                {
                    joined[m] = new Tensor(new TensorShape(views, hidden[m].Shape[1] + skip[m].Shape[1], hidden[m].Shape[2], hidden[m].Shape[3]), DType.F32);
                    backend.Concat(joined[m], [hidden[m], skip[m]], 1);
                    hidden[m].Dispose();
                    skip[m].Dispose();
                }
                hidden = joined;
                hidden = Step(hidden, h => _upResnets[i][j].Forward(backend, h, temb));
                if (_upAttn[i][j] is { } attention)
                    hidden = Replace(hidden, attention.Forward(backend, hidden, state));
            }
            if (i < _levels - 1)
            {
                long upH = skips[^1][0].Shape[2], upW = skips[^1][0].Shape[3];
                hidden = Step(hidden, h =>
                {
                    Tensor up = new(new TensorShape(views, h.Shape[1], upH, upW), DType.F32);
                    backend.UpsampleNearest2DToSize(up, h);
                    Tensor conv = new(up.Shape, DType.F32);
                    backend.Conv2D(conv, up, _upsampleW[i]!, _upsampleB[i], 1, 1, 1, 1);
                    up.Dispose();
                    return conv;
                });
            }
        }
        temb.Dispose();

        Tensor[] output = new Tensor[materials];
        for (int m = 0; m < materials; m++)
        {
            Tensor activated = new(hidden[m].Shape, DType.F32);
            backend.GroupNormSilu(activated, hidden[m], _normOutW!, _normOutB!, _config.NormGroups, 1e-5f);
            hidden[m].Dispose();
            output[m] = new Tensor(new TensorShape(views, _config.OutChannels, height, width), DType.F32);
            backend.Conv2D(output[m], activated, _convOutW!, _convOutB, 1, 1, 1, 1);
            activated.Dispose();
        }
        return output;
    }

    private Tensor TimeEmbedding(IBackend backend, float timestep, int views, ReadOnlySpan<int> classLabels)
    {
        Span<float> steps = stackalloc float[views];
        steps.Fill(timestep);
        Tensor temb = _time.Forward(backend, steps, views);
        if (_classEmbedding is null)
            return temb;
        if (classLabels.Length != views)
            throw new ArgumentException($"Expected {views} camera classes, got {classLabels.Length}.", nameof(classLabels));
        long dim = _classEmbedding.Shape[1];
        Tensor classes = new(new TensorShape(views, dim), DType.F32);
        unsafe
        {
            for (int v = 0; v < views; v++)
            {
                if ((uint)classLabels[v] >= (uint)_classEmbedding.Shape[0])
                    throw new ArgumentOutOfRangeException(nameof(classLabels), $"Camera class {classLabels[v]} is outside the embedding table.");
                new ReadOnlySpan<float>((float*)_classEmbedding.DataPointer + classLabels[v] * dim, (int)dim).CopyTo(new Span<float>((float*)classes.DataPointer + v * dim, (int)dim));
            }
        }
        Tensor sum = new(temb.Shape, DType.F32);
        backend.Add(sum, temb, classes);
        temb.Dispose();
        classes.Dispose();
        return sum;
    }

    private static Tensor[] Step(Tensor[] hidden, Func<Tensor, Tensor> op)
    {
        Tensor[] next = new Tensor[hidden.Length];
        for (int m = 0; m < hidden.Length; m++)
        {
            next[m] = op(hidden[m]);
            hidden[m].Dispose();
        }
        return next;
    }

    private static Tensor[] Replace(Tensor[] old, Tensor[] next)
    {
        foreach (Tensor t in old)
            t.Dispose();
        return next;
    }

    private static Tensor[] Clone(IBackend backend, Tensor[] source)
    {
        Tensor[] copy = new Tensor[source.Length];
        for (int m = 0; m < source.Length; m++)
        {
            copy[m] = new Tensor(source[m].Shape, DType.F32);
            backend.Scale(copy[m], source[m], 1f);
        }
        return copy;
    }
}
