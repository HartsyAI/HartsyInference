using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Onnx;

namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>Structure read from a Kim_Vocal_2-style TFC-TDF-UNet graph: widths per level and the frequency size at the top.</summary>
/// <param name="InChannels">Spectrogram channels in and out (real/imag of left and right: 4).</param>
/// <param name="Widths">Channel width of each encoder level, then the bottleneck width last.</param>
/// <param name="DimF">Frequency bins at the top level.</param>
/// <param name="TfcLayers">3x3 conv layers per TFC block.</param>
public readonly record struct KimVocal2Geometry(int InChannels, int[] Widths, int DimF, int TfcLayers)
{
    /// <summary>Encoder levels (the bottleneck is not counted).</summary>
    public int Levels => Widths.Length - 1;
}

/// <summary>Reads a Kim_Vocal_2 (UVR MDX-Net, TFC-TDF-UNet) <c>.onnx</c> into logically named F32 tensors by walking its node list, so
/// the anonymized conv and MatMul initializers (<c>onnx::…</c> / numeric names) bind to layers by position and the graph is checked
/// op by op: any node that is not the expected one fails the load instead of binding the wrong weight.
/// <para>Layout of the result (names are this class's, not the file's):
/// <c>first_conv.*</c>, <c>final_conv.*</c> (1x1); per block <c>{encoding_blocks.i|bottleneck_block|decoding_blocks.i}.tfc.{0..2}.weight/bias</c>
/// (3x3 convs, BatchNorm already folded by the exporter), <c>.tdf.0.weight</c> and <c>.tdf.2.weight</c> (the two MatMuls, transposed to
/// <c>[out,in]</c> for <c>Linear</c>), <c>.tdf.1.scale/shift</c> and <c>.tdf.3.scale/shift</c> (the inference BatchNorms as per-channel affine, since they
/// follow a MatMul shared across channels and cannot fold into it); <c>encoding_blocks.i.down.*</c> (2x2 stride-2 conv);
/// <c>decoding_blocks.i.up.*</c> (2x2 stride-2 ConvTranspose with its BatchNorm folded along weight axis 1).</para></summary>
public static class KimVocal2OnnxConverter
{
    private const float DefaultBnEpsilon = 1e-5f;
    private const int TfcLayerCount = 3;

    /// <summary>Reads the file at <paramref name="path"/>; the returned tensors are owned by the caller.</summary>
    public static Dictionary<string, Tensor> Convert(string path, out KimVocal2Geometry geometry)
    {
        using OnnxWeightLoader loader = new();
        loader.Load(path);
        return Convert(loader.Model ?? throw new InvalidOperationException("ONNX model did not parse."), loader.GetAllTensors(), out geometry);
    }

    /// <summary>Binds <paramref name="initializers"/> (name to tensor, F32) to layers by walking <paramref name="model"/>'s nodes.</summary>
    public static Dictionary<string, Tensor> Convert(OnnxModel model, IReadOnlyDictionary<string, Tensor> initializers,
        out KimVocal2Geometry geometry)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(initializers);
        Dictionary<string, Tensor> result = [];
        try
        {
            Walker w = new(model.Nodes, initializers, result);
            geometry = w.Run();
            return result;
        }
        catch
        {
            foreach (Tensor t in result.Values) t.Dispose();
            throw;
        }
    }

    private sealed class Walker(IReadOnlyList<OnnxNode> nodes, IReadOnlyDictionary<string, Tensor> init, Dictionary<string, Tensor> output)
    {
        private int _pos;
        private string _cur = string.Empty;

        public KimVocal2Geometry Run()
        {
            if (nodes.Count == 0) throw Fail("the graph has no nodes");
            _cur = nodes[0].Inputs[0];
            Conv("first_conv", 1, 1, 0, out int inCh, out int firstWidth);
            Next("Relu");
            OnnxNode transpose = Next("Transpose");
            if (!transpose.IntAttributes.TryGetValue("perm", out long[]? perm) || !perm.AsSpan().SequenceEqual([0L, 1, 3, 2]))
                throw Fail("expected the frequency/time transpose perm [0,1,3,2] after the first conv");

            List<int> widths = [];
            List<string> skips = [];
            int dimF = 0;
            int tfcLayers = 0;
            while (true)
            {
                string prefix = $"encoding_blocks.{widths.Count}";
                (int width, int freq) = TfcBlock(prefix, out string blockOut, out tfcLayers);
                if (_pos < nodes.Count && nodes[_pos].OpType == "Conv")
                {
                    if (widths.Count == 0) dimF = freq;
                    widths.Add(width);
                    skips.Add(blockOut);
                    Conv(prefix + ".down", 2, 2, 0, out _, out _);
                    Next("Relu");
                    continue;
                }
                // The first TFC block not followed by a downsample is the bottleneck. It was parsed under the encoder's prefix.
                Rename(prefix, "bottleneck_block");
                widths.Add(width);
                break;
            }
            if (widths[0] != firstWidth) throw Fail($"first conv width {firstWidth} does not match the first block width {widths[0]}");

            for (int i = 0; i < skips.Count; i++)
            {
                string prefix = $"decoding_blocks.{i}";
                ConvTransposeBn(prefix + ".up");
                OnnxNode mul = Next("Mul");
                string skip = skips[skips.Count - 1 - i];
                if (!(mul.Inputs[0] == _cur && mul.Inputs[1] == skip) && !(mul.Inputs[1] == _cur && mul.Inputs[0] == skip))
                    throw Fail($"decoder {i}: the multiplicative skip is not wired to encoder {skips.Count - 1 - i}'s output");
                _cur = mul.Outputs[0];
                TfcBlock(prefix, out _, out _);
            }

            OnnxNode last = Next("Transpose");
            if (!last.IntAttributes.TryGetValue("perm", out long[]? perm2) || !perm2.AsSpan().SequenceEqual([0L, 1, 3, 2]))
                throw Fail("expected the closing transpose perm [0,1,3,2]");
            Conv("final_conv", 1, 1, 0, out int finalIn, out int finalOut);
            if (_pos != nodes.Count) throw Fail($"{nodes.Count - _pos} unexpected node(s) after the final conv");
            if (finalIn != widths[0] || finalOut != inCh)
                throw Fail($"final conv maps {finalIn}→{finalOut}, expected {widths[0]}→{inCh}");
            return new KimVocal2Geometry(inCh, [.. widths], dimF, tfcLayers);
        }

        private (int Width, int Freq) TfcBlock(string prefix, out string blockOut, out int tfcLayers)
        {
            int width = 0;
            for (int j = 0; j < TfcLayerCount; j++)
            {
                Conv($"{prefix}.tfc.{j}", 3, 1, 1, out int cin, out int cout);
                if (cin != cout) throw Fail($"{prefix}.tfc.{j}: a TFC conv must keep its width ({cin}→{cout})");
                width = cout;
                Next("Relu");
            }
            tfcLayers = TfcLayerCount;
            string tfcOut = _cur;
            int freq = MatMul($"{prefix}.tdf.0", out int mid);
            Bn($"{prefix}.tdf.1", width);
            Next("Relu");
            int freqBack = MatMul($"{prefix}.tdf.2", out int midIn);
            if (freqBack != mid || midIn != freq)
                throw Fail($"{prefix}: the TDF MatMuls do not mirror each other ({freq}→{mid}, then {freqBack}→{midIn})");
            Bn($"{prefix}.tdf.3", width);
            Next("Relu");
            string tdfOut = _cur;
            OnnxNode add = Next("Add");
            if (!(add.Inputs[0] == tfcOut && add.Inputs[1] == tdfOut) && !(add.Inputs[1] == tfcOut && add.Inputs[0] == tdfOut))
                throw Fail($"{prefix}: the residual Add is not tfc output + tdf output");
            _cur = add.Outputs[0];
            blockOut = _cur;
            return (width, freq);
        }

        /// <summary>A Conv node whose input is the running tensor; stores <c>{name}.weight/bias</c> and returns the weight's channel counts.</summary>
        private void Conv(string name, int kernel, int stride, int pad, out int cin, out int cout)
        {
            OnnxNode node = Next("Conv");
            Chained(node, 0);
            ExpectInts(node, "kernel_shape", kernel, kernel);
            ExpectInts(node, "strides", stride, stride);
            ExpectInts(node, "pads", pad, pad, pad, pad);
            ExpectInts(node, "dilations", 1, 1);
            if (node.IntAttributes.TryGetValue("group", out long[]? g) && g[0] != 1) throw Fail($"{node.Name}: grouped convs are not part of this model");
            Tensor weight = Init(node, 1);
            Tensor bias = Init(node, 2);
            if (weight.Shape.Rank != 4 || weight.Shape[2] != kernel || weight.Shape[3] != kernel)
                throw Fail($"{node.Name}: weight {weight.Shape} is not a {kernel}x{kernel} conv kernel");
            cout = (int)weight.Shape[0];
            cin = (int)weight.Shape[1];
            if (bias.ElementCount != cout) throw Fail($"{node.Name}: bias has {bias.ElementCount} elements for {cout} output channels");
            Put(name + ".weight", Copy(weight));
            Put(name + ".bias", Copy(bias));
            _cur = node.Outputs[0];
        }

        private int MatMul(string name, out int outDim)
        {
            OnnxNode node = Next("MatMul");
            Chained(node, 0);
            Tensor weight = Init(node, 1);
            if (weight.Shape.Rank != 2) throw Fail($"{node.Name}: MatMul weight {weight.Shape} is not rank 2");
            int inDim = (int)weight.Shape[0];
            outDim = (int)weight.Shape[1];
            Put(name + ".weight", TransposeCopy(weight, inDim, outDim));
            _cur = node.Outputs[0];
            return inDim;
        }

        /// <summary>A BatchNormalization node stored as <c>{name}.scale/shift</c>.</summary>
        private void Bn(string name, int channels)
        {
            (Tensor scale, Tensor shift) = ReadBn(Next("BatchNormalization"), channels);
            Put(name + ".scale", scale);
            Put(name + ".shift", shift);
        }

        /// <summary>ConvTranspose (2x2, stride 2) followed by its BatchNorm, folded into one weight and bias.</summary>
        private void ConvTransposeBn(string name)
        {
            OnnxNode node = Next("ConvTranspose");
            Chained(node, 0);
            ExpectInts(node, "kernel_shape", 2, 2);
            ExpectInts(node, "strides", 2, 2);
            ExpectInts(node, "pads", 0, 0, 0, 0);
            Tensor weight = Init(node, 1);
            Tensor bias = Init(node, 2);
            if (weight.Shape.Rank != 4) throw Fail($"{node.Name}: ConvTranspose weight {weight.Shape} is not rank 4");
            int cin = (int)weight.Shape[0];
            int cout = (int)weight.Shape[1];
            long taps = weight.Shape[2] * weight.Shape[3];
            _cur = node.Outputs[0];
            OnnxNode bnNode = Next("BatchNormalization");
            Chained(bnNode, 0);
            (Tensor scale, Tensor shift) = ReadBn(bnNode, cout, advance: false);
            Tensor foldedW = new(weight.Shape, DType.F32);
            Tensor foldedB = new(new TensorShape(cout), DType.F32);
            unsafe
            {
                float* w = (float*)weight.DataPointer;
                float* b = (float*)bias.DataPointer;
                float* s = (float*)scale.DataPointer;
                float* t = (float*)shift.DataPointer;
                float* fw = (float*)foldedW.DataPointer;
                float* fb = (float*)foldedB.DataPointer;
                for (int ci = 0; ci < cin; ci++)
                {
                    for (int co = 0; co < cout; co++)
                    {
                        long o = ((long)ci * cout + co) * taps;
                        for (long k = 0; k < taps; k++) fw[o + k] = w[o + k] * s[co];
                    }
                }
                for (int co = 0; co < cout; co++) fb[co] = b[co] * s[co] + t[co];
            }
            scale.Dispose();
            shift.Dispose();
            Put(name + ".weight", foldedW);
            Put(name + ".bias", foldedB);
            _cur = bnNode.Outputs[0];
            Next("Relu");
        }

        private (Tensor Scale, Tensor Shift) ReadBn(OnnxNode node, int channels, bool advance = true)
        {
            if (advance) Chained(node, 0);
            if (node.Inputs.Length < 5) throw Fail($"{node.Name}: BatchNormalization needs scale, bias, mean and variance inputs");
            Tensor gamma = Init(node, 1);
            Tensor beta = Init(node, 2);
            Tensor mean = Init(node, 3);
            Tensor variance = Init(node, 4);
            foreach (Tensor p in new[] { gamma, beta, mean, variance })
            {
                if (p.ElementCount != channels) throw Fail($"{node.Name}: BatchNormalization parameter has {p.ElementCount} elements for {channels} channels");
            }
            float eps = node.FloatAttributes.TryGetValue("epsilon", out float e) ? e : DefaultBnEpsilon;
            Tensor scale = new(new TensorShape(channels), DType.F32);
            Tensor shift = new(new TensorShape(channels), DType.F32);
            unsafe
            {
                float* g = (float*)gamma.DataPointer;
                float* b = (float*)beta.DataPointer;
                float* m = (float*)mean.DataPointer;
                float* v = (float*)variance.DataPointer;
                float* ps = (float*)scale.DataPointer;
                float* pt = (float*)shift.DataPointer;
                for (int c = 0; c < channels; c++)
                {
                    float s = g[c] / MathF.Sqrt(v[c] + eps);
                    ps[c] = s;
                    pt[c] = b[c] - m[c] * s;
                }
            }
            if (advance) _cur = node.Outputs[0];
            return (scale, shift);
        }

        private void Rename(string fromPrefix, string toPrefix)
        {
            foreach (string key in output.Keys.Where(k => k.StartsWith(fromPrefix + ".", StringComparison.Ordinal)).ToArray())
            {
                output[toPrefix + key[fromPrefix.Length..]] = output[key];
                output.Remove(key);
            }
        }

        private OnnxNode Next(string opType)
        {
            if (_pos >= nodes.Count) throw Fail($"the graph ended where a {opType} was expected");
            OnnxNode node = nodes[_pos];
            if (node.OpType != opType) throw Fail($"node {_pos} is {node.OpType} '{node.Name}', expected {opType}");
            _pos++;
            if (opType == "Relu" || opType == "Transpose")
            {
                Chained(node, 0);
                _cur = node.Outputs[0];
            }
            return node;
        }

        private void Chained(OnnxNode node, int input)
        {
            if (node.Inputs[input] != _cur) throw Fail($"{node.OpType} '{node.Name}' does not consume the previous node's output '{_cur}'");
        }

        private Tensor Init(OnnxNode node, int input)
        {
            if (input >= node.Inputs.Length || !init.TryGetValue(node.Inputs[input], out Tensor? t))
                throw Fail($"{node.OpType} '{node.Name}' input {input} is not a float initializer");
            if (t.DType != DType.F32) throw Fail($"{node.OpType} '{node.Name}': initializer '{node.Inputs[input]}' is {t.DType}, expected F32");
            return t;
        }

        private void ExpectInts(OnnxNode node, string attribute, params long[] expected)
        {
            if (!node.IntAttributes.TryGetValue(attribute, out long[]? actual)) throw Fail($"{node.OpType} '{node.Name}' has no {attribute}");
            if (!actual.AsSpan().SequenceEqual(expected))
                throw Fail($"{node.OpType} '{node.Name}' {attribute} is [{string.Join(",", actual)}], expected [{string.Join(",", expected)}]");
        }

        private void Put(string name, Tensor tensor) => output.Add(name, tensor);

        private static InvalidDataException Fail(string message) => new($"Kim_Vocal_2 graph does not match the TFC-TDF-UNet layout: {message}.");

        private static unsafe Tensor Copy(Tensor source)
        {
            Tensor copy = new(source.Shape, DType.F32);
            long bytes = source.ElementCount * sizeof(float);
            Buffer.MemoryCopy(source.DataPointer, copy.DataPointer, bytes, bytes);
            return copy;
        }

        /// <summary>The ONNX MatMul weight is <c>[in,out]</c>; <c>Linear</c> wants <c>[out,in]</c>.</summary>
        private static unsafe Tensor TransposeCopy(Tensor source, int rows, int cols)
        {
            Tensor copy = new(new TensorShape(cols, rows), DType.F32);
            float* s = (float*)source.DataPointer;
            float* d = (float*)copy.DataPointer;
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++) d[(long)c * rows + r] = s[(long)r * cols + c];
            }
            return copy;
        }
    }
}
