using System.Security.Cryptography;
using System.Text.Json;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Tensors;

namespace HartsyInference.GpuBenchmarks.Attention;

/// <summary>Loads verified raw storage once; both benchmark engines consume the same bytes and views.</summary>
internal sealed unsafe class AttentionFixture : IDisposable
{
    private readonly List<Tensor> _owned = [];
    private AttentionCall _call;

    private AttentionFixture() { }

    /// <summary>Stable fixture cell identifier.</summary>
    public string CaseId { get; private set; } = "";
    /// <summary>Exact fixture manifest hash.</summary>
    public string ManifestHash { get; private set; } = "";
    /// <summary>Predeclared relative-L2 acceptance.</summary>
    public float RelativeTolerance { get; private set; }
    /// <summary>Whether the acceptance definition passed review.</summary>
    public bool AcceptanceReviewed { get; private set; }
    /// <summary>Additional absolute-error gate for quantized permissions.</summary>
    public float? MeanAbsoluteTolerance { get; private set; }
    /// <summary>Borrowed operands owned by this fixture.</summary>
    public AttentionCall Call => _call;

    /// <summary>Loads and validates storage before any operation or timed region.</summary>
    public static AttentionFixture Load(string path)
    {
        if (!BitConverter.IsLittleEndian) throw new NotSupportedException("Fixtures use little-endian storage.");
        AttentionFixture fixture = new();
        try
        {
            byte[] manifest = File.ReadAllBytes(path);
            fixture.ManifestHash = Convert.ToHexString(SHA256.HashData(manifest)).ToLowerInvariant();
            using JsonDocument json = JsonDocument.Parse(manifest);
            JsonElement root = json.RootElement;
            if (root.GetProperty("schema").GetInt32() != 1) throw new InvalidDataException("Unsupported fixture schema.");
            fixture.CaseId = root.GetProperty("case_id").GetString() ?? throw new InvalidDataException("Missing case id.");
            JsonElement tensors = root.GetProperty("tensors");
            JsonElement spec = root.GetProperty("call");
            ValidateSemantics(spec, tensors);
            string directory = Path.GetDirectoryName(Path.GetFullPath(path))!;
            AttentionOperand q = fixture.LoadOperand(directory, tensors.GetProperty("q"));
            AttentionOperand k = fixture.LoadOperand(directory, tensors.GetProperty("k"));
            AttentionOperand v = fixture.LoadOperand(directory, tensors.GetProperty("v"));
            Tensor? bias = tensors.TryGetProperty("bias", out JsonElement mask) ? fixture.LoadOperand(directory, mask).Storage : null;
            int validLength = ValidateLengths(spec, q.Batch, k.Tokens);
            Tensor output = new(new TensorShape(q.Batch, q.Heads, q.Tokens, v.HeadDim), q.DType);
            fixture._owned.Add(output);
            fixture._call = new AttentionCall
            {
                Query = q, Key = k, Value = v, Output = AttentionOperand.HeadMajor(output),
                Scale = spec.GetProperty("scale").GetSingle(), Mask = new AttentionMask
                {
                    Bias = bias, Causal = spec.GetProperty("causal").GetBoolean(),
                },
                Positions = AttentionPositions.Host(validLength,
                    spec.GetProperty("query_offset").GetInt32()),
                Route = Enum.Parse<AttentionRoute>(spec.GetProperty("route").GetString()!),
                Numerics = NumericalPolicy(spec, q, k, v),
            };
            JsonElement acceptance = root.GetProperty("acceptance");
            fixture.RelativeTolerance = acceptance.GetProperty("rel_l2").GetSingle();
            fixture.AcceptanceReviewed = acceptance.GetProperty("reviewed").GetBoolean();
            if (acceptance.TryGetProperty("mean_abs", out JsonElement mean) && mean.ValueKind == JsonValueKind.Number)
                fixture.MeanAbsoluteTolerance = mean.GetSingle();
            if (!float.IsFinite(fixture.RelativeTolerance) || fixture.RelativeTolerance <= 0)
                throw new InvalidDataException("Acceptance must be frozen before measurement.");
            return fixture;
        }
        catch
        {
            fixture.Dispose();
            throw;
        }
    }

    private static void ValidateSemantics(JsonElement spec, JsonElement tensors)
    {
        if (spec.ValueKind != JsonValueKind.Object || tensors.ValueKind != JsonValueKind.Object)
            throw new InvalidDataException("Fixture call and tensors must be objects.");
        if (spec.TryGetProperty("api", out JsonElement api))
        {
            if (api.ValueKind != JsonValueKind.String)
                throw new InvalidDataException("Fixture API must be a string.");
            if (api.GetString() != "dense")
                throw new NotSupportedException("The C# fixture adapter and oracle only support ordinary dense attention API semantics.");
        }
        foreach (JsonProperty field in spec.EnumerateObject())
        {
            if (field.Name is not ("api" or "route" or "scale" or "causal" or "kv_length" or "kv_lengths"
                or "query_offset" or "legacy_asserted_bounded" or "precision" or "reuse_count"))
                throw new NotSupportedException("Unsupported fixture call semantics: " + field.Name);
        }
        if (spec.TryGetProperty("reuse_count", out JsonElement reuse)
            && (reuse.ValueKind != JsonValueKind.Number || !reuse.TryGetInt32(out int count) || count != 1))
            throw new NotSupportedException("The C# fixture adapter does not implement prepared-state reuse semantics.");
        foreach (JsonProperty tensor in tensors.EnumerateObject())
        {
            if (tensor.Name is not ("q" or "k" or "v" or "bias"))
                throw new NotSupportedException("Unsupported fixture operand semantics: " + tensor.Name);
        }
    }

    private static int ValidateLengths(JsonElement spec, int batch, int capacity)
    {
        int validLength = spec.GetProperty("kv_length").GetInt32();
        if (validLength < 0 || validLength > capacity) throw new InvalidDataException("KV length exceeds cache capacity.");
        if (!spec.TryGetProperty("kv_lengths", out JsonElement lengths)) return validLength;
        if (lengths.ValueKind != JsonValueKind.Array || lengths.GetArrayLength() != batch)
            throw new InvalidDataException("Per-sequence KV lengths must contain one entry for each batch element.");
        foreach (JsonElement item in lengths.EnumerateArray())
        {
            if (item.ValueKind != JsonValueKind.Number || !item.TryGetInt32(out int length) || length < 0 || length > capacity)
                throw new InvalidDataException("Per-sequence KV length is invalid or exceeds cache capacity.");
            if (length != validLength)
                throw new NotSupportedException("The C# fixture adapter cannot honor per-sequence KV lengths that differ from kv_length.");
        }
        return validLength;
    }

    /// <summary>Creates the independent sampled oracle or rejects unsupported semantics.</summary>
    public AttentionOracle Oracle()
    {
        AttentionCall c = _call;
        if (c.QkHeadDim != c.ValueHeadDim || c.ValidKvLength != c.KeyTokens || c.Mask.Bias is not null)
            throw new NotSupportedException("The baseline sampled oracle does not cover this fixture; it cannot be certified.");
        AttentionBenchShape shape = new(CaseId, c.Batch, c.Heads, c.KvHeads, c.QueryTokens, c.KeyTokens,
            c.QkHeadDim, "hm", "none", c.Route == AttentionRoute.Flash ? "flash" : "sdpa", c.Mask.Causal, "shared fixture");
        if (c.Mask.Causal && c.Positions.QueryOffset != shape.QOffset)
            throw new NotSupportedException("The sampled oracle needs bottom-right alignment.");
        return new AttentionOracle(shape, Canonical(c.Query), Canonical(c.Key), Canonical(c.Value), null, c.Scale);
    }

    /// <summary>Reads completed output, outside timing.</summary>
    public double OutputAt(int batch, int head, int row, int channel) => Read(_call.Output, batch, head, row, channel);

    /// <summary>Hashes the complete output storage after validation, outside timing.</summary>
    public string OutputHash() => Convert.ToHexString(SHA256.HashData(new ReadOnlySpan<byte>(
        (void*)_call.Output.Storage.DataPointer,
        checked((int)_call.Output.DType.ComputeByteCount(_call.Output.Storage.ElementCount))))).ToLowerInvariant();

    private static AttentionNumerics NumericalPolicy(JsonElement spec, AttentionOperand q, AttentionOperand k, AttentionOperand v)
    {
        string precision = spec.TryGetProperty("precision", out JsonElement value) ? value.GetString()! : "f32";
        AttentionFormats granted = precision switch
        {
            "f32" => AttentionFormats.None, "tf32" => AttentionFormats.Tf32, "f16" => AttentionFormats.F16,
            "bf16" => AttentionFormats.Bf16,
            "int8" => AttentionFormats.F16 | AttentionFormats.Bf16 | AttentionFormats.Int8,
            _ => throw new InvalidDataException("Unknown fixture numerical policy."),
        };
        return new AttentionNumerics
        {
            QueryKey = granted, ProbValue = granted,
            QueryKeyBound = Math.Max(Bound(q), Bound(k)), ValueBound = Bound(v),
            LegacyAssertedBounded = spec.GetProperty("legacy_asserted_bounded").GetBoolean(),
        };
    }

    private static float Bound(AttentionOperand operand)
    {
        double bound = 0;
        for (int b = 0; b < operand.Batch; b++)
            for (int h = 0; h < operand.Heads; h++)
                for (int r = 0; r < operand.Tokens; r++)
                    for (int c = 0; c < operand.HeadDim; c++)
                    {
                        double value = Read(operand, b, h, r, c);
                        if (!double.IsFinite(value)) throw new InvalidDataException("Nonfinite operand needs a dedicated case.");
                        bound = Math.Max(bound, Math.Abs(value));
                    }
        return (float)bound;
    }

    private AttentionOperand LoadOperand(string directory, JsonElement descriptor)
    {
        string file = descriptor.GetProperty("file").GetString() ?? "";
        if (string.IsNullOrWhiteSpace(file) || file.Contains('/') || file.Contains('\\') || Path.IsPathRooted(file))
            throw new InvalidDataException("Tensor files must be direct fixture children.");
        string path = Path.Combine(directory, file);
        if ((File.GetAttributes(path) & FileAttributes.ReparsePoint) != 0)
            throw new InvalidDataException("Fixture links are not supported.");
        byte[] storage = File.ReadAllBytes(path);
        string hash = Convert.ToHexString(SHA256.HashData(storage)).ToLowerInvariant();
        if (hash != descriptor.GetProperty("sha256").GetString()) throw new InvalidDataException("Tensor hash mismatch: " + file);
        DType dtype = descriptor.GetProperty("dtype").GetString() switch
        {
            "f32" => DType.F32, "f16" => DType.F16, "bf16" => DType.BF16,
            _ => throw new InvalidDataException("Unsupported fixture dtype."),
        };
        long[] physical = descriptor.GetProperty("storage_shape").EnumerateArray().Select(n => n.GetInt64()).ToArray();
        long[] logical = descriptor.GetProperty("shape").EnumerateArray().Select(n => n.GetInt64()).ToArray();
        long[] strides = descriptor.GetProperty("strides").EnumerateArray().Select(n => n.GetInt64()).ToArray();
        if (physical.Length is < 1 or > 6 || logical.Length != 4 || strides.Length != 4
            || physical.Any(n => n <= 0) || logical.Any(n => n <= 0 || n > int.MaxValue) || strides.Any(n => n < 0))
            throw new InvalidDataException("Invalid fixture geometry.");
        long count = 1;
        foreach (long dim in physical) count = checked(count * dim);
        if (checked(count * dtype.SizeInBytes) != storage.Length) throw new InvalidDataException("Wrong storage length.");
        long offset = descriptor.GetProperty("offset").GetInt64();
        long extent = offset;
        for (int i = 0; i < 4; i++) extent = checked(extent + (logical[i] - 1) * strides[i]);
        if (offset < 0 || extent >= count || strides[3] != 1) throw new InvalidDataException("Fixture view exceeds storage.");
        Tensor tensor = new(new TensorShape(physical), dtype);
        _owned.Add(tensor);
        storage.AsSpan().CopyTo(new Span<byte>((void*)tensor.DataPointer, storage.Length));
        AttentionLayout layout = descriptor.TryGetProperty("layout", out JsonElement label)
            ? Enum.Parse<AttentionLayout>(label.GetString()!) : AttentionLayout.HeadMajor;
        return new(tensor, layout, (int)logical[0], (int)logical[1], (int)logical[2], (int)logical[3],
            offset, strides[0], strides[1], strides[2]);
    }

    private static float[] Canonical(AttentionOperand operand)
    {
        float[] values = new float[checked(operand.Batch * operand.Heads * operand.Tokens * operand.HeadDim)];
        int i = 0;
        for (int b = 0; b < operand.Batch; b++)
            for (int h = 0; h < operand.Heads; h++)
                for (int r = 0; r < operand.Tokens; r++)
                    for (int c = 0; c < operand.HeadDim; c++) values[i++] = (float)Read(operand, b, h, r, c);
        return values;
    }

    private static double Read(AttentionOperand operand, int b, int h, int r, int c)
    {
        long index = operand.Offset + b * operand.BatchStride + h * operand.HeadStride + r * operand.TokenStride + c;
        void* data = operand.Storage.DataPointer;
        if (operand.DType == DType.F32) return ((float*)data)[index];
        ushort bits = ((ushort*)data)[index];
        return operand.DType == DType.F16 ? (double)BitConverter.UInt16BitsToHalf(bits)
            : BitConverter.UInt32BitsToSingle((uint)bits << 16);
    }

    /// <summary>Releases fixture-owned inputs and output.</summary>
    public void Dispose()
    {
        foreach (Tensor tensor in _owned) tensor.Dispose();
        _owned.Clear();
    }
}
