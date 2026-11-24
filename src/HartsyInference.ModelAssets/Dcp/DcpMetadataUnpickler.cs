using System.Text;
using HartsyInference.ModelAssets.PyTorch;

namespace HartsyInference.ModelAssets.Dcp;

/// <summary>Allow-list pickle interpreter for the <c>.metadata</c> file of a PyTorch Distributed Checkpoint. It builds plain CLR values (<see cref="DcpDict"/>, lists, tuples, <see cref="DcpObject"/>) and never imports a module or calls anything:
/// a GLOBAL outside <see cref="IsAllowedGlobal"/> fails the read, and every reduce or object construction yields an inert stub. The opcode set is the one pickle protocols 2 to 5 emit for dataclasses, dicts, lists and tuples.
/// <see cref="PickleGlobal"/> is shared with the torch checkpoint reader; its <c>PickleMachine</c> cannot be used as is because its dicts take string keys only.</summary>
internal sealed class DcpMetadataUnpickler
{
    private static readonly object Mark = new();
    private static readonly HashSet<string> TorchNames = new(StringComparer.Ordinal)
    {
        "Size", "float32", "float16", "bfloat16", "float64", "int8", "int16", "int32", "int64", "uint8", "bool", "complex64",
        "float8_e4m3fn", "float8_e5m2", "strided", "sparse_coo", "contiguous_format", "channels_last", "channels_last_3d",
        "preserve_format", "device",
    };
    private static readonly HashSet<string> BuiltinNames = new(StringComparer.Ordinal) { "set", "frozenset", "dict", "list", "tuple" };
    private static readonly HashSet<string> CopyregNames = new(StringComparer.Ordinal) { "_reconstructor", "__newobj__", "__newobj_ex__" };
    private static readonly HashSet<string> PathNames = new(StringComparer.Ordinal)
    {
        "PosixPath", "PurePosixPath", "WindowsPath", "PureWindowsPath",
    };

    private readonly byte[] _b;
    private int _i;
    private readonly List<object?> _stack = [];
    private readonly List<object?> _memo = [];

    public DcpMetadataUnpickler(byte[] data)
    {
        ArgumentNullException.ThrowIfNull(data);
        _b = data;
    }

    /// <summary>Whether a pickle GLOBAL may appear in a DCP metadata file.</summary>
    public static bool IsAllowedGlobal(string module, string name) => module switch
    {
        "torch" => TorchNames.Contains(name),
        "collections" => name == "OrderedDict",
        "builtins" => BuiltinNames.Contains(name),
        "copyreg" => CopyregNames.Contains(name),
        "pathlib" => PathNames.Contains(name),
        _ => module == "torch.distributed.checkpoint" || module.StartsWith("torch.distributed.checkpoint.", StringComparison.Ordinal)
            || module.StartsWith("torch.distributed._shard.", StringComparison.Ordinal),
    };

    public object? Run()
    {
        while (_i < _b.Length)
        {
            byte op = _b[_i++];
            switch (op)
            {
                case 0x80: _i++; break;
                case 0x95: _i += 8; break;
                case (byte)'.': return Pop();
                case (byte)'(': _stack.Add(Mark); break;
                case (byte)'}': _stack.Add(new DcpDict()); break;
                case (byte)']': _stack.Add(new List<object?>()); break;
                case (byte)')': _stack.Add(Array.Empty<object?>()); break;
                case 0x8f: _stack.Add(new List<object?>()); break;
                case (byte)'N': _stack.Add(null); break;
                case 0x88: _stack.Add(true); break;
                case 0x89: _stack.Add(false); break;
                case (byte)'K': _stack.Add((long)_b[_i++]); break;
                case (byte)'M': _stack.Add((long)ReadU16()); break;
                case (byte)'J': _stack.Add((long)ReadI32()); break;
                case 0x8a: _stack.Add(ReadLong(_b[_i++])); break;
                case 0x8b: _stack.Add(ReadLong(ReadI32())); break;
                case (byte)'G': _stack.Add(ReadF64()); break;
                case 0x8c: _stack.Add(ReadStr(_b[_i++])); break;
                case (byte)'X': _stack.Add(ReadStr(ReadI32())); break;
                case 0x8d: _stack.Add(ReadStr(checked((int)ReadI64()))); break;
                case (byte)'C': _stack.Add(ReadBytes(_b[_i++])); break;
                case (byte)'B': _stack.Add(ReadBytes(ReadI32())); break;
                case 0x8e: _stack.Add(ReadBytes(checked((int)ReadI64()))); break;
                case (byte)'q': Memoize(_b[_i++]); break;
                case (byte)'r': Memoize(ReadI32()); break;
                case 0x94: _memo.Add(Peek()); break;
                case (byte)'h': _stack.Add(_memo[_b[_i++]]); break;
                case (byte)'j': _stack.Add(_memo[ReadI32()]); break;
                case 0x85: { object? a = Pop(); _stack.Add(new[] { a }); break; }
                case 0x86: { object? b = Pop(), a = Pop(); _stack.Add(new[] { a, b }); break; }
                case 0x87: { object? c = Pop(), b = Pop(), a = Pop(); _stack.Add(new[] { a, b, c }); break; }
                case (byte)'t': _stack.Add(PopToMark().ToArray()); break;
                case (byte)'s': { object? v = Pop(), k = Pop(); ((DcpDict)Peek()!).Entries.Add(new(k, v)); break; }
                case (byte)'u': SetItems(); break;
                case (byte)'a': { object? v = Pop(); ((List<object?>)Peek()!).Add(v); break; }
                case (byte)'e':
                case 0x90: { List<object?> items = PopToMark(); ((List<object?>)Peek()!).AddRange(items); break; }
                case 0x91: _stack.Add(PopToMark()); break;
                case (byte)'c': _stack.Add(Global(ReadLine(), ReadLine())); break;
                case 0x93: { object? n = Pop(), m = Pop(); _stack.Add(Global(Text(m), Text(n))); break; }
                case (byte)'R': Reduce(); break;
                case 0x81: { Pop(); _stack.Add(NewObject(Pop())); break; }
                case 0x92: { Pop(); Pop(); _stack.Add(NewObject(Pop())); break; }
                case (byte)'b': Build(); break;
                default: throw new NotSupportedException($"Unsupported pickle opcode 0x{op:x2} at {_i - 1} in a distributed-checkpoint metadata file.");
            }
        }
        throw new InvalidDataException("The metadata pickle ended without a STOP opcode.");
    }

    private static PickleGlobal Global(string module, string name)
        => IsAllowedGlobal(module, name)
            ? new PickleGlobal(module, name)
            : throw new InvalidDataException(
                $"The metadata pickle references '{module}.{name}', which a distributed-checkpoint metadata file has no reason to contain; "
                + "the file was not read. Review it, and extend DcpMetadataUnpickler.IsAllowedGlobal only for a class that holds plain data.");

    private void Reduce()
    {
        object?[] args = (object?[])Pop()!;
        PickleGlobal callee = (PickleGlobal)Pop()!;
        if (callee is { Module: "torch", Name: "Size" })
        {
            object?[] dims = args.Length == 1 && args[0] is object?[] t ? t : [];
            long[] size = new long[dims.Length];
            for (int k = 0; k < size.Length; k++) size[k] = Convert.ToInt64(dims[k]);
            _stack.Add(size);
        }
        else if (callee is { Module: "collections", Name: "OrderedDict" } or { Module: "builtins", Name: "dict" })
        {
            _stack.Add(new DcpDict());
        }
        else if (callee is { Module: "builtins", Name: "set" or "frozenset" or "list" })
        {
            _stack.Add(args.Length == 1 && args[0] is List<object?> items ? new List<object?>(items) : []);
        }
        else if (callee is { Module: "builtins", Name: "tuple" })
        {
            _stack.Add(args.Length == 1 && args[0] is List<object?> items ? items.ToArray() : []);
        }
        else
        {
            _stack.Add(new DcpObject(callee.Module, callee.Name) { Args = args });
        }
    }

    private static DcpObject NewObject(object? cls)
        => cls is PickleGlobal g ? new DcpObject(g.Module, g.Name) : throw new InvalidDataException("An object was constructed from something that is not a class.");

    private void Build()
    {
        object? state = Pop();
        if (Peek() is not DcpObject target) return;
        if (state is DcpDict fields) Assign(target, fields);
        else if (state is object?[] { Length: 2 } pair && pair[0] is DcpDict first) Assign(target, first);
    }

    private static void Assign(DcpObject target, DcpDict fields)
    {
        foreach (KeyValuePair<object?, object?> entry in fields.Entries)
            if (entry.Key is string key) target.Fields[key] = entry.Value;
    }

    private void SetItems()
    {
        List<object?> items = PopToMark();
        DcpDict dict = (DcpDict)Peek()!;
        for (int k = 0; k + 1 < items.Count; k += 2) dict.Entries.Add(new(items[k], items[k + 1]));
    }

    private List<object?> PopToMark()
    {
        int idx = _stack.LastIndexOf(Mark);
        if (idx < 0) throw new InvalidDataException("The metadata pickle closed a group that was never opened.");
        List<object?> items = _stack.GetRange(idx + 1, _stack.Count - idx - 1);
        _stack.RemoveRange(idx, _stack.Count - idx);
        return items;
    }

    private void Memoize(int idx)
    {
        while (_memo.Count <= idx) _memo.Add(null);
        _memo[idx] = Peek();
    }

    private object? Pop()
    {
        if (_stack.Count == 0) throw new InvalidDataException("The metadata pickle popped an empty stack.");
        object? v = _stack[^1];
        _stack.RemoveAt(_stack.Count - 1);
        return v;
    }

    private object? Peek() => _stack.Count == 0 ? throw new InvalidDataException("The metadata pickle read an empty stack.") : _stack[^1];

    private ushort ReadU16() { ushort v = (ushort)(_b[_i] | (_b[_i + 1] << 8)); _i += 2; return v; }

    private int ReadI32() { int v = BitConverter.ToInt32(_b, _i); _i += 4; return v; }

    private long ReadI64() { long v = BitConverter.ToInt64(_b, _i); _i += 8; return v; }

    private double ReadF64()
    {
        Span<byte> s = stackalloc byte[8];
        for (int k = 0; k < 8; k++) s[k] = _b[_i + 7 - k];
        _i += 8;
        return BitConverter.ToDouble(s);
    }

    private string ReadStr(int len) { string s = Encoding.UTF8.GetString(_b, _i, len); _i += len; return s; }

    private byte[] ReadBytes(int len) { byte[] r = _b.AsSpan(_i, len).ToArray(); _i += len; return r; }

    private long ReadLong(int n)
    {
        if (n > 8) throw new NotSupportedException("A metadata integer wider than 64 bits.");
        long v = 0;
        for (int k = 0; k < n; k++) v |= (long)_b[_i + k] << (8 * k);
        if (n > 0 && n < 8 && (_b[_i + n - 1] & 0x80) != 0) v -= 1L << (8 * n);
        _i += n;
        return v;
    }

    private string ReadLine()
    {
        int start = _i;
        while (_b[_i] != (byte)'\n') _i++;
        string s = Encoding.ASCII.GetString(_b, start, _i - start);
        _i++;
        return s;
    }

    private static string Text(object? o) => o as string ?? throw new InvalidDataException("A STACK_GLOBAL name was not a string.");
}
