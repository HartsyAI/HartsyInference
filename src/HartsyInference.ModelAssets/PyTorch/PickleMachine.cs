using System.Text;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.PyTorch;

/// <summary>A torch storage reference recovered from a pickle <c>persistent_id</c>.</summary>
internal sealed record PickleStorage(DType DType, string Key, long NumEl);

/// <summary>A tensor reconstruction recovered from <c>torch._utils._rebuild_tensor_v2</c>.</summary>
internal sealed record PickleTensor(PickleStorage Storage, long Offset, long[] Size, long[] Stride);

/// <summary>A pickle <c>GLOBAL</c> reference (module + qualified name) — only torch reduce funcs and storage
/// types are interpreted; everything else is inert.</summary>
internal sealed record PickleGlobal(string Module, string Name);

/// <summary>Minimal, side-effect-free pickle interpreter covering exactly the opcodes a torch <c>state_dict</c>
/// emits (protocols 2–5). It builds plain CLR objects (dicts, lists, tuples, longs, strings) plus the torch
/// markers above. It never imports modules or executes code — unknown reduce targets resolve to null.</summary>
internal sealed class PickleMachine
{
    private static readonly object Mark = new();
    private readonly byte[] _b;
    private int _i;
    private readonly List<object?> _stack = [];
    private readonly List<object?> _memo = [];

    public PickleMachine(byte[] data, int start = 0) { _b = data; _i = start; }

    /// <summary>Byte offset just past the <c>STOP</c> opcode after <see cref="Run"/> returns. Used to parse the
    /// legacy <c>torch.save</c> format's five back-to-back pickle streams from one buffer.</summary>
    public int Position => _i;

    public object? Run()
    {
        while (_i < _b.Length)
        {
            byte op = _b[_i++];
            switch (op)
            {
                case 0x80: _i++; break;                       // PROTO (1-byte version)
                case 0x95: _i += 8; break;                    // FRAME (8-byte length)
                case (byte)'.': return Pop();                 // STOP

                case (byte)'(': _stack.Add(Mark); break;      // MARK
                case (byte)'}': _stack.Add(new Dictionary<string, object?>()); break; // EMPTY_DICT
                case (byte)']': _stack.Add(new List<object?>()); break;               // EMPTY_LIST
                case (byte)')': _stack.Add(Array.Empty<object?>()); break;            // EMPTY_TUPLE
                case (byte)'N': _stack.Add(null); break;      // NONE
                case 0x88: _stack.Add(true); break;           // NEWTRUE
                case 0x89: _stack.Add(false); break;          // NEWFALSE

                case (byte)'K': _stack.Add((long)_b[_i++]); break;            // BININT1
                case (byte)'M': _stack.Add((long)ReadU16()); break;          // BININT2
                case (byte)'J': _stack.Add((long)ReadI32()); break;          // BININT
                case 0x8a: _stack.Add(ReadLong1()); break;                   // LONG1
                case (byte)'G': _stack.Add(ReadF64()); break;                // BINFLOAT

                case 0x8c: _stack.Add(ReadStr(_b[_i++])); break;             // SHORT_BINUNICODE
                case (byte)'X': _stack.Add(ReadStr(ReadI32())); break;       // BINUNICODE
                case 0x8d: _stack.Add(ReadStr((int)ReadI64())); break;       // BINUNICODE8

                case (byte)'q': Memoize(_b[_i++]); break;                   // BINPUT
                case (byte)'r': Memoize(ReadI32()); break;                  // LONG_BINPUT
                case 0x94: _memo.Add(Peek()); break;                         // MEMOIZE
                case (byte)'h': _stack.Add(_memo[_b[_i++]]); break;          // BINGET
                case (byte)'j': _stack.Add(_memo[ReadI32()]); break;         // LONG_BINGET

                case 0x85: { object? a = Pop(); _stack.Add(new[] { a }); break; }            // TUPLE1
                case 0x86: { object? b2 = Pop(), a = Pop(); _stack.Add(new[] { a, b2 }); break; } // TUPLE2
                case 0x87: { object? c = Pop(), b2 = Pop(), a = Pop(); _stack.Add(new[] { a, b2, c }); break; } // TUPLE3
                case (byte)'t': _stack.Add(PopToMark().ToArray()); break;    // TUPLE

                case (byte)'s': { object? v = Pop(), k = Pop(); ((Dictionary<string, object?>)Peek()!)[Str(k)] = v; break; } // SETITEM
                case (byte)'u': SetItems(); break;                          // SETITEMS
                case (byte)'a': { object? v = Pop(); ((List<object?>)Peek()!).Add(v); break; } // APPEND
                case (byte)'e': Appends(); break;                           // APPENDS

                case (byte)'c': _stack.Add(new PickleGlobal(ReadLine(), ReadLine())); break;  // GLOBAL
                case 0x93: { object? n = Pop(), m = Pop(); _stack.Add(new PickleGlobal(Str(m), Str(n))); break; } // STACK_GLOBAL
                case (byte)'Q': _stack.Add(ResolvePersid(Pop())); break;     // BINPERSID
                case (byte)'R': Reduce(); break;                            // REDUCE
                // NEWOBJ (proto 2) / NEWOBJ_EX (proto 4): cls.__new__(cls, *args[, **kwargs]). In a state_dict
                // pickle the constructed object is a container (e.g. an OrderedDict/Module subclass) whose real
                // contents arrive via the following BUILD/SETITEMS — so push a fillable dict, mirroring how
                // REDUCE models OrderedDict. Args/kwargs/cls are inert for tensor extraction. (X-Codec's
                // ckpt_00360000.pth uses NEWOBJ; without this the loader threw "Unsupported pickle opcode 0x81".)
                case 0x81: { Pop(); Pop(); _stack.Add(new Dictionary<string, object?>()); break; }        // NEWOBJ (cls, args)
                case 0x92: { Pop(); Pop(); Pop(); _stack.Add(new Dictionary<string, object?>()); break; } // NEWOBJ_EX (cls, args, kwargs)
                case (byte)'b':                                            // BUILD (apply dict state if any)
                {
                    object? state = Pop();
                    if (Peek() is Dictionary<string, object?> d && state is Dictionary<string, object?> sd)
                        foreach ((string sk, object? sv) in sd) d[sk] = sv;
                    break;
                }

                default: throw new NotSupportedException($"Unsupported pickle opcode 0x{op:x2} at {_i - 1}.");
            }
        }
        throw new InvalidOperationException("Pickle stream ended without STOP.");
    }

    private void Reduce()
    {
        object?[] args = (object?[])Pop()!;
        object? func = Pop();
        if (func is PickleGlobal g)
        {
            switch (g.Name)
            {
                case "_rebuild_tensor_v2" or "_rebuild_tensor":
                    _stack.Add(new PickleTensor(
                        (PickleStorage)args[0]!, ToLong(args[1]), ToLongArray(args[2]), ToLongArray(args[3])));
                    return;
                case "_rebuild_parameter":
                    _stack.Add(args[0]); return; // wraps a tensor; unwrap
                case "OrderedDict":
                    _stack.Add(new Dictionary<string, object?>()); return;
            }
        }
        _stack.Add(null); // inert for any reduce we don't model
    }

    private object ResolvePersid(object? pid)
    {
        // ('storage', <storage_type GLOBAL>, key, location, numel)
        object?[] t = (object?[])pid!;
        PickleGlobal st = (PickleGlobal)t[1]!;
        return new PickleStorage(StorageDType(st.Name), Str(t[2]), ToLong(t[4]));
    }

    private static DType StorageDType(string name) => name switch
    {
        "FloatStorage" => DType.F32,
        "HalfStorage" => DType.F16,
        "BFloat16Storage" => DType.BF16,
        "DoubleStorage" => DType.F64,
        "LongStorage" => DType.I64,
        "IntStorage" => DType.I32,
        "CharStorage" => DType.I8,
        "ByteStorage" => DType.U8,
        "BoolStorage" => DType.Bool,
        _ => throw new NotSupportedException($"Unsupported torch storage type '{name}'."),
    };

    private void SetItems()
    {
        List<object?> items = PopToMark();
        Dictionary<string, object?> d = (Dictionary<string, object?>)Peek()!;
        for (int k = 0; k + 1 < items.Count; k += 2) d[Str(items[k])] = items[k + 1];
    }

    private void Appends()
    {
        List<object?> items = PopToMark();
        ((List<object?>)Peek()!).AddRange(items);
    }

    private List<object?> PopToMark()
    {
        int idx = _stack.LastIndexOf(Mark);
        List<object?> items = _stack.GetRange(idx + 1, _stack.Count - idx - 1);
        _stack.RemoveRange(idx, _stack.Count - idx);
        return items;
    }

    private void Memoize(int idx) { while (_memo.Count <= idx) _memo.Add(null); _memo[idx] = Peek(); }
    private object? Pop() { object? v = _stack[^1]; _stack.RemoveAt(_stack.Count - 1); return v; }
    private object? Peek() => _stack[^1];

    private ushort ReadU16() { ushort v = (ushort)(_b[_i] | (_b[_i + 1] << 8)); _i += 2; return v; }
    private int ReadI32() { int v = _b[_i] | (_b[_i + 1] << 8) | (_b[_i + 2] << 16) | (_b[_i + 3] << 24); _i += 4; return v; }
    private long ReadI64() { long v = BitConverter.ToInt64(_b, _i); _i += 8; return v; }
    private double ReadF64() { /* BINFLOAT is big-endian */ Span<byte> s = stackalloc byte[8]; for (int k = 0; k < 8; k++) s[k] = _b[_i + 7 - k]; _i += 8; return BitConverter.ToDouble(s); }
    private string ReadStr(int len) { string s = Encoding.UTF8.GetString(_b, _i, len); _i += len; return s; }

    private long ReadLong1()
    {
        int n = _b[_i++];
        long v = 0;
        for (int k = 0; k < n; k++) v |= (long)_b[_i + k] << (8 * k);
        if (n > 0 && n < 8 && (_b[_i + n - 1] & 0x80) != 0) v -= 1L << (8 * n); // sign-extend
        _i += n;
        return v;
    }

    private string ReadLine()
    {
        int start = _i;
        while (_b[_i] != (byte)'\n') _i++;
        string s = Encoding.ASCII.GetString(_b, start, _i - start);
        _i++; // consume newline
        return s;
    }

    private static string Str(object? o) => o as string ?? o?.ToString() ?? "";
    private static long ToLong(object? o) => o is long l ? l : Convert.ToInt64(o);
    private static long[] ToLongArray(object? o)
    {
        object?[] a = (object?[])o!;
        long[] r = new long[a.Length];
        for (int k = 0; k < a.Length; k++) r[k] = ToLong(a[k]);
        return r;
    }
}
