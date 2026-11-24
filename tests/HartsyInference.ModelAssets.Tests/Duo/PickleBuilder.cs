using System.Text;

namespace HartsyInference.ModelAssets.Tests.Duo;

/// <summary>Emits the pickle opcodes Python writes for dataclasses, dicts, lists and tuples (protocol 4 for DCP metadata, protocol 2 for torch.save), so tests can build a metadata file without Python.</summary>
internal sealed class PickleBuilder
{
    private readonly MemoryStream _out = new();
    private readonly bool _proto4;

    public PickleBuilder(bool proto4 = true)
    {
        _proto4 = proto4;
        _out.WriteByte(0x80);
        _out.WriteByte(proto4 ? (byte)4 : (byte)2);
    }

    public byte[] Finish()
    {
        _out.WriteByte((byte)'.');
        return _out.ToArray();
    }

    public PickleBuilder Op(byte op) { _out.WriteByte(op); return this; }

    public PickleBuilder None() => Op((byte)'N');

    public PickleBuilder Bool(bool value) => Op(value ? (byte)0x88 : (byte)0x89);

    public PickleBuilder Int(long value)
    {
        if (value is >= 0 and < 256) { Op((byte)'K'); Op((byte)value); }
        else if (value is >= 0 and < 65536) { Op((byte)'M'); Op((byte)value); Op((byte)(value >> 8)); }
        else if (value is >= int.MinValue and <= int.MaxValue) { Op((byte)'J'); _out.Write(BitConverter.GetBytes((int)value)); }
        else { Op(0x8a); Op(8); _out.Write(BitConverter.GetBytes(value)); }
        return this;
    }

    public PickleBuilder Str(string value)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(value);
        if (bytes.Length < 256) { Op(0x8c); Op((byte)bytes.Length); }
        else { Op((byte)'X'); _out.Write(BitConverter.GetBytes(bytes.Length)); }
        _out.Write(bytes);
        return this;
    }

    /// <summary>A class reference: STACK_GLOBAL in protocol 4, GLOBAL in protocol 2.</summary>
    public PickleBuilder Global(string module, string name)
    {
        if (_proto4) { Str(module); Str(name); Op(0x93); }
        else { Op((byte)'c'); _out.Write(Encoding.ASCII.GetBytes($"{module}\n{name}\n")); }
        return this;
    }

    public PickleBuilder Tuple(params Action<PickleBuilder>[] items)
    {
        Op((byte)'(');
        foreach (Action<PickleBuilder> item in items) item(this);
        return Op((byte)'t');
    }

    public PickleBuilder List(IEnumerable<Action<PickleBuilder>> items)
    {
        Op((byte)']');
        Op((byte)'(');
        foreach (Action<PickleBuilder> item in items) item(this);
        return Op((byte)'e');
    }

    public PickleBuilder Dict(IEnumerable<(Action<PickleBuilder> Key, Action<PickleBuilder> Value)> entries)
    {
        Op((byte)'}');
        Op((byte)'(');
        foreach ((Action<PickleBuilder> key, Action<PickleBuilder> value) in entries) { key(this); value(this); }
        return Op((byte)'u');
    }

    /// <summary><c>torch.Size([...])</c> as pickle writes it: the class, a tuple of the dims, REDUCE.</summary>
    public PickleBuilder TorchSize(params long[] dims)
    {
        Global("torch", "Size");
        Tuple(b => b.Tuple([.. dims.Select<long, Action<PickleBuilder>>(d => x => x.Int(d))]));
        return Op((byte)'R');
    }

    /// <summary>An object whose state is a tuple: class, empty args, NEWOBJ, the tuple, BUILD (what a class with a custom <c>__getstate__</c> returning a tuple pickles to).</summary>
    public PickleBuilder ObjectWithTupleState(string module, string name, params Action<PickleBuilder>[] state)
    {
        Global(module, name);
        Op((byte)')');
        Op(0x81);
        Tuple(state);
        return Op((byte)'b');
    }

    /// <summary>An object with a state dict: class, empty args, NEWOBJ, then the fields through BUILD, as <c>object.__reduce_ex__</c> does for a dataclass.</summary>
    public PickleBuilder Object(string module, string name, params (string Field, Action<PickleBuilder> Value)[] fields)
    {
        Global(module, name);
        Op((byte)')');
        Op(0x81);
        Dict(fields.Select<(string Field, Action<PickleBuilder> Value), (Action<PickleBuilder>, Action<PickleBuilder>)>(
            f => (b => b.Str(f.Field), f.Value)));
        return Op((byte)'b');
    }
}
