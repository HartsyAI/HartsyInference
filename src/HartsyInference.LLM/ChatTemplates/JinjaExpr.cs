using System.Globalization;
using System.Text;

namespace HartsyInference.LLM.ChatTemplates;

/// <summary>Expression AST + recursive-descent parser + value runtime for <see cref="JinjaEngine"/>.</summary>
internal abstract class Expr
{
    public abstract object? Eval(JinjaEngine.Scope scope);
}

/// <summary>Dynamic-typed value helpers (string / bool / long / double / null / list / dict).</summary>
internal static class Values
{
    public static bool Truthy(object? v) => v switch
    {
        null => false,
        bool b => b,
        string s => s.Length > 0,
        long l => l != 0,
        double d => d != 0,
        System.Collections.ICollection c => c.Count > 0,
        _ => true,
    };

    public static string ToStr(object? v) => v switch
    {
        null => string.Empty,
        bool b => b ? "True" : "False",
        string s => s,
        long l => l.ToString(CultureInfo.InvariantCulture),
        double d => d.ToString(CultureInfo.InvariantCulture),
        _ => v.ToString() ?? string.Empty,
    };

    // Jinja/Python iterates a string character-by-character (not an error) — e.g. Llama-3.2-Vision's chat
    // template does `for content in message['content']` unconditionally (before checking `content is string`)
    // to scan every message for image parts; a plain-string content must degrade to a char sequence, not throw.
    public static List<object?> AsList(object? v) => v switch
    {
        null => [],
        List<object?> l => l,
        string s => [.. s.Select(ch => (object?)ch.ToString())],
        System.Collections.IEnumerable e => [.. e.Cast<object?>()],
        _ => throw new InvalidOperationException($"Value is not iterable: {v.GetType().Name}"),
    };

    /// <summary>Serializes a value to compact JSON for the <c>tojson</c> filter (used by the tool-calling blocks
    /// of the Qwen/Llama templates). Handles the value graph the engine produces: null / bool / long / double /
    /// string / List / Dictionary.</summary>
    public static string ToJson(object? v)
    {
        StringBuilder sb = new();
        WriteJson(sb, v);
        return sb.ToString();
    }

    private static void WriteJson(StringBuilder sb, object? v)
    {
        switch (v)
        {
            case null: sb.Append("null"); break;
            case bool b: sb.Append(b ? "true" : "false"); break;
            case long l: sb.Append(l.ToString(CultureInfo.InvariantCulture)); break;
            case int i: sb.Append(i.ToString(CultureInfo.InvariantCulture)); break;
            case double d: sb.Append(d.ToString("R", CultureInfo.InvariantCulture)); break;
            case string s: WriteJsonString(sb, s); break;
            case Dictionary<string, object?> dict:
                sb.Append('{');
                bool firstK = true;
                foreach (KeyValuePair<string, object?> kv in dict)
                {
                    if (!firstK) sb.Append(", ");
                    firstK = false;
                    WriteJsonString(sb, kv.Key);
                    sb.Append(": ");
                    WriteJson(sb, kv.Value);
                }
                sb.Append('}');
                break;
            default:
                sb.Append('[');
                bool firstE = true;
                foreach (object? e in AsList(v))
                {
                    if (!firstE) sb.Append(", ");
                    firstE = false;
                    WriteJson(sb, e);
                }
                sb.Append(']');
                break;
        }
    }

    private static void WriteJsonString(StringBuilder sb, string s)
    {
        sb.Append('"');
        foreach (char c in s)
            switch (c)
            {
                case '"': sb.Append("\\\""); break;
                case '\\': sb.Append("\\\\"); break;
                case '\n': sb.Append("\\n"); break;
                case '\r': sb.Append("\\r"); break;
                case '\t': sb.Append("\\t"); break;
                default:
                    if (c < 0x20) sb.Append("\\u").Append(((int)c).ToString("x4", CultureInfo.InvariantCulture));
                    else sb.Append(c);
                    break;
            }
        sb.Append('"');
    }

    public static bool Eq(object? a, object? b)
    {
        if (a is null || b is null) return a is null && b is null;
        if (a is long la && b is long lb) return la == lb;
        if (a is bool ba && b is bool bb) return ba == bb;
        if (IsNum(a) && IsNum(b)) return ToD(a) == ToD(b);
        return string.Equals(ToStr(a), ToStr(b), StringComparison.Ordinal) && a is string == b is string
            || (a is string && b is string && (string)a == (string)b);
    }

    public static bool In(object? item, object? container) => container switch
    {
        string s => s.Contains(ToStr(item), StringComparison.Ordinal),
        Dictionary<string, object?> d => d.ContainsKey(ToStr(item)),
        System.Collections.IEnumerable e => e.Cast<object?>().Any(x => Eq(x, item)),
        _ => false,
    };

    public static bool IsNum(object? v) => v is long or double or int;
    public static double ToD(object? v) => v switch { long l => l, int i => i, double d => d, _ => 0 };
}

// ── Literal / variable / unary / binary / ternary ──────────────────────────────────────────────────────
internal sealed class LiteralExpr(object? value) : Expr
{
    public override object? Eval(JinjaEngine.Scope scope) => value;
}

internal sealed class VarExpr(string name) : Expr
{
    public override object? Eval(JinjaEngine.Scope scope) => scope.Get(name);
    public string GetName() => name;
}

internal sealed class MemberExpr(Expr target, string name) : Expr
{
    public override object? Eval(JinjaEngine.Scope scope)
    {
        object? t = target.Eval(scope);
        return t is Dictionary<string, object?> d && d.TryGetValue(name, out object? v) ? v : null;
    }
    public string Name => name;
    public Expr Target => target;
}

internal sealed class IndexExpr(Expr target, Expr index) : Expr
{
    public override object? Eval(JinjaEngine.Scope scope)
    {
        object? t = target.Eval(scope);
        object? k = index.Eval(scope);
        if (t is Dictionary<string, object?> d) return d.TryGetValue(Values.ToStr(k), out object? v) ? v : null;
        if (t is List<object?> l && k is long i && i >= 0 && i < l.Count) return l[(int)i];
        return null;
    }
}

internal sealed class UnaryNotExpr(Expr inner) : Expr
{
    public override object? Eval(JinjaEngine.Scope scope) => !Values.Truthy(inner.Eval(scope));
}

internal sealed class SliceExpr(Expr target, Expr? start, Expr? stop) : Expr
{
    public override object? Eval(JinjaEngine.Scope scope)
    {
        object? t = target.Eval(scope);
        List<object?> list = Values.AsList(t);
        int n = list.Count;
        int lo = Norm(start?.Eval(scope), 0, n);
        int hi = Norm(stop?.Eval(scope), n, n);
        if (lo < 0) lo = 0; if (hi > n) hi = n; if (hi < lo) hi = lo;
        List<object?> slice = list.GetRange(lo, hi - lo);
        return t is string ? string.Concat(slice.Select(Values.ToStr)) : slice;
    }
    private static int Norm(object? v, int def, int n)
    {
        if (v is null) return def;
        int i = (int)Values.ToD(v);
        return i < 0 ? n + i : i;
    }
}

internal sealed class BinaryExpr(string op, Expr left, Expr right) : Expr
{
    public override object? Eval(JinjaEngine.Scope scope)
    {
        if (op == "and") return Values.Truthy(left.Eval(scope)) ? right.Eval(scope) : left.Eval(scope);
        if (op == "or") { object? l = left.Eval(scope); return Values.Truthy(l) ? l : right.Eval(scope); }

        object? a = left.Eval(scope);
        object? b = right.Eval(scope);
        return op switch
        {
            "==" => Values.Eq(a, b),
            "!=" => !Values.Eq(a, b),
            "in" => Values.In(a, b),
            "notin" => !Values.In(a, b),
            "<" => Values.ToD(a) < Values.ToD(b),
            ">" => Values.ToD(a) > Values.ToD(b),
            "<=" => Values.ToD(a) <= Values.ToD(b),
            ">=" => Values.ToD(a) >= Values.ToD(b),
            "~" => Values.ToStr(a) + Values.ToStr(b),
            "+" => Values.IsNum(a) && Values.IsNum(b) ? (object)((long)Values.ToD(a) + (long)Values.ToD(b)) : Values.ToStr(a) + Values.ToStr(b),
            "-" => (object)((long)Values.ToD(a) - (long)Values.ToD(b)),
            "*" => (object)((long)Values.ToD(a) * (long)Values.ToD(b)),
            "//" => (object)((long)Values.ToD(a) / (long)Values.ToD(b)),
            "/" => (object)(Values.ToD(a) / Values.ToD(b)),
            "%" => (object)((long)Values.ToD(a) % (long)Values.ToD(b)),
            _ => throw new NotSupportedException($"Jinja operator '{op}'"),
        };
    }
}

/// <summary>Unary <c>-x</c> — negates a numeric operand (long, matching <see cref="BinaryExpr"/>'s integer
/// arithmetic; every Jinja number literal here is a long).</summary>
internal sealed class UnaryMinusExpr(Expr operand) : Expr
{
    public override object? Eval(JinjaEngine.Scope scope) => -(long)Values.ToD(operand.Eval(scope));
}

internal sealed class TernaryExpr(Expr cond, Expr ifTrue, Expr ifFalse) : Expr
{
    public override object? Eval(JinjaEngine.Scope scope) => Values.Truthy(cond.Eval(scope)) ? ifTrue.Eval(scope) : ifFalse.Eval(scope);
}

internal sealed class FilterExpr(Expr inner, string name, List<Expr> args) : Expr
{
    public override object? Eval(JinjaEngine.Scope scope)
    {
        object? v = inner.Eval(scope);
        switch (name)
        {
            case "trim": return Values.ToStr(v).Trim();
            case "lower": return Values.ToStr(v).ToLowerInvariant();
            case "upper": return Values.ToStr(v).ToUpperInvariant();
            // length/count: char count for strings, element count for sequences/dicts (Jinja semantics).
            case "length": case "count":
                return v is string sLen ? (long)sLen.Length : (long)Values.AsList(v).Count;
            case "first":
                if (v is string sFirst) return sFirst.Length > 0 ? sFirst[0].ToString() : "";
                { List<object?> lf = Values.AsList(v); return lf.Count > 0 ? lf[0] : null; }
            case "last":
                if (v is string sLast) return sLast.Length > 0 ? sLast[^1].ToString() : "";
                { List<object?> ll = Values.AsList(v); return ll.Count > 0 ? ll[^1] : null; }
            case "join":
                { string sep = args.Count > 0 ? Values.ToStr(args[0].Eval(scope)) : "";
                  return string.Join(sep, Values.AsList(v).Select(Values.ToStr)); }
            case "tojson": return Values.ToJson(v);
            case "list": return Values.AsList(v);
            case "string": return Values.ToStr(v);
            case "default": return Values.Truthy(v) ? v : (args.Count > 0 ? args[0].Eval(scope) : v);
            default: throw new NotSupportedException($"Jinja filter '{name}'");
        }
    }
}

/// <summary>A keyword argument <c>name=value</c> in a call. Evaluates to its value so positional consumers work;
/// <see cref="CallExpr"/> for <c>namespace(...)</c> inspects the name.</summary>
internal sealed class NamedArgExpr(string name, Expr value) : Expr
{
    public string Name => name;
    public Expr Value => value;
    public override object? Eval(JinjaEngine.Scope scope) => value.Eval(scope);
}

internal sealed class CallExpr(Expr callee, List<Expr> args) : Expr
{
    public override object? Eval(JinjaEngine.Scope scope)
    {
        // Method calls on the result of a member access (e.g. content.strip()).
        if (callee is MemberExpr m)
        {
            object? target = m.Target.Eval(scope);
            string s = Values.ToStr(target);
            object? Arg(int idx) => idx < args.Count ? args[idx].Eval(scope) : null;
            return m.Name switch
            {
                "strip" => s.Trim(),
                "lstrip" => s.TrimStart(),
                "rstrip" => s.TrimEnd(),
                "lower" => s.ToLowerInvariant(),
                "upper" => s.ToUpperInvariant(),
                "title" => CultureInfo.InvariantCulture.TextInfo.ToTitleCase(s),
                "capitalize" => s.Length == 0 ? s : char.ToUpperInvariant(s[0]) + s[1..].ToLowerInvariant(),
                "startswith" => s.StartsWith(Values.ToStr(Arg(0)), StringComparison.Ordinal),
                "endswith" => s.EndsWith(Values.ToStr(Arg(0)), StringComparison.Ordinal),
                "replace" => s.Replace(Values.ToStr(Arg(0)), Values.ToStr(Arg(1))),
                "split" => SplitString(s, Arg(0)),
                "get" => target is Dictionary<string, object?> d ? (d.TryGetValue(Values.ToStr(Arg(0)), out object? gv) ? gv : Arg(1)) : Arg(1),
                _ => throw new NotSupportedException($"Jinja method '{m.Name}'"),
            };
        }
        // Bare function calls templates use: raise_exception, namespace, strftime_now.
        if (callee is VarExpr v)
        {
            switch (v.GetName())
            {
                case "raise_exception" or "raise":
                    throw new ChatTemplateRaiseException("chat template raised: " + (args.Count > 0 ? Values.ToStr(args[0].Eval(scope)) : ""));
                case "namespace":
                {
                    Dictionary<string, object?> ns = new();
                    foreach (Expr a in args)
                        if (a is NamedArgExpr na) ns[na.Name] = na.Value.Eval(scope);
                    return ns;
                }
                case "strftime_now":
                    // Date-string helper used in some default system prompts; value is not load-bearing.
                    return "01 Jan 2025";
                case "range":
                {
                    // Python/Jinja range(stop) / range(start, stop) / range(start, stop, step) — Gemma-4's chat
                    // template walks message history backwards via range(idx-1, -1, -1).
                    long start = 0, step = 1;
                    long stop = (long)Values.ToD(args[0].Eval(scope));
                    if (args.Count >= 2) { start = stop; stop = (long)Values.ToD(args[1].Eval(scope)); }
                    if (args.Count >= 3) step = (long)Values.ToD(args[2].Eval(scope));
                    List<object?> result = [];
                    if (step > 0) for (long x = start; x < stop; x += step) result.Add(x);
                    else if (step < 0) for (long x = start; x > stop; x += step) result.Add(x);
                    return result;
                }
            }
        }
        throw new NotSupportedException("Unsupported Jinja call expression.");
    }

    private static List<object?> SplitString(string s, object? sep)
    {
        string[] parts = sep is null
            ? s.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries)
            : s.Split(Values.ToStr(sep));
        List<object?> list = new(parts.Length);
        foreach (string p in parts) list.Add(p);
        return list;
    }
}

internal static class ExprParser
{
    public static Expr ParseExpr(string src)
    {
        Tokenizer tz = new(src);
        Expr e = ParseTernary(tz);
        tz.ExpectEnd();
        return e;
    }

    private static Expr ParseTernary(Tokenizer t)
    {
        Expr e = ParseOr(t);
        if (t.TryKeyword("if"))
        {
            Expr cond = ParseOr(t);
            t.ExpectKeyword("else");
            Expr other = ParseTernary(t);
            return new TernaryExpr(cond, e, other);
        }
        return e;
    }

    private static Expr ParseOr(Tokenizer t)
    {
        Expr e = ParseAnd(t);
        while (t.TryKeyword("or")) e = new BinaryExpr("or", e, ParseAnd(t));
        return e;
    }

    private static Expr ParseAnd(Tokenizer t)
    {
        Expr e = ParseNot(t);
        while (t.TryKeyword("and")) e = new BinaryExpr("and", e, ParseNot(t));
        return e;
    }

    private static Expr ParseNot(Tokenizer t)
    {
        if (t.TryKeyword("not")) return new UnaryNotExpr(ParseNot(t));
        return ParseComparison(t);
    }

    private static Expr ParseComparison(Tokenizer t)
    {
        Expr e = ParseConcat(t);
        while (true)
        {
            if (t.TryKeyword("is"))
            {
                bool neg = t.TryKeyword("not");
                string test = t.ReadIdent();
                // 'is none'/'is not none' may also appear as 'is not none'; 'defined' etc. take no args.
                e = new IsTestExpr(e, test, neg);
                continue;
            }
            if (t.TrySymbol("==")) e = new BinaryExpr("==", e, ParseConcat(t));
            else if (t.TrySymbol("!=")) e = new BinaryExpr("!=", e, ParseConcat(t));
            else if (t.TrySymbol("<=")) e = new BinaryExpr("<=", e, ParseConcat(t));
            else if (t.TrySymbol(">=")) e = new BinaryExpr(">=", e, ParseConcat(t));
            else if (t.TrySymbol("<")) e = new BinaryExpr("<", e, ParseConcat(t));
            else if (t.TrySymbol(">")) e = new BinaryExpr(">", e, ParseConcat(t));
            else if (t.TryKeyword("not")) { t.ExpectKeyword("in"); e = new BinaryExpr("notin", e, ParseConcat(t)); }
            else if (t.TryKeyword("in")) e = new BinaryExpr("in", e, ParseConcat(t));
            else break;
        }
        return e;
    }

    private static Expr ParseConcat(Tokenizer t)
    {
        Expr e = ParseMulDiv(t);
        while (true)
        {
            if (t.TrySymbol("~")) e = new BinaryExpr("~", e, ParseMulDiv(t));
            else if (t.TrySymbol("+")) e = new BinaryExpr("+", e, ParseMulDiv(t));
            else if (t.TrySymbol("-")) e = new BinaryExpr("-", e, ParseMulDiv(t));
            else break;
        }
        return e;
    }

    /// <summary>Multiplicative level (binds tighter than +/-): <c>*</c>, <c>//</c> (int div), <c>/</c>, <c>%</c>
    /// (modulo — used by chat templates like Gemma's <c>loop.index0 % 2</c>). <c>//</c> is tried before <c>/</c>.</summary>
    private static Expr ParseMulDiv(Tokenizer t)
    {
        Expr e = ParseUnary(t);
        while (true)
        {
            if (t.TrySymbol("*")) e = new BinaryExpr("*", e, ParseUnary(t));
            else if (t.TrySymbol("//")) e = new BinaryExpr("//", e, ParseUnary(t));
            else if (t.TrySymbol("/")) e = new BinaryExpr("/", e, ParseUnary(t));
            else if (t.TrySymbol("%")) e = new BinaryExpr("%", e, ParseUnary(t));
            else break;
        }
        return e;
    }

    /// <summary>Unary <c>-</c>/<c>+</c> (e.g. a literal <c>-1</c> arg to <c>range(x, -1, -1)</c> — a real
    /// construct in Gemma-4's tool-calling chat template). Binds tighter than <c>*</c>/<c>/</c>, matching
    /// Python/Jinja precedence; right-recursive so <c>--x</c> parses (harmless, just unusual).</summary>
    private static Expr ParseUnary(Tokenizer t)
    {
        if (t.TrySymbol("-")) return new UnaryMinusExpr(ParseUnary(t));
        if (t.TrySymbol("+")) return ParseUnary(t);
        return ParsePostfix(t);
    }

    private static Expr ParsePostfix(Tokenizer t)
    {
        Expr e = ParsePrimary(t);
        while (true)
        {
            if (t.TrySymbol("."))
            {
                string name = t.ReadIdent();
                e = new MemberExpr(e, name);
            }
            else if (t.TrySymbol("["))
            {
                Expr? start = null;
                if (!t.TrySymbol(":"))
                {
                    start = ParseTernary(t);
                    if (t.TrySymbol("]")) { e = new IndexExpr(e, start); continue; }
                    t.ExpectSymbol(":");
                }
                Expr? stop = t.TrySymbol("]") ? null : ParseTernary(t);
                if (stop is not null) t.ExpectSymbol("]");
                e = new SliceExpr(e, start, stop);
            }
            else if (t.TrySymbol("("))
            {
                List<Expr> args = ParseArgs(t);
                e = new CallExpr(e, args);
            }
            else if (t.TrySymbol("|"))
            {
                string fname = t.ReadIdent();
                List<Expr> args = t.TrySymbol("(") ? ParseArgs(t) : [];
                e = new FilterExpr(e, fname, args);
            }
            else break;
        }
        return e;
    }

    private static List<Expr> ParseArgs(Tokenizer t)
    {
        List<Expr> args = [];
        if (t.TrySymbol(")")) return args;
        do
        {
            // Keyword argument: ident '=' expr (but not '==', which is a comparison).
            int save = t.Save();
            if (t.TryReadIdent(out string name) && !t.PeekSymbol("==") && t.TrySymbol("="))
                args.Add(new NamedArgExpr(name, ParseTernary(t)));
            else { t.Restore(save); args.Add(ParseTernary(t)); }
        } while (t.TrySymbol(","));
        t.ExpectSymbol(")");
        return args;
    }

    private static Expr ParsePrimary(Tokenizer t)
    {
        if (t.TryString(out string str)) return new LiteralExpr(str);
        if (t.TryNumber(out long num)) return new LiteralExpr(num);
        if (t.TrySymbol("("))
        {
            Expr e = ParseTernary(t);
            t.ExpectSymbol(")");
            return e;
        }
        if (t.TrySymbol("["))
        {
            List<Expr> items = [];
            if (!t.TrySymbol("]"))
            {
                do { items.Add(ParseTernary(t)); } while (t.TrySymbol(","));
                t.ExpectSymbol("]");
            }
            return new ListLiteralExpr(items);
        }
        string ident = t.ReadIdent();
        return ident switch
        {
            "true" or "True" => new LiteralExpr(true),
            "false" or "False" => new LiteralExpr(false),
            "none" or "None" or "null" => new LiteralExpr(null),
            _ => new VarExpr(ident),
        };
    }
}

internal sealed class ListLiteralExpr(List<Expr> items) : Expr
{
    public override object? Eval(JinjaEngine.Scope scope) => items.Select(e => e.Eval(scope)).ToList();
}

/// <summary>Jinja <c>is</c> tests: <c>defined</c>, <c>none</c>, <c>string</c>, <c>mapping</c>, <c>iterable</c>
/// (with optional <c>not</c>). <c>defined</c> uses scope membership so a missing variable reads as undefined.</summary>
internal sealed class IsTestExpr(Expr target, string test, bool negated) : Expr
{
    public override object? Eval(JinjaEngine.Scope scope)
    {
        bool result = test switch
        {
            "defined" => target is VarExpr v ? scope.Has(v.GetName()) : target.Eval(scope) is not null,
            "undefined" => target is VarExpr v2 ? !scope.Has(v2.GetName()) : target.Eval(scope) is null,
            "none" => target.Eval(scope) is null,
            "string" => target.Eval(scope) is string,
            "mapping" => target.Eval(scope) is Dictionary<string, object?>,
            "iterable" => target.Eval(scope) is System.Collections.IEnumerable,
            // Python/Jinja "sequence": ordered, indexable — lists and strings, but not dicts (mappings aren't
            // sequences even though they're iterable). Gemma-4's template checks content is sequence/is string.
            "sequence" => target.Eval(scope) is List<object?> or string,
            "number" => Values.IsNum(target.Eval(scope)),
            _ => throw new NotSupportedException($"Jinja 'is {test}' test"),
        };
        return negated ? !result : result;
    }
}

/// <summary>Tiny tokenizer for Jinja expression source (identifiers, strings, ints, symbols, keywords).</summary>
internal sealed class Tokenizer
{
    private readonly string _s;
    private int _p;

    public Tokenizer(string s) { _s = s; _p = 0; }

    private void SkipWs() { while (_p < _s.Length && char.IsWhiteSpace(_s[_p])) _p++; }

    public void ExpectEnd() { SkipWs(); if (_p < _s.Length) throw new FormatException($"Trailing input in Jinja expr at {_p}: '{_s[_p..]}'"); }

    public bool TrySymbol(string sym)
    {
        SkipWs();
        if (_p + sym.Length <= _s.Length && string.CompareOrdinal(_s, _p, sym, 0, sym.Length) == 0)
        {
            // Avoid matching '<' when the text is '<=' etc. — caller order handles multi-char first.
            _p += sym.Length;
            return true;
        }
        return false;
    }

    public void ExpectSymbol(string sym) { if (!TrySymbol(sym)) throw new FormatException($"Expected '{sym}' in Jinja expr near {_p}."); }

    public int Save() => _p;
    public void Restore(int p) => _p = p;

    public bool PeekSymbol(string sym)
    {
        int save = _p; SkipWs();
        bool hit = _p + sym.Length <= _s.Length && string.CompareOrdinal(_s, _p, sym, 0, sym.Length) == 0;
        _p = save; return hit;
    }

    public bool TryReadIdent(out string ident)
    {
        int save = _p; SkipWs();
        if (_p < _s.Length && (char.IsLetter(_s[_p]) || _s[_p] == '_'))
        {
            int start = _p; _p++;
            while (_p < _s.Length && IsIdentChar(_s[_p])) _p++;
            ident = _s[start.._p]; return true;
        }
        _p = save; ident = string.Empty; return false;
    }

    public bool TryKeyword(string kw)
    {
        SkipWs();
        int start = _p;
        if (_p + kw.Length <= _s.Length && string.CompareOrdinal(_s, _p, kw, 0, kw.Length) == 0)
        {
            int end = _p + kw.Length;
            if (end == _s.Length || !IsIdentChar(_s[end])) { _p = end; return true; }
        }
        _p = start;
        return false;
    }

    public void ExpectKeyword(string kw) { if (!TryKeyword(kw)) throw new FormatException($"Expected keyword '{kw}' in Jinja expr near {_p}."); }

    public bool TryString(out string value)
    {
        SkipWs();
        value = string.Empty;
        if (_p >= _s.Length || (_s[_p] != '\'' && _s[_p] != '"')) return false;
        char q = _s[_p++];
        System.Text.StringBuilder sb = new();
        while (_p < _s.Length && _s[_p] != q)
        {
            if (_s[_p] == '\\' && _p + 1 < _s.Length)
            {
                _p++;
                sb.Append(_s[_p] switch { 'n' => '\n', 't' => '\t', 'r' => '\r', '\\' => '\\', '\'' => '\'', '"' => '"', char o => o });
                _p++;
            }
            else sb.Append(_s[_p++]);
        }
        if (_p >= _s.Length) throw new FormatException("Unterminated string in Jinja expr.");
        _p++; // closing quote
        value = sb.ToString();
        return true;
    }

    public bool TryNumber(out long value)
    {
        SkipWs();
        value = 0;
        int start = _p;
        while (_p < _s.Length && char.IsDigit(_s[_p])) _p++;
        if (_p == start) return false;
        value = long.Parse(_s.AsSpan(start, _p - start), CultureInfo.InvariantCulture);
        return true;
    }

    public string ReadIdent()
    {
        SkipWs();
        int start = _p;
        if (_p < _s.Length && (char.IsLetter(_s[_p]) || _s[_p] == '_')) _p++;
        else throw new FormatException($"Expected identifier in Jinja expr near {_p}: '{_s[_p..]}'");
        while (_p < _s.Length && IsIdentChar(_s[_p])) _p++;
        return _s[start.._p];
    }

    private static bool IsIdentChar(char c) => char.IsLetterOrDigit(c) || c == '_';
}
