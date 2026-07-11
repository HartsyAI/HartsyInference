using System.Globalization;
using System.Text;

namespace HartsyInference.LLM.ChatTemplates;

/// <summary>A minimal Jinja2 interpreter covering the subset HuggingFace chat templates use: <c>{{ expr }}</c>,
/// <c>{% if/elif/else/endif %}</c>, <c>{% for x in seq %}/{% endfor %}</c>, <c>{% set %}</c>, whitespace control
/// (<c>{%- -%}</c> / <c>{{- -}}</c>), member/index access, the <c>loop</c> object, <c>+</c>/<c>~</c> concat,
/// comparisons + <c>and/or/not/in</c>, ternary <c>a if c else b</c>, common filters (<c>trim/default/length/
/// lower/upper</c>) and string methods (<c>strip/startswith/endswith/replace/split</c>). Not a full Jinja — it
/// targets Llama-3/Qwen/Mistral/Phi/Gemma/ChatML/DeepSeek chat templates; unsupported constructs throw.</summary>
public sealed class JinjaEngine
{
    private readonly List<Node> _program;

    public JinjaEngine(string template) => _program = Parser.Parse(Lexer.Lex(template));

    /// <summary>Renders the template against <paramref name="context"/> (variable name → value; values are
    /// string / bool / long / double / List&lt;object?&gt; / Dictionary&lt;string,object?&gt; / null).</summary>
    public string Render(Dictionary<string, object?> context)
    {
        StringBuilder sb = new();
        Scope scope = new(context);
        foreach (Node n in _program) n.Render(sb, scope);
        return sb.ToString();
    }

    // ── Runtime scope ────────────────────────────────────────────────────────────────────────────────
    internal sealed class Scope
    {
        private readonly Dictionary<string, object?> _vars;
        public Scope(Dictionary<string, object?> vars) => _vars = vars;
        public Scope Child() => new(new Dictionary<string, object?>(_vars));
        public object? Get(string name) => _vars.TryGetValue(name, out object? v) ? v : null;
        public bool Has(string name) => _vars.ContainsKey(name);
        public void Set(string name, object? value) => _vars[name] = value;
    }

    // ── Lexer ────────────────────────────────────────────────────────────────────────────────────────
    internal enum SegKind { Text, Expr, Stmt }
    internal readonly record struct Seg(SegKind Kind, string Value, bool TrimLeft, bool TrimRight);

    internal static class Lexer
    {
        public static List<Seg> Lex(string t)
        {
            List<Seg> segs = [];
            int i = 0;
            while (i < t.Length)
            {
                int open = t.IndexOf('{', i);
                if (open < 0 || open + 1 >= t.Length) { segs.Add(new Seg(SegKind.Text, t[i..], false, false)); break; }
                char c = t[open + 1];
                if (c == '#') // comment {# ... #} — strip entirely (honoring whitespace-control)
                {
                    if (open > i) segs.Add(new Seg(SegKind.Text, t[i..open], false, false));
                    int cc = FindClose(t, open + 2, "#}");
                    if (cc < 0) throw new FormatException($"Unclosed Jinja comment at {open}.");
                    string cInner = t[(open + 2)..cc];
                    // Emit a no-op stmt segment carrying only whitespace-control so adjacent text trims apply.
                    segs.Add(new Seg(SegKind.Stmt, "#comment", cInner.StartsWith('-'), cInner.EndsWith('-')));
                    i = cc + 2; continue;
                }
                if (c != '{' && c != '%') { // not a tag; emit text up to and including this brace, continue
                    int next = t.IndexOf('{', open + 1);
                    if (next < 0) { segs.Add(new Seg(SegKind.Text, t[i..], false, false)); break; }
                    segs.Add(new Seg(SegKind.Text, t[i..next], false, false));
                    i = next; continue;
                }
                if (open > i) segs.Add(new Seg(SegKind.Text, t[i..open], false, false));
                string closeTok = c == '{' ? "}}" : "%}";
                int close = FindClose(t, open + 2, closeTok);
                if (close < 0) throw new FormatException($"Unclosed Jinja tag at {open}.");
                string inner = t[(open + 2)..close];
                bool trimL = inner.StartsWith('-');
                bool trimR = inner.EndsWith('-');
                inner = inner.Trim('-').Trim();
                segs.Add(new Seg(c == '{' ? SegKind.Expr : SegKind.Stmt, inner, trimL, trimR));
                i = close + 2;
            }
            // Apply whitespace control: trim adjacent text.
            for (int k = 0; k < segs.Count; k++)
            {
                if (segs[k].Kind == SegKind.Text) continue;
                if (segs[k].TrimLeft && k > 0 && segs[k - 1].Kind == SegKind.Text)
                    segs[k - 1] = segs[k - 1] with { Value = segs[k - 1].Value.TrimEnd() };
                if (segs[k].TrimRight && k + 1 < segs.Count && segs[k + 1].Kind == SegKind.Text)
                    segs[k + 1] = segs[k + 1] with { Value = segs[k + 1].Value.TrimStart() };
            }
            return segs;
        }

        /// <summary>Finds the block-closing token (<c>}}</c>, <c>%}</c> or <c>#}</c>) starting at <paramref name="start"/>,
        /// skipping over any single- or double-quoted string literals so a close token embedded inside a string
        /// (e.g. Qwen's tool-call template, which contains <c>}}</c> inside a quoted example) is not mistaken for
        /// the real block end. Honors backslash escapes inside strings. Returns the index of the token, or -1.</summary>
        private static int FindClose(string t, int start, string closeTok)
        {
            char quote = '\0';
            for (int p = start; p < t.Length; p++)
            {
                char ch = t[p];
                if (quote != '\0') // inside a string literal
                {
                    if (ch == '\\') { p++; continue; } // skip escaped char
                    if (ch == quote) quote = '\0';
                    continue;
                }
                if (ch == '\'' || ch == '"') { quote = ch; continue; }
                if (ch == closeTok[0] && p + 1 < t.Length && t[p + 1] == closeTok[1]) return p;
            }
            return -1;
        }
    }

    // ── AST nodes ────────────────────────────────────────────────────────────────────────────────────
    internal abstract class Node { public abstract void Render(StringBuilder sb, Scope scope); }

    internal sealed class TextNode(string text) : Node
    {
        public override void Render(StringBuilder sb, Scope scope) => sb.Append(text);
    }

    internal sealed class ExprNode(Expr expr) : Node
    {
        public override void Render(StringBuilder sb, Scope scope) => sb.Append(Values.ToStr(expr.Eval(scope)));
    }

    internal sealed class SetNode(string name, Expr expr) : Node
    {
        public override void Render(StringBuilder sb, Scope scope) => AssignSet(name, expr.Eval(scope), scope);
    }

    /// <summary>Block-capture <c>{% set name %}...{% endset %}</c>: renders the body to a string and assigns it
    /// (Gemma-4's chat template captures a message's rendered content this way before post-processing it).</summary>
    internal sealed class SetBlockNode(string name, List<Node> body) : Node
    {
        public override void Render(StringBuilder sb, Scope scope)
        {
            StringBuilder inner = new();
            foreach (Node n in body) n.Render(inner, scope);
            AssignSet(name, inner.ToString(), scope);
        }
    }

    /// <summary>Shared <c>{% set %}</c> assignment target resolution: a plain variable, or <c>ns.attr</c>
    /// (namespace attribute assignment — <c>ns</c> is a mutable dict already in scope).</summary>
    internal static void AssignSet(string name, object? value, Scope scope)
    {
        int dot = name.IndexOf('.');
        if (dot < 0) { scope.Set(name, value); return; }
        string baseName = name[..dot];
        string attr = name[(dot + 1)..];
        if (scope.Get(baseName) is Dictionary<string, object?> ns) ns[attr] = value;
        else throw new InvalidOperationException($"Jinja set: '{baseName}' is not a namespace/object.");
    }

    internal sealed class IfNode(List<(Expr? Cond, List<Node> Body)> branches) : Node
    {
        public override void Render(StringBuilder sb, Scope scope)
        {
            foreach ((Expr? cond, List<Node> body) in branches)
                if (cond is null || Values.Truthy(cond.Eval(scope)))
                {
                    foreach (Node n in body) n.Render(sb, scope);
                    return;
                }
        }
    }

    /// <summary>A <c>{% for x in seq %}</c> loop, optionally with Jinja's inline filter clause
    /// (<c>{% for x in seq if cond %}</c> — only iterates items where <paramref name="filter"/> is truthy,
    /// evaluated with <paramref name="var"/> already bound). <c>loop.index</c>/<c>loop.length</c>/etc. reflect
    /// the FILTERED count, matching real Jinja (the filter narrows the sequence before iteration starts, it
    /// isn't a per-item skip inside a still-full-length loop).</summary>
    internal sealed class ForNode(string var, Expr seq, List<Node> body, Expr? filter = null) : Node
    {
        public override void Render(StringBuilder sb, Scope scope)
        {
            List<object?> raw = Values.AsList(seq.Eval(scope));
            List<object?> items = raw;
            if (filter is not null)
            {
                items = [];
                foreach (object? item in raw)
                {
                    Scope probe = scope.Child();
                    probe.Set(var, item);
                    if (Values.Truthy(filter.Eval(probe))) items.Add(item);
                }
            }
            for (int idx = 0; idx < items.Count; idx++)
            {
                Scope child = scope.Child();
                child.Set(var, items[idx]);
                child.Set("loop", new Dictionary<string, object?>
                {
                    ["index0"] = (long)idx, ["index"] = (long)(idx + 1),
                    ["first"] = idx == 0, ["last"] = idx == items.Count - 1, ["length"] = (long)items.Count,
                });
                try { foreach (Node n in body) n.Render(sb, child); }
                catch (LoopContinue) { continue; }
                catch (LoopBreak) { break; }
            }
        }
    }

    /// <summary>Signals for Jinja <c>{% break %}</c> / <c>{% continue %}</c>, caught by the enclosing <see cref="ForNode"/>.</summary>
    internal sealed class LoopBreak : Exception { }
    internal sealed class LoopContinue : Exception { }

    internal sealed class BreakNode : Node { public override void Render(StringBuilder sb, Scope scope) => throw new LoopBreak(); }
    internal sealed class ContinueNode : Node { public override void Render(StringBuilder sb, Scope scope) => throw new LoopContinue(); }

    // ── Parser (segments → nodes) ────────────────────────────────────────────────────────────────────
    internal static class Parser
    {
        public static List<Node> Parse(List<Seg> segs) { int i = 0; return ParseBlock(segs, ref i, null); }

        private static List<Node> ParseBlock(List<Seg> segs, ref int i, string[]? stops)
        {
            List<Node> nodes = [];
            while (i < segs.Count)
            {
                Seg s = segs[i];
                if (s.Kind == SegKind.Text) { nodes.Add(new TextNode(s.Value)); i++; continue; }
                if (s.Kind == SegKind.Expr) { nodes.Add(new ExprNode(ExprParser.ParseExpr(s.Value))); i++; continue; }

                string kw = FirstWord(s.Value);
                if (stops is not null && Array.IndexOf(stops, kw) >= 0) return nodes;

                switch (kw)
                {
                    case "set":
                    {
                        string rest = s.Value["set".Length..].Trim();
                        int eq = rest.IndexOf('=');
                        if (eq < 0)
                        {
                            // Block-capture form: {% set name %}...{% endset %} — rest is just the target name.
                            i++;
                            List<Node> setBody = ParseBlock(segs, ref i, ["endset"]);
                            Expect(segs, ref i, "endset");
                            nodes.Add(new SetBlockNode(rest, setBody));
                            break;
                        }
                        nodes.Add(new SetNode(rest[..eq].Trim(), ExprParser.ParseExpr(rest[(eq + 1)..].Trim())));
                        i++; break;
                    }
                    case "if":
                    {
                        i++;
                        List<(Expr?, List<Node>)> branches = [];
                        branches.Add((ExprParser.ParseExpr(s.Value["if".Length..].Trim()), ParseBlock(segs, ref i, ["elif", "else", "endif"])));
                        while (i < segs.Count && FirstWord(segs[i].Value) == "elif")
                        {
                            Seg e = segs[i]; i++;
                            branches.Add((ExprParser.ParseExpr(e.Value["elif".Length..].Trim()), ParseBlock(segs, ref i, ["elif", "else", "endif"])));
                        }
                        if (i < segs.Count && FirstWord(segs[i].Value) == "else")
                        {
                            i++; branches.Add((null, ParseBlock(segs, ref i, ["endif"])));
                        }
                        Expect(segs, ref i, "endif");
                        nodes.Add(new IfNode(branches)); break;
                    }
                    case "for":
                    {
                        string rest = s.Value["for".Length..].Trim();
                        int inPos = FindKeyword(rest, "in");
                        if (inPos < 0) throw new FormatException($"Malformed for: {s.Value}");
                        string loopVar = rest[..inPos].Trim();
                        // Jinja's for-tag grammar allows a trailing inline filter (`for x in seq if cond`),
                        // parsed separately from a ternary so `for x in a if c else b` isn't ambiguous —
                        // ParseForIterable stops at the `or_expr` level (no ternary) for the sequence itself,
                        // matching real Jinja, then optionally consumes ` if cond`.
                        (Expr seqExpr, Expr? filterExpr) = ExprParser.ParseForIterable(rest[(inPos + 2)..].Trim());
                        i++;
                        List<Node> body = ParseBlock(segs, ref i, ["endfor"]);
                        Expect(segs, ref i, "endfor");
                        nodes.Add(new ForNode(loopVar, seqExpr, body, filterExpr)); break;
                    }
                    case "generation": case "endgeneration": // {% generation %} markers (assistant-mask) — ignore
                    case "#comment": // stripped comment placeholder
                        i++; break;
                    case "break": nodes.Add(new BreakNode()); i++; break;
                    case "continue": nodes.Add(new ContinueNode()); i++; break;
                    case "macro":
                    {
                        // Macro definitions (e.g. Cohere's RAG/tool document_turn) are skipped wholesale — the
                        // basic chat path never invokes them. Skip the body without parsing it (it may use
                        // constructs only meaningful inside a macro), honoring nesting. Full macro support TODO.
                        i++;
                        int depth = 1;
                        while (i < segs.Count && depth > 0)
                        {
                            if (segs[i].Kind == SegKind.Stmt)
                            {
                                string w = FirstWord(segs[i].Value);
                                if (w == "macro") depth++;
                                else if (w == "endmacro") depth--;
                            }
                            i++;
                        }
                        break;
                    }
                    default:
                        throw new NotSupportedException($"Unsupported Jinja statement '{kw}' in: {s.Value}");
                }
            }
            return nodes;
        }

        private static void Expect(List<Seg> segs, ref int i, string kw)
        {
            if (i >= segs.Count || FirstWord(segs[i].Value) != kw) throw new FormatException($"Expected {{% {kw} %}}.");
            i++;
        }

        private static string FirstWord(string s)
        {
            int sp = s.IndexOf(' ');
            return sp < 0 ? s.Trim() : s[..sp].Trim();
        }

        // Find a top-level keyword (not inside quotes/brackets) — for the for-loop "in".
        private static int FindKeyword(string s, string kw)
        {
            int depth = 0; char quote = '\0';
            for (int p = 0; p + kw.Length <= s.Length; p++)
            {
                char c = s[p];
                if (quote != '\0') { if (c == quote) quote = '\0'; continue; }
                if (c is '\'' or '"') { quote = c; continue; }
                if (c is '(' or '[') depth++;
                else if (c is ')' or ']') depth--;
                else if (depth == 0 && (p == 0 || !char.IsLetterOrDigit(s[p - 1])) &&
                         string.CompareOrdinal(s, p, kw, 0, kw.Length) == 0 &&
                         (p + kw.Length == s.Length || !char.IsLetterOrDigit(s[p + kw.Length])))
                    return p;
            }
            return -1;
        }
    }
}
