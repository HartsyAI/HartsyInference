using System.Globalization;
using System.Text;

namespace HartsyInference.LLM.ChatTemplates;

/// <summary>Minimal Jinja2 interpreter covering only the subset HuggingFace chat templates use (if/for/set, whitespace control, loop object, comparisons, ternary, common filters/string methods) targeting Llama-3/Qwen/Mistral/Phi/Gemma/ChatML/DeepSeek; unsupported constructs throw rather than silently misrendering.</summary>
public sealed class JinjaEngine
{
    private readonly List<Node> _program;

    public JinjaEngine(string template) => _program = Parser.Parse(Lexer.Lex(template));

    /// <summary>Renders the template against <paramref name="context"/> (variable name → value; values are string/bool/long/double/List&lt;object?&gt;/Dictionary&lt;string,object?&gt;/null).</summary>
    public string Render(Dictionary<string, object?> context)
    {
        StringBuilder sb = new();
        Scope scope = new(context);
        foreach (Node n in _program) n.Render(sb, scope);
        return sb.ToString();
    }

    /// <summary>True when an executable expression in the template compares a message role to <paramref name="role"/>
    /// (<c>x.role == 'r'</c>, <c>x['role'] != 'r'</c>, or <c>x.role in ['r', ...]</c>). Comments and text never become
    /// expressions and a string literal is a single leaf, so a mention in either cannot match. A role copied into another
    /// variable first (<c>{% set r = m.role %}</c>) is not followed; that misses toward keeping the default role.</summary>
    public bool ComparesRoleTo(string role)
    {
        Stack<Node> nodes = new(_program);
        while (nodes.Count > 0)
        {
            Node node = nodes.Pop();
            foreach (Node child in node.Children) nodes.Push(child);
            foreach (Expr expr in node.Exprs)
                if (ExprComparesRole(expr, role)) return true;
        }
        return false;
    }

    private static bool ExprComparesRole(Expr root, string role)
    {
        Stack<Expr> pending = new();
        pending.Push(root);
        while (pending.Count > 0)
        {
            Expr e = pending.Pop();
            if (e is BinaryExpr b)
            {
                if ((b.Op is "==" or "!=") && ((IsRoleAccess(b.Left) && IsLiteral(b.Right, role)) || (IsRoleAccess(b.Right) && IsLiteral(b.Left, role))))
                    return true;
                if ((b.Op is "in" or "notin") && IsRoleAccess(b.Left) && b.Right is ListLiteralExpr list && list.Items.Any(i => IsLiteral(i, role)))
                    return true;
            }
            foreach (Expr child in e.Children) pending.Push(child);
        }
        return false;
    }

    private static bool IsLiteral(Expr e, string value) => e is LiteralExpr { Value: string s } && s == value;

    private static bool IsRoleAccess(Expr e) => e switch
    {
        MemberExpr m => m.Name == "role",
        IndexExpr i => IsLiteral(i.Index, "role"),
        VarExpr v => v.GetName() == "role",
        _ => false,
    };

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

        /// <summary>Finds the block-closing token, skipping over quoted string literals (honoring backslash escapes) so a close token embedded inside a string — e.g. Qwen's tool-call template contains <c>}}</c> inside a quoted example — isn't mistaken for the real block end.</summary>
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
    internal abstract class Node
    {
        public abstract void Render(StringBuilder sb, Scope scope);
        /// <summary>Expressions this node evaluates itself, for static inspection.</summary>
        public virtual IEnumerable<Expr> Exprs => [];
        /// <summary>Nested statement bodies, for static inspection.</summary>
        public virtual IEnumerable<Node> Children => [];
    }

    internal sealed class TextNode(string text) : Node
    {
        public override void Render(StringBuilder sb, Scope scope) => sb.Append(text);
    }

    internal sealed class ExprNode(Expr expr) : Node
    {
        public override IEnumerable<Expr> Exprs => [expr];
        public override void Render(StringBuilder sb, Scope scope) => sb.Append(Values.ToStr(expr.Eval(scope)));
    }

    internal sealed class SetNode(string name, Expr expr) : Node
    {
        public override IEnumerable<Expr> Exprs => [expr];
        public override void Render(StringBuilder sb, Scope scope) => AssignSet(name, expr.Eval(scope), scope);
    }

    /// <summary>Block-capture <c>{% set name %}...{% endset %}</c>: renders the body to a string and assigns it (Gemma-4's chat template captures a message's rendered content this way before post-processing it).</summary>
    internal sealed class SetBlockNode(string name, List<Node> body) : Node
    {
        public override IEnumerable<Node> Children => body;
        public override void Render(StringBuilder sb, Scope scope)
        {
            StringBuilder inner = new();
            foreach (Node n in body) n.Render(inner, scope);
            AssignSet(name, inner.ToString(), scope);
        }
    }

    /// <summary>Shared <c>{% set %}</c> assignment target resolution: a plain variable, or <c>ns.attr</c> (namespace attribute assignment — <c>ns</c> is a mutable dict already in scope).</summary>
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
        public override IEnumerable<Expr> Exprs => branches.Select(b => b.Cond).OfType<Expr>();
        public override IEnumerable<Node> Children => branches.SelectMany(b => b.Body);
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

    /// <summary>A <c>{% for x in seq %}</c> loop, optionally with Jinja's inline filter clause (<c>{% for x in seq if cond %}</c>); <c>loop.index</c>/<c>loop.length</c>/etc. reflect the FILTERED count, matching real Jinja — the filter narrows the sequence before iteration starts, it isn't a per-item skip inside a still-full-length loop.</summary>
    internal sealed class ForNode(string var, Expr seq, List<Node> body, Expr? filter = null) : Node
    {
        public override IEnumerable<Expr> Exprs => filter is null ? [seq] : [seq, filter];
        public override IEnumerable<Node> Children => body;
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
                    ["revindex0"] = (long)(items.Count - 1 - idx), ["revindex"] = (long)(items.Count - idx),
                    // Neighbouring items (Jinja's undefined, here null, at the ends): Qwen3.5 opens one user turn for a run of tool results with it.
                    ["previtem"] = idx > 0 ? items[idx - 1] : null, ["nextitem"] = idx < items.Count - 1 ? items[idx + 1] : null,
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

    /// <summary>A template-defined macro: its parameters (with optional default expressions) and body. Stored in scope under its name by <see cref="MacroNode"/>.</summary>
    internal sealed class MacroDef(string name, List<(string Name, Expr? Default)> parameters, List<Node> body)
    {
        public string Name => name;
        public List<Node> Body => body;
        public IEnumerable<Expr> DefaultExprs => parameters.Where(p => p.Default is not null).Select(p => p.Default!);

        /// <summary>Renders the macro body for one call. Parameters bind by position, then by keyword; a parameter left unset takes its default (evaluated in the
        /// macro's scope, so it may use earlier parameters) or Jinja's undefined (null). The body sees everything defined before the call, and its own
        /// <c>set</c>s stay inside the call.</summary>
        public string Invoke(List<object?> positional, Dictionary<string, object?> named, Scope callerScope)
        {
            Scope local = callerScope.Child();
            for (int i = 0; i < parameters.Count; i++)
            {
                (string pName, Expr? pDefault) = parameters[i];
                object? value;
                if (i < positional.Count) value = positional[i];
                else if (named.TryGetValue(pName, out object? kw)) value = kw;
                else value = pDefault?.Eval(local);
                local.Set(pName, value);
            }
            foreach (string key in named.Keys)
                if (!parameters.Any(p => p.Name == key))
                    throw new NotSupportedException($"Jinja macro '{name}' has no parameter '{key}'.");
            if (positional.Count > parameters.Count)
                throw new NotSupportedException($"Jinja macro '{name}' takes {parameters.Count} argument(s), got {positional.Count}.");
            StringBuilder sb = new();
            foreach (Node n in body) n.Render(sb, local);
            return sb.ToString();
        }
    }

    /// <summary><c>{% macro %}</c>: rendering the definition binds the macro in scope and emits nothing.</summary>
    internal sealed class MacroNode(MacroDef def) : Node
    {
        public override IEnumerable<Expr> Exprs => def.DefaultExprs;
        public override IEnumerable<Node> Children => def.Body;
        public override void Render(StringBuilder sb, Scope scope) => scope.Set(def.Name, def);
    }
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
                        // {% macro name(a, b, c=default) %}...{% endmacro %}: the header parses as a call expression (positional names, name=default
                        // for the optional ones), the body as an ordinary block. Executing the node binds the macro in scope; a call renders the body
                        // in a child scope (see MacroDef.Invoke). Varargs, kwargs and call blocks are not supported.
                        string header = s.Value["macro".Length..].Trim();
                        if (ExprParser.ParseExpr(header) is not CallExpr { Callee: VarExpr macroName } headerCall)
                            throw new FormatException($"Malformed macro: {s.Value}");
                        List<(string Name, Expr? Default)> parameters = [];
                        foreach (Expr arg in headerCall.Args)
                        {
                            if (arg is NamedArgExpr named) parameters.Add((named.Name, named.Value));
                            else if (arg is VarExpr bare) parameters.Add((bare.GetName(), null));
                            else throw new FormatException($"Malformed macro parameter in: {s.Value}");
                        }
                        i++;
                        List<Node> macroBody = ParseBlock(segs, ref i, ["endmacro"]);
                        Expect(segs, ref i, "endmacro");
                        nodes.Add(new MacroNode(new MacroDef(macroName.GetName(), parameters, macroBody))); break;
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
