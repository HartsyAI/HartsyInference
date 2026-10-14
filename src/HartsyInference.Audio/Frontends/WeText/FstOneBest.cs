namespace HartsyInference.Audio.Frontends.WeText;

/// <summary>OpenFst 1.8's <c>Compose(linear acceptor, fst)</c> followed by <c>ShortestPath(composed, 1)</c>, the pair
/// kaldifst's <c>TextNormalizer::Normalize</c> runs. Equal-weight paths are not broken by any ordering of their own:
/// the winner is whichever path OpenFst's queue relaxes first, so this reproduces its state numbering and arc order
/// (sequence compose filter, epsilon and label matches in rule-arc order, <c>Connect</c>) and its <c>AutoQueue</c>
/// choice (topological order for an acyclic rule or composition, LIFO for an unweighted one).</summary>
internal static class FstOneBest
{
    /// <summary>Writes the output labels (epsilons dropped) of the one-best path through the composition of the
    /// linear acceptor over <paramref name="input"/> with <paramref name="fst"/>; false when no path exists.</summary>
    public static bool TryRun(StdVectorFst fst, ReadOnlySpan<int> input, List<int> output)
    {
        ArgumentNullException.ThrowIfNull(fst);
        ArgumentNullException.ThrowIfNull(output);
        output.Clear();
        foreach (int label in input)
            if (label <= 0) throw new ArgumentException("input labels must be positive; 0 is epsilon", nameof(input));
        if (fst.Start < 0) return false;

        Graph composed = Compose(fst, input);
        Graph? connected = Connect(composed);
        if (connected is null) return false;
        int[] parentArc = ShortestPathParents(connected, fst.IsAcyclic, out int finalState);
        if (finalState < 0) return false;

        int depth = 0;
        for (int s = finalState; parentArc[s] >= 0; s = connected.Source[parentArc[s]]) depth++;
        int[] path = new int[depth];
        for (int s = finalState, i = depth - 1; parentArc[s] >= 0; s = connected.Source[parentArc[s]], i--)
            path[i] = parentArc[s];
        foreach (int arc in path)
            if (connected.OLabel[arc] != 0) output.Add(connected.OLabel[arc]);
        return true;
    }

    // Composed states are (input position, rule state) pairs numbered in discovery order; the sequence filter never
    // leaves state 0 because the acceptor has no epsilons, so rule epsilon-input arcs advance alone.
    private static Graph Compose(StdVectorFst fst, ReadOnlySpan<int> input)
    {
        Dictionary<long, int> ids = [];
        List<int> position = [];
        List<int> ruleState = [];
        List<int> arcStart = [];
        List<int> olabels = [];
        List<float> weights = [];
        List<int> targets = [];
        List<float> finals = [];
        FindState(ids, position, ruleState, 0, fst.Start);
        for (int k = 0; k < position.Count; k++)
        {
            int at = position[k];
            int label = at < input.Length ? input[at] : -1;
            arcStart.Add(olabels.Count);
            foreach (StdArc arc in fst.Arcs(ruleState[k]))
            {
                int next;
                if (arc.ILabel == 0) next = FindState(ids, position, ruleState, at, arc.NextState);
                else if (arc.ILabel == label) next = FindState(ids, position, ruleState, at + 1, arc.NextState);
                else continue;
                olabels.Add(arc.OLabel);
                weights.Add(arc.Weight);
                targets.Add(next);
            }
            finals.Add(at == input.Length ? fst.Final(ruleState[k]) : float.PositiveInfinity);
        }
        arcStart.Add(olabels.Count);
        return new Graph(0, [.. arcStart], [.. olabels], [.. weights], [.. targets], [.. finals]);
    }

    private static int FindState(Dictionary<long, int> ids, List<int> position, List<int> ruleState, int at, int state)
    {
        long key = ((long)at << 32) | (uint)state;
        if (ids.TryGetValue(key, out int id)) return id;
        id = position.Count;
        ids.Add(key, id);
        position.Add(at);
        ruleState.Add(state);
        return id;
    }

    // OpenFst Connect: drop states that cannot reach a final state, renumbering the rest in order.
    private static Graph? Connect(Graph graph)
    {
        Scc(graph, out _, out bool[] coaccess, out _);
        if (!coaccess[graph.Start]) return null;
        int[] newId = new int[graph.StateCount];
        int kept = 0;
        for (int s = 0; s < graph.StateCount; s++) newId[s] = coaccess[s] ? kept++ : -1;
        int[] arcStart = new int[kept + 1];
        List<int> olabels = new(graph.OLabel.Length);
        List<float> weights = new(graph.OLabel.Length);
        List<int> targets = new(graph.OLabel.Length);
        float[] finals = new float[kept];
        for (int s = 0; s < graph.StateCount; s++)
        {
            if (newId[s] < 0) continue;
            arcStart[newId[s]] = olabels.Count;
            finals[newId[s]] = graph.Final[s];
            for (int a = graph.ArcStart[s]; a < graph.ArcStart[s + 1]; a++)
            {
                int t = newId[graph.Target[a]];
                if (t < 0) continue;
                olabels.Add(graph.OLabel[a]);
                weights.Add(graph.Weight[a]);
                targets.Add(t);
            }
        }
        arcStart[kept] = olabels.Count;
        return new Graph(newId[graph.Start], arcStart, [.. olabels], [.. weights], [.. targets], finals);
    }

    // OpenFst SingleShortestPath under the AutoQueue it would pick; returns each state's best incoming arc.
    private static int[] ShortestPathParents(Graph graph, bool ruleAcyclic, out int finalState)
    {
        Scc(graph, out int[] scc, out _, out bool acyclic);
        bool lifo = false;
        if (!ruleAcyclic)
        {
            bool unweighted = true;
            foreach (float w in graph.Weight)
                if (w != 0f && w != float.PositiveInfinity) unweighted = false;
            lifo = unweighted;
            if (!lifo && !acyclic)
                throw new NotSupportedException("the composition has a weighted cycle; OpenFst's SCC queue is not ported");
        }

        int n = graph.StateCount;
        float[] distance = new float[n];
        Array.Fill(distance, float.PositiveInfinity);
        int[] parentArc = new int[n];
        Array.Fill(parentArc, -1);
        bool[] enqueued = new bool[n];
        Stack<int> stack = new();
        int[] slot = new int[n];
        Array.Fill(slot, -1);
        int front = 0, back = -1;

        distance[graph.Start] = 0f;
        Enqueue(graph.Start);
        enqueued[graph.Start] = true;
        float finalDistance = float.PositiveInfinity;
        finalState = -1;
        while (lifo ? stack.Count > 0 : front <= back)
        {
            int s = lifo ? stack.Pop() : DequeueTopOrder();
            enqueued[s] = false;
            float sd = distance[s];
            float final = graph.Final[s];
            if (final != float.PositiveInfinity)
            {
                float plus = Plus(finalDistance, sd + final);
                if (finalDistance != plus)
                {
                    finalDistance = plus;
                    finalState = s;
                }
                if (!Member(finalDistance)) throw new InvalidOperationException("shortest path hit a non-member weight");
            }
            for (int a = graph.ArcStart[s]; a < graph.ArcStart[s + 1]; a++)
            {
                int t = graph.Target[a];
                float plus = Plus(distance[t], sd + graph.Weight[a]);
                if (distance[t] == plus) continue;
                distance[t] = plus;
                if (!Member(plus)) throw new InvalidOperationException("shortest path hit a non-member weight");
                parentArc[t] = a;
                if (enqueued[t]) continue;
                Enqueue(t);
                enqueued[t] = true;
            }
        }
        return parentArc;

        // OpenFst TopOrderQueue keyed by SCC number, which is a topological order when every SCC is trivial.
        void Enqueue(int s)
        {
            if (lifo)
            {
                stack.Push(s);
                return;
            }
            int order = scc[s];
            if (front > back) front = back = order;
            else if (order > back) back = order;
            else if (order < front) front = order;
            slot[order] = s;
        }

        int DequeueTopOrder()
        {
            int s = slot[front];
            slot[front] = -1;
            while (front <= back && slot[front] == -1) front++;
            return s;
        }
    }

    private static float Plus(float a, float b) => !Member(a) || !Member(b) ? float.NaN : a < b ? a : b;

    private static bool Member(float w) => !float.IsNaN(w) && w != float.NegativeInfinity;

    // OpenFst DfsVisit with SccVisitor: Tarjan over arcs in order from the start state, then any unvisited state;
    // SCC numbers are reversed at the end so they run in topological order.
    private static void Scc(Graph graph, out int[] scc, out bool[] coaccess, out bool acyclic)
    {
        int n = graph.StateCount;
        int[] component = new int[n];
        bool[] reach = new bool[n];
        bool noCycle = true;
        int[] dfn = new int[n];
        int[] low = new int[n];
        bool[] onStack = new bool[n];
        byte[] color = new byte[n];
        List<int> sccStack = [];
        Stack<(int State, int Arc)> dfs = new();
        int visited = 0, components = 0;
        for (int root = graph.Start, r = 0; root >= 0;)
        {
            Discover(root);
            while (dfs.Count > 0)
            {
                (int s, int a) = dfs.Peek();
                if (a == graph.ArcStart[s + 1])
                {
                    color[s] = 2;
                    dfs.Pop();
                    Finish(s);
                    if (dfs.Count > 0)
                    {
                        (int p, int pa) = dfs.Pop();
                        if (reach[s]) reach[p] = true;
                        if (low[s] < low[p]) low[p] = low[s];
                        dfs.Push((p, pa + 1));
                    }
                    continue;
                }
                int t = graph.Target[a];
                if (color[t] == 0)
                {
                    Discover(t);
                    continue;
                }
                if (color[t] == 1)
                {
                    noCycle = false;
                    if (dfn[t] < low[s]) low[s] = dfn[t];
                }
                else if (dfn[t] < dfn[s] && onStack[t] && dfn[t] < low[s])
                {
                    low[s] = dfn[t];
                }
                if (reach[t]) reach[s] = true;
                dfs.Pop();
                dfs.Push((s, a + 1));
            }
            while (r < n && color[r] != 0) r++;
            root = r < n ? r : -1;
        }
        for (int s = 0; s < n; s++) component[s] = components - 1 - component[s];
        scc = component;
        coaccess = reach;
        acyclic = noCycle;

        void Discover(int s)
        {
            color[s] = 1;
            dfn[s] = low[s] = visited++;
            onStack[s] = true;
            sccStack.Add(s);
            dfs.Push((s, graph.ArcStart[s]));
        }

        void Finish(int s)
        {
            if (graph.Final[s] != float.PositiveInfinity) reach[s] = true;
            if (dfn[s] != low[s]) return;
            bool reachesFinal = false;
            for (int i = sccStack.Count - 1; ; i--)
            {
                if (reach[sccStack[i]]) reachesFinal = true;
                if (sccStack[i] == s) break;
            }
            int t;
            do
            {
                t = sccStack[^1];
                sccStack.RemoveAt(sccStack.Count - 1);
                component[t] = components;
                if (reachesFinal) reach[t] = true;
                onStack[t] = false;
            }
            while (t != s);
            components++;
        }
    }

    private sealed class Graph
    {
        public Graph(int start, int[] arcStart, int[] olabel, float[] weight, int[] target, float[] final)
        {
            Start = start;
            ArcStart = arcStart;
            OLabel = olabel;
            Weight = weight;
            Target = target;
            Final = final;
            Source = new int[target.Length];
            for (int s = 0; s < final.Length; s++)
                for (int a = arcStart[s]; a < arcStart[s + 1]; a++) Source[a] = s;
        }

        public int Start { get; }
        public int[] ArcStart { get; }
        public int[] OLabel { get; }
        public float[] Weight { get; }
        public int[] Target { get; }
        public int[] Source { get; }
        public float[] Final { get; }
        public int StateCount => Final.Length;
    }
}
