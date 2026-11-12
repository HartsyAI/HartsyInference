using HartsyInference.Core.Backends;
using HartsyInference.Core.Moe;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Transformer;

/// <summary>Expert-parallel routed MoE over tensor-parallel ranks, for latent MoE blocks. The routed bank is split into contiguous ranges,
/// one per rank (<see cref="MoeFeedForward.ShardExperts"/>), while the router and latent projections are whole on every rank. Each rank
/// sources a contiguous slice of the tokens: it routes them, sends each latent row to the rank holding its expert, and gets the expert
/// outputs back for a weighted combine in latent space. The routed output lands in a per-rank buffer that is zero outside the rank's own
/// slice, so the caller adds any other per-rank partial and sums everything with one AllReduce.</summary>
/// <remarks>The send and receive counts come from the routes on the host. The virtual-rank transport holds every rank's routes, so the
/// counts are exact there; a multi-process deployment exchanges the counts before the payloads.</remarks>
public sealed class ExpertParallelMoe
{
    private readonly IReadOnlyList<IBackend> _backends;
    private readonly IReadOnlyList<MoeFeedForward> _blocks;
    private readonly ICollectiveComm? _comm;
    private readonly int _experts;
    private readonly int _topK;

    /// <param name="rankBackends">One backend per rank; rank r runs its router, latent projections and experts on entry r.</param>
    /// <param name="blocks">One block per rank, sharded to that rank's experts with <see cref="MoeFeedForward.ShardExperts"/>.</param>
    /// <param name="comm">The communicator over every rank; null only for a single rank.</param>
    /// <exception cref="ArgumentException">The experts do not split evenly, or the blocks do not hold the ranges their ranks own.</exception>
    public ExpertParallelMoe(IReadOnlyList<IBackend> rankBackends, IReadOnlyList<MoeFeedForward> blocks, ICollectiveComm? comm, int experts, int topK)
    {
        ArgumentNullException.ThrowIfNull(rankBackends);
        ArgumentNullException.ThrowIfNull(blocks);
        int ranks = rankBackends.Count;
        if (ranks < 1) throw new ArgumentException("Expert parallelism needs at least one rank.", nameof(rankBackends));
        if (blocks.Count != ranks) throw new ArgumentException($"Expected one MoE block per rank ({ranks}), got {blocks.Count}.", nameof(blocks));
        if (ranks > 1 && (comm is null || comm.RankCount != ranks))
            throw new ArgumentException($"Expert parallelism over {ranks} ranks needs a communicator spanning every rank.", nameof(comm));
        int perRank = ExpertParallelDispatch.ExpertsPerRank(experts, ranks);
        if (topK < 1 || topK > experts) throw new ArgumentOutOfRangeException(nameof(topK), topK, $"Top-k must be in [1, {experts}].");
        for (int r = 0; r < ranks; r++)
        {
            MoeFeedForward block = blocks[r] ?? throw new ArgumentNullException(nameof(blocks));
            if (!block.HasLatent) throw new ArgumentException($"Rank {r}'s block has no latent projection; expert parallelism runs latent blocks.", nameof(blocks));
            if (block.FirstExpert != r * perRank || block.LocalExperts != perRank)
                throw new ArgumentException($"Rank {r} must hold experts [{r * perRank}, {(r + 1) * perRank}), got [{block.FirstExpert}, {block.FirstExpert + block.LocalExperts}).", nameof(blocks));
            if (block.ExpertStreamWidth != blocks[0].ExpertStreamWidth)
                throw new ArgumentException("Every rank's block must share one latent width.", nameof(blocks));
        }
        _backends = rankBackends;
        _blocks = blocks;
        _comm = comm;
        _experts = experts;
        _topK = topK;
    }

    /// <summary>Routes the first <paramref name="tokens"/> rows of the input and writes each rank's routed output into <paramref name="accum"/>:
    /// the rows that rank sources, zero elsewhere. Every <paramref name="input"/> entry holds the same replicated <c>[1, tokens, hidden]</c>
    /// values; every <paramref name="accum"/> entry is <c>[1, tokens, hidden]</c> and is overwritten.</summary>
    public void Forward(IReadOnlyList<Tensor> input, IReadOnlyList<Tensor> accum, int tokens)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(accum);
        int ranks = _backends.Count;
        if (input.Count != ranks || accum.Count != ranks) throw new ArgumentException($"Expected one input and one output per rank ({ranks}).");
        if (tokens < 1) throw new ArgumentOutOfRangeException(nameof(tokens), tokens, "At least one token is required.");
        int hidden = (int)input[0].Shape[2];
        for (int r = 0; r < ranks; r++)
        {
            if (input[r].ElementCount != (long)tokens * hidden || accum[r].ElementCount != (long)tokens * hidden)
                throw new ArgumentException($"Rank {r}'s input and output must hold {tokens} x {hidden} values.");
        }

        // Tokens split into contiguous chunks of ceil(tokens / ranks), the same source rule ExpertParallelDispatch.Plan applies.
        int width = _blocks[0].ExpertStreamWidth, k = _topK;
        int perRank = (tokens + ranks - 1) / ranks;
        int[] start = new int[ranks], count = new int[ranks];
        for (int r = 0; r < ranks; r++)
        {
            start[r] = Math.Min(r * perRank, tokens);
            count[r] = Math.Min(perRank, tokens - start[r]);
        }

        List<Tensor> owned = new();
        try
        {
            int[] routes = new int[tokens * k];
            float[] weights = new float[tokens * k];
            Tensor?[] latent = new Tensor?[ranks];
            for (int r = 0; r < ranks; r++)
            {
                if (count[r] == 0) continue;
                Tensor slice = SliceRows(input[r], start[r], count[r], hidden);
                int[] localRoutes = new int[count[r] * k];
                float[] localWeights = new float[count[r] * k];
                try
                {
                    _blocks[r].RouteRows(_backends[r], slice, count[r], localRoutes, localWeights);
                    localRoutes.CopyTo(routes, start[r] * k);
                    localWeights.CopyTo(weights, start[r] * k);
                    latent[r] = _blocks[r].LatentEnter(_backends[r], slice, count[r]);
                    owned.Add(latent[r]!);
                }
                finally
                {
                    slice.Dispose();
                }
            }

            IReadOnlyList<IReadOnlyList<ExpertParallelDispatch.Entry>> plan = ExpertParallelDispatch.Plan(tokens, _experts, ranks, routes, k);
            int[,] sendPairs = ExpertParallelDispatch.SendCounts(plan, ranks);
            long[,] rows = ToLong(sendPairs, ranks);
            long[,] backRows = Transpose(rows, ranks);

            // Dispatch: each source ships its latent rows, and the expert id of each row, to the rank that holds the expert.
            Tensor[] sendRows = new Tensor[ranks], sendIds = new Tensor[ranks], recvRows = new Tensor[ranks], recvIds = new Tensor[ranks];
            for (int r = 0; r < ranks; r++)
            {
                long total = SumRow(rows, r, ranks);
                sendRows[r] = Own(owned, NewRows(total, width));
                sendIds[r] = Own(owned, NewRows(total, 1));
            }
            for (int d = 0; d < ranks; d++)
            {
                long total = SumColumn(rows, d, ranks);
                recvRows[d] = Own(owned, NewRows(total, width));
                recvIds[d] = Own(owned, NewRows(total, 1));
            }
            int[,] cursor = new int[ranks, ranks];
            for (int d = 0; d < ranks; d++)
            {
                foreach (ExpertParallelDispatch.Entry e in plan[d])
                {
                    int r = e.SourceRank;
                    long row = AllToAllLayout.SendOffset(rows, r, d) + cursor[r, d]++;
                    int localToken = e.Token - start[r];
                    latent[r]!.AsReadOnlySpan<float>().Slice(localToken * width, width).CopyTo(sendRows[r].AsSpan<float>().Slice((int)row * width, width));
                    sendIds[r].AsSpan<float>()[(int)row] = e.Expert;
                }
            }
            Exchange(sendRows, recvRows, rows, width);
            Exchange(sendIds, recvIds, rows, 1);

            // Each destination runs its experts on the rows it received, then returns them in the order they arrived.
            Tensor[] back = new Tensor[ranks];
            for (int d = 0; d < ranks; d++)
            {
                long total = SumColumn(rows, d, ranks);
                back[d] = Own(owned, NewRows(total, width));
                if (total == 0) continue;
                int[] ids = new int[total];
                ReadOnlySpan<float> idSpan = recvIds[d].AsReadOnlySpan<float>();
                for (int i = 0; i < total; i++) ids[i] = (int)idSpan[i];
                _blocks[d].ApplyLocalExperts(_backends[d], recvRows[d], (int)total, ids, back[d]);
            }
            Tensor[] recvBack = new Tensor[ranks];
            for (int r = 0; r < ranks; r++) recvBack[r] = Own(owned, NewRows(SumRow(rows, r, ranks), width));
            Exchange(back, recvBack, backRows, width);

            // Combine on each source: row j of the return from destination d is the j-th entry d received from that source.
            float[][] combined = new float[ranks][];
            for (int r = 0; r < ranks; r++) combined[r] = new float[count[r] * width];
            int[,] returned = new int[ranks, ranks];
            for (int d = 0; d < ranks; d++)
            {
                foreach (ExpertParallelDispatch.Entry e in plan[d])
                {
                    int r = e.SourceRank;
                    long row = AllToAllLayout.SendOffset(rows, r, d) + returned[r, d]++;
                    float weight = weights[e.Token * k + e.Slot];
                    int localToken = e.Token - start[r];
                    ReadOnlySpan<float> got = recvBack[r].AsReadOnlySpan<float>().Slice((int)row * width, width);
                    float[] target = combined[r];
                    for (int j = 0; j < width; j++) target[localToken * width + j] += weight * got[j];
                }
            }

            for (int r = 0; r < ranks; r++)
            {
                Span<float> outRows = accum[r].AsSpan<float>();
                outRows.Clear();
                if (count[r] == 0) continue;
                using Tensor sum = new(new TensorShape(1, count[r], width), DType.F32);
                combined[r].AsSpan().CopyTo(sum.AsSpan<float>());
                using Tensor routed = _blocks[r].LatentExit(_backends[r], sum, count[r]);
                routed.AsReadOnlySpan<float>().CopyTo(outRows.Slice(start[r] * hidden, count[r] * hidden));
            }
        }
        finally
        {
            foreach (Tensor t in owned) t.Dispose();
        }
    }

    // One rank has nothing to move, so its send buffer stands as the receive side. Counts are rows; each row carries `width` values.
    private void Exchange(Tensor[] send, Tensor[] recv, long[,] rowCounts, int width)
    {
        int ranks = send.Length;
        if (ranks == 1)
        {
            recv[0] = send[0];
            return;
        }
        _comm!.AllToAll(send, recv, Scale(rowCounts, width, ranks));
    }

    private static Tensor NewRows(long rows, int width) => new(new TensorShape(1, (int)Math.Max(1, rows), width), DType.F32);

    private static Tensor Own(List<Tensor> owned, Tensor t)
    {
        owned.Add(t);
        return t;
    }

    private static Tensor SliceRows(Tensor x, int start, int count, int width)
    {
        Tensor slice = new(new TensorShape(1, count, width), DType.F32);
        x.AsReadOnlySpan<float>().Slice(start * width, count * width).CopyTo(slice.AsSpan<float>());
        return slice;
    }

    private static long SumRow(long[,] counts, int row, int ranks)
    {
        long total = 0;
        for (int j = 0; j < ranks; j++) total += counts[row, j];
        return total;
    }

    private static long SumColumn(long[,] counts, int column, int ranks)
    {
        long total = 0;
        for (int i = 0; i < ranks; i++) total += counts[i, column];
        return total;
    }

    private static long[,] ToLong(int[,] counts, int ranks)
    {
        long[,] result = new long[ranks, ranks];
        for (int i = 0; i < ranks; i++)
            for (int j = 0; j < ranks; j++) result[i, j] = counts[i, j];
        return result;
    }

    private static long[,] Transpose(long[,] counts, int ranks)
    {
        long[,] result = new long[ranks, ranks];
        for (int i = 0; i < ranks; i++)
            for (int j = 0; j < ranks; j++) result[j, i] = counts[i, j];
        return result;
    }

    // Element counts of a row-count matrix: each row carries `width` values.
    private static long[,] Scale(long[,] counts, int width, int ranks)
    {
        long[,] result = new long[ranks, ranks];
        for (int i = 0; i < ranks; i++)
            for (int j = 0; j < ranks; j++) result[i, j] = counts[i, j] * width;
        return result;
    }
}
