using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.MiniMaxH3;

/// <summary>Generation-scoped device residency and head fusion for one PDD bank/backend pair.</summary>
public sealed class PddHeadFusionSession : IDisposable
{
    private readonly IBackend _backend;
    private readonly PddHeadBank _bank;
    private readonly IReadOnlyList<Tensor> _residentWeights;
    private int _disposed;

    /// <summary>Bulk-preloads every bank row so no host transfer occurs inside sampling.</summary>
    public PddHeadFusionSession(IBackend backend, PddHeadBank bank)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        _bank = bank ?? throw new ArgumentNullException(nameof(bank));
        _residentWeights = bank.EnumerateWeights().ToArray();
        _backend.PreloadWeights(_residentWeights);
    }

    /// <summary>Fuses the heads for an actual current/next sigma pair entirely through backend operations.</summary>
    public PddFusedHeads Fuse(MiniMaxH3PddSchedule schedule, double sigma, double sigmaNext)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ArgumentNullException.ThrowIfNull(schedule);
        MiniMaxH3PddStep step = schedule.Resolve(sigma, sigmaNext);
        if (step.FineStart + step.FineCount > _bank.StepCount)
            throw new HartsyInferenceException("PDD schedule addresses beyond the loaded head bank.");

        Tensor? videoWeight = null;
        Tensor? videoBias = null;
        Tensor? audioWeight = null;
        Tensor? audioBias = null;
        try
        {
            videoWeight = FuseRows(_bank.GetVideoWeight, step.VideoWeights, step.FineStart,
                new TensorShape(_bank.VideoChannels, _bank.HiddenSize));
            videoBias = FuseRows(_bank.GetVideoBias, step.VideoWeights, step.FineStart,
                new TensorShape(_bank.VideoChannels));
            audioWeight = FuseRows(_bank.GetAudioWeight, step.AudioWeights, step.FineStart,
                new TensorShape(_bank.AudioChannels, _bank.HiddenSize));
            audioBias = FuseRows(_bank.GetAudioBias, step.AudioWeights, step.FineStart,
                new TensorShape(_bank.AudioChannels));
            return new PddFusedHeads(videoWeight, videoBias, audioWeight, audioBias);
        }
        catch
        {
            videoWeight?.Dispose();
            videoBias?.Dispose();
            audioWeight?.Dispose();
            audioBias?.Dispose();
            throw;
        }
    }

    private Tensor FuseRows(Func<int, Tensor> rowAt, IReadOnlyList<float> coefficients, int start,
        TensorShape shape)
    {
        int firstIndex;
        Tensor current = new Tensor(shape, DType.F32);
        if (_bank.Layout == MiniMaxH3PddHeadLayout.BasePlusOffsets)
        {
            _backend.Scale(current, rowAt(0), 1.0f);
            firstIndex = Math.Max(start, 1);
        }
        else
        {
            _backend.Scale(current, rowAt(start), coefficients[0]);
            firstIndex = start + 1;
        }

        try
        {
            int stop = start + coefficients.Count;
            for (int index = firstIndex; index < stop; index++)
            {
                using Tensor scaled = new Tensor(shape, DType.F32);
                _backend.Scale(scaled, rowAt(index), coefficients[index - start]);
                Tensor next = new Tensor(shape, DType.F32);
                try
                {
                    _backend.Add(next, current, scaled);
                }
                catch
                {
                    next.Dispose();
                    throw;
                }
                current.Dispose();
                current = next;
            }
            return current;
        }
        catch
        {
            current.Dispose();
            throw;
        }
    }

    /// <summary>Evicts the bank rows from this backend after the generation completes.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        _backend.FreeWeights(_residentWeights);
    }
}
