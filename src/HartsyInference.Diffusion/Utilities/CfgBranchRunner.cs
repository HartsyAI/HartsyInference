using System.Runtime.ExceptionServices;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Utilities;

/// <summary>Runs a CFG step's cond/uncond forward concurrently: uncond on a dedicated worker thread against a
/// second backend, cond on the calling thread — the VRAM-scaling counterpart to sharding (weights replicated on
/// both backends instead of split), the real 2-GPU latency win when the denoiser fits on both cards.
///
/// <para><b>Why a real <see cref="Thread"/>, not <c>Task.Run</c>.</b> The worker's first op into the second
/// backend must be the FIRST thing it does — that call's <c>EnterOp</c> is what binds the <c>[ThreadStatic]</c>
/// ambient backend state for this thread (see <c>GpuTransferHelper</c>). A thread-pool thread recycled from
/// unrelated prior work self-heals on that first call (EnterOp unconditionally rebinds), but a pool thread can
/// also sit blocked for the whole GPU forward pass, which is exactly the kind of long synchronous work the pool
/// is tuned to avoid absorbing (thread-injection stalls under load). A dedicated background thread has neither
/// concern.</para>
///
/// <para><b>Caller responsibilities.</b> Any tensor the two branches would otherwise both read (e.g. the current
/// latent) must be independent objects per branch — clone before calling <see cref="Run"/>, not inside the
/// uncond thunk (cloning inside the thunk still races the main thread's read of the shared source). Weight
/// tensors are safe to share: as long as both backends' weight caches are warm before the parallel loop starts
/// (<c>PreloadWeights</c> on both, called sequentially — not concurrently — before stepping), every in-loop
/// weight read is a per-backend cache-hit dictionary lookup that never touches the tensor's mutable GPU-binding
/// state, so two threads reading the same weight tensor concurrently never contend.</para></summary>
public static class CfgBranchRunner
{
    /// <summary>Runs <paramref name="uncond"/> on a worker thread concurrent with <paramref name="cond"/> on the
    /// calling thread, then joins. If <paramref name="cond"/> throws, the worker's outcome is still observed
    /// (so a faulted thread doesn't leak an unhandled exception) but <paramref name="cond"/>'s exception is what
    /// propagates, matching the sequential code path's exception ordering. If only <paramref name="uncond"/>
    /// throws, that exception propagates after <paramref name="cond"/> completes. On either error path the
    /// surviving branch's result tensor is disposed here before the rethrow.</summary>
    public static (Tensor Cond, Tensor Uncond) Run(Func<Tensor> cond, Func<Tensor> uncond)
    {
        ArgumentNullException.ThrowIfNull(cond);
        ArgumentNullException.ThrowIfNull(uncond);

        Tensor? uncondResult = null;
        Exception? uncondException = null;
        Thread worker = new Thread(() =>
        {
            try
            {
                uncondResult = uncond();
            }
            catch (Exception ex)
            {
                uncondException = ex;
            }
        })
        {
            IsBackground = true,
            Name = "CfgBranchRunner.Uncond",
        };
        worker.Start();

        Tensor condResult;
        try
        {
            condResult = cond();
        }
        catch
        {
            // Join before any dispose — the worker may still be writing uncondResult.
            worker.Join();
            uncondResult?.Dispose();
            throw;
        }

        worker.Join();
        if (uncondException is not null)
        {
            condResult.Dispose();
            ExceptionDispatchInfo.Capture(uncondException).Throw();
        }
        return (condResult, uncondResult!);
    }
}
