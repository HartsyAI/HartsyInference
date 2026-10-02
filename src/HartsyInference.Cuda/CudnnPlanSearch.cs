using System.Collections.Concurrent;
using System.Diagnostics;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Logging;
using static HartsyInference.Cuda.CudnnApi;

namespace HartsyInference.Cuda;

/// <summary>Where one execution-plan build spent its time and which engine it settled on. Filled by
/// <see cref="CudnnPlanSearch.BuildExecutionPlan"/> when the caller passes one; the build itself is unchanged.</summary>
internal sealed class PlanBuildProbe
{
    /// <summary>Stopwatch ticks in the heuristic query: descriptor creation, finalize and reading the results.</summary>
    public long HeuristicTicks;

    /// <summary>Stopwatch ticks finalizing execution plans, the rejected candidates included.</summary>
    public long FinalizeTicks;

    /// <summary>Engine configs finalized before one produced a usable plan, that one included.</summary>
    public int ConfigsTried;

    /// <summary>The chosen engine's global index, or -1 when it could not be read.</summary>
    public long EngineGlobalIndex = -1;

    /// <summary>Whether the chosen engine carries <c>CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION</c> (an NVRTC kernel built
    /// for the shape at finalize time).</summary>
    public bool RuntimeCompiled;

    /// <summary>Set by a caller that will plan other shapes from this build's choice: read the knob choices too.</summary>
    public bool CaptureKnobs;

    /// <summary>The chosen config's knob choices (type, value) when <see cref="CaptureKnobs"/> is set, or null when they were
    /// not asked for or could not be read.</summary>
    public (int Type, long Value)[]? Knobs;
}

/// <summary>An engine configuration lifted off one plan so another shape of the same operation can be planned with it
/// without a heuristic query: the engine's global index and every knob choice the heuristic made.</summary>
internal sealed record EngineChoice(long GlobalIndex, (int Type, long Value)[] Knobs)
{
    /// <summary>The choice the probe recorded, or null when the engine or its knobs could not be read.</summary>
    public static EngineChoice? From(PlanBuildProbe probe) =>
        probe.EngineGlobalIndex >= 0 && probe.Knobs is not null ? new EngineChoice(probe.EngineGlobalIndex, probe.Knobs) : null;
}

/// <summary>The cuDNN backend-graph engine-config search shared by <see cref="CudnnConv"/> and <see cref="CudnnSdpa"/>.</summary>
internal static class CudnnPlanSearch
{
    /// <summary>(<paramref name="what"/>, engine global index, shape signature) triples already logged, so a hot
    /// path that replans the same shape repeatedly (every length-bucket miss, every call before the plan cache is
    /// warm) logs once per DISTINCT shape, not per call -- but every distinct shape that lands on a nondeterministic
    /// engine still gets its own line, so the fleet's exposure is answerable from logs alone without a digest
    /// bisection (see issue #20).</summary>
    private static readonly ConcurrentDictionary<(string What, long EngineIndex, string Shape), byte> _loggedDeterminism = new();

    /// <summary>True when the engine behind <paramref name="cfg"/> carries <c>CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC</c>
    /// (an atomic-accumulation reduction whose float summation order varies run to run -- see issue #20). False on
    /// any read failure, so a cuDNN version without this attribute, or a transient read error, behaves like today:
    /// engines are never filtered out for a reason this build cannot see.</summary>
    private static unsafe bool IsNondeterministic(nint cfg, out long engineGlobalIndex)
    {
        engineGlobalIndex = -1;
        if (cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINE_DESCRIPTOR, out nint engine) != CUDNN_STATUS_SUCCESS)
            return false;
        try
        {
            nint target = engine;
            if (cudnnBackendGetAttribute(cfg, CUDNN_ATTR_ENGINECFG_ENGINE, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1,
                    out long engines, &target) != CUDNN_STATUS_SUCCESS || engines < 1)
                return false;
            long index = -1;
            if (cudnnBackendGetAttribute(engine, CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64, 1, out _, &index)
                == CUDNN_STATUS_SUCCESS)
                engineGlobalIndex = index;
            const int maxNotes = 8;
            int* notes = stackalloc int[maxNotes];
            if (cudnnBackendGetAttribute(engine, CUDNN_ATTR_ENGINE_NUMERICAL_NOTE, CUDNN_TYPE_NUMERICAL_NOTE, maxNotes,
                    out long noteCount, notes) != CUDNN_STATUS_SUCCESS)
                return false;
            for (int i = 0; i < noteCount && i < maxNotes; i++)
                if (notes[i] == CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC)
                    return true;
            return false;
        }
        finally
        {
            cudnnBackendDestroyDescriptor(engine);
        }
    }

    /// <summary>Logs once per (<paramref name="what"/>, engine, shape) the first time this process sees a
    /// nondeterministic engine land for it, whether it was picked (knob off, or no deterministic alternative) or
    /// skipped (knob on, a deterministic candidate took over) -- so the fleet's exposure to issue #20's class of
    /// drift is visible in the log regardless of which way the knob is set, and answerable per-shape without a
    /// digest bisection. <paramref name="shape"/> is caller-supplied (op/Cin/Cout/kernel/stride/dtype or
    /// op/batch/heads/seqlen/headdim/dtype) rather than read back off the opaque graph/engine descriptors, since
    /// the caller already has it in scope when it builds the graph.</summary>
    private static void LogNondeterministicOnce(string what, long engineGlobalIndex, string shape, bool skipped)
    {
        if (_loggedDeterminism.TryAdd((what, engineGlobalIndex, shape), 0))
        {
            Logs.Info(skipped
                ? $"[CudnnPlanSearch] {what} [{shape}]: engine {engineGlobalIndex} is CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC -- skipped (numerics.cudnnDeterministic on), trying the next candidate."
                : $"[CudnnPlanSearch] {what} [{shape}]: engine {engineGlobalIndex} is CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC and was used (numerics.cudnnDeterministic off) -- this shape's output is not reproducible run to run.");
        }
    }

    /// <summary>Asks heuristics (mode A = recommended runtime-compiled fused engines first, FALLBACK second) for engine configs and finalizes the first that yields a valid plan whose workspace fits <paramref name="maxWorkspaceBytes"/>. When <see cref="EngineKnobs.CudnnDeterministic"/> is on, an engine carrying <c>CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC</c> is skipped in favor of the next candidate (across both heuristic modes); if every candidate is nondeterministic, this throws and the caller's existing direct-kernel fallback takes over, same as any other "no engine config" failure. <paramref name="shape"/> identifies the op/shape/dtype for the nondeterminism log line only (see <see cref="LogNondeterministicOnce"/>); null logs as "unknown-shape" rather than failing. A non-null <paramref name="probe"/> records the time per phase and the chosen engine.</summary>
    internal static unsafe (nint exec, long wsBytes) BuildExecutionPlan(nint handle, nint graph, List<nint> owned, long maxWorkspaceBytes,
        string what, string? shape = null, PlanBuildProbe? probe = null)
    {
        string shapeOrUnknown = shape ?? "unknown-shape";
        bool deterministic = EngineKnobs.CudnnDeterministic.Value;
        foreach (int mode in new[] { CUDNN_HEUR_MODE_A, CUDNN_HEUR_MODE_FALLBACK })
        {
            long heuristicStart = Stopwatch.GetTimestamp();
            nint heur;
            if (cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, out heur) != CUDNN_STATUS_SUCCESS)
                continue;
            owned.Add(heur);
            void* gp = (void*)graph;
            SetAttr(heur, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &gp);
            int m = mode;
            SetAttr(heur, CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, 1, &m);
            if (cudnnBackendFinalize(heur) != CUDNN_STATUS_SUCCESS)
            {
                if (probe is not null) probe.HeuristicTicks += Stopwatch.GetTimestamp() - heuristicStart;
                continue;
            }

            const int maxCfgs = 32;
            nint[] cfgs = new nint[maxCfgs];
            for (int i = 0; i < maxCfgs; i++)
                cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, out cfgs[i]);
            long returned;
            fixed (nint* cfgPtr = cfgs)
            {
                int gst = cudnnBackendGetAttribute(heur, CUDNN_ATTR_ENGINEHEUR_RESULTS,
                    CUDNN_TYPE_BACKEND_DESCRIPTOR, maxCfgs, out returned, cfgPtr);
                if (gst != CUDNN_STATUS_SUCCESS) returned = 0;
            }
            if (probe is not null) probe.HeuristicTicks += Stopwatch.GetTimestamp() - heuristicStart;
            // TryPlan can throw (SetAttr failure, e.g. a transient host-allocation error) instead of
            // returning ok=false — without this try/finally, an exception mid-loop would skip every
            // remaining cfgs[i]'s destroy call (both the current index and every index not yet reached),
            // leaking up to 32 backend descriptors per throw. Under exactly the resource-pressure
            // conditions that cause such a throw, that leak would make the underlying pressure worse with
            // every retry — tracked per-index so the normal (non-throwing) destroy calls below aren't
            // double-freed here.
            bool[] destroyed = new bool[maxCfgs];
            try
            {
                for (int i = 0; i < maxCfgs; i++)
                {
                    if (i < returned)
                    {
                        long finalizeStart = Stopwatch.GetTimestamp();
                        (nint exec, long ws, bool ok) = TryPlan(handle, cfgs[i]);
                        if (probe is not null)
                        {
                            probe.FinalizeTicks += Stopwatch.GetTimestamp() - finalizeStart;
                            probe.ConfigsTried++;
                        }
                        if (ok && ws > maxWorkspaceBytes)
                        {
                            cudnnBackendDestroyDescriptor(exec);
                            ok = false;
                        }
                        if (ok && IsNondeterministic(cfgs[i], out long engineIdx))
                        {
                            if (deterministic)
                            {
                                cudnnBackendDestroyDescriptor(exec);
                                ok = false;
                                LogNondeterministicOnce(what, engineIdx, shapeOrUnknown, skipped: true);
                            }
                            else
                            {
                                LogNondeterministicOnce(what, engineIdx, shapeOrUnknown, skipped: false);
                            }
                        }
                        if (ok)
                        {
                            if (probe is not null)
                                DescribeEngine(cfgs[i], probe);
                            for (int j = 0; j < maxCfgs; j++)
                            {
                                if (!destroyed[j]) { cudnnBackendDestroyDescriptor(cfgs[j]); destroyed[j] = true; }
                            }
                            return (exec, ws);
                        }
                    }
                    cudnnBackendDestroyDescriptor(cfgs[i]);
                    destroyed[i] = true;
                }
            }
            finally
            {
                for (int i = 0; i < maxCfgs; i++)
                    if (!destroyed[i]) cudnnBackendDestroyDescriptor(cfgs[i]);
            }
        }
        throw new InvalidOperationException($"cuDNN {what}: no engine config produced a valid execution plan");
    }

    /// <summary>Builds an execution plan for <paramref name="graph"/> from <paramref name="choice"/> instead of asking the
    /// heuristic: the engine at the choice's global index with the choice's knobs. False when the engine does not support
    /// this graph, a knob value is invalid for it, the plan does not finalize, or it wants more workspace than
    /// <paramref name="maxWorkspaceBytes"/>; the caller then runs the heuristic as before. Descriptors it creates go into
    /// <paramref name="owned"/>; the plan does not depend on them once finalized.</summary>
    internal static unsafe bool TryPlanFromChoice(nint handle, nint graph, List<nint> owned, EngineChoice choice,
        long maxWorkspaceBytes, out nint exec, out long workspaceBytes)
    {
        exec = 0;
        workspaceBytes = 0;
        try
        {
            if (cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINE_DESCRIPTOR, out nint engine) != CUDNN_STATUS_SUCCESS)
                return false;
            owned.Add(engine);
            void* gp = (void*)graph;
            long index = choice.GlobalIndex;
            if (cudnnBackendSetAttribute(engine, CUDNN_ATTR_ENGINE_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &gp) != CUDNN_STATUS_SUCCESS
                || cudnnBackendSetAttribute(engine, CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64, 1, &index) != CUDNN_STATUS_SUCCESS
                || cudnnBackendFinalize(engine) != CUDNN_STATUS_SUCCESS)
                return false;

            nint[] knobs = new nint[choice.Knobs.Length];
            for (int i = 0; i < knobs.Length; i++)
            {
                if (cudnnBackendCreateDescriptor(CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR, out knobs[i]) != CUDNN_STATUS_SUCCESS)
                    return false;
                owned.Add(knobs[i]);
                int type = choice.Knobs[i].Type;
                long value = choice.Knobs[i].Value;
                if (cudnnBackendSetAttribute(knobs[i], CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE, CUDNN_TYPE_KNOB_TYPE, 1, &type) != CUDNN_STATUS_SUCCESS
                    || cudnnBackendSetAttribute(knobs[i], CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE, CUDNN_TYPE_INT64, 1, &value) != CUDNN_STATUS_SUCCESS
                    || cudnnBackendFinalize(knobs[i]) != CUDNN_STATUS_SUCCESS)
                    return false;
            }

            if (cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, out nint cfg) != CUDNN_STATUS_SUCCESS)
                return false;
            owned.Add(cfg);
            void* ep = (void*)engine;
            if (cudnnBackendSetAttribute(cfg, CUDNN_ATTR_ENGINECFG_ENGINE, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &ep) != CUDNN_STATUS_SUCCESS)
                return false;
            if (knobs.Length > 0)
            {
                fixed (nint* kp = knobs)
                {
                    if (cudnnBackendSetAttribute(cfg, CUDNN_ATTR_ENGINECFG_KNOB_CHOICES, CUDNN_TYPE_BACKEND_DESCRIPTOR, knobs.Length, kp)
                        != CUDNN_STATUS_SUCCESS)
                        return false;
                }
            }
            if (cudnnBackendFinalize(cfg) != CUDNN_STATUS_SUCCESS)
                return false;

            (nint plan, long ws, bool ok) = TryPlan(handle, cfg);
            if (!ok)
                return false;
            if (ws > maxWorkspaceBytes)
            {
                cudnnBackendDestroyDescriptor(plan);
                return false;
            }
            exec = plan;
            workspaceBytes = ws;
            return true;
        }
        catch (CudnnStatusException)
        {
            // TryPlan's attribute writes throw rather than return a status; any refusal here means "plan this length
            // the usual way", never a reason to give up on cuDNN for the session.
            return false;
        }
    }

    /// <summary>Reads the engine behind <paramref name="cfg"/> into <paramref name="probe"/>: its global index, whether it
    /// is runtime-compiled, and the config's knob choices. A failed read leaves the probe's defaults.</summary>
    private static unsafe void DescribeEngine(nint cfg, PlanBuildProbe probe)
    {
        if (probe.CaptureKnobs)
            probe.Knobs = ReadKnobChoices(cfg);
        if (cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINE_DESCRIPTOR, out nint engine) != CUDNN_STATUS_SUCCESS)
            return;
        try
        {
            nint target = engine;
            if (cudnnBackendGetAttribute(cfg, CUDNN_ATTR_ENGINECFG_ENGINE, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1,
                    out long engines, &target) != CUDNN_STATUS_SUCCESS || engines < 1)
                return;
            long index = -1;
            if (cudnnBackendGetAttribute(engine, CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64, 1, out _, &index)
                == CUDNN_STATUS_SUCCESS)
                probe.EngineGlobalIndex = index;
            const int maxNotes = 8;
            int* notes = stackalloc int[maxNotes];
            if (cudnnBackendGetAttribute(engine, CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE, CUDNN_TYPE_BEHAVIOR_NOTE, maxNotes,
                    out long noteCount, notes) == CUDNN_STATUS_SUCCESS)
            {
                for (int i = 0; i < noteCount && i < maxNotes; i++)
                {
                    if (notes[i] == CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION)
                        probe.RuntimeCompiled = true;
                }
            }
        }
        finally
        {
            cudnnBackendDestroyDescriptor(engine);
        }
    }

    /// <summary>The knob choices of a finalized engine config, or null when they could not be read.</summary>
    private static unsafe (int Type, long Value)[]? ReadKnobChoices(nint cfg)
    {
        // Every knob type cuDNN 9 defines fits (44 in 9.20); the array is only the receiving capacity.
        const int maxKnobs = 64;
        nint[] descriptors = new nint[maxKnobs];
        int created = 0;
        try
        {
            for (; created < maxKnobs; created++)
            {
                if (cudnnBackendCreateDescriptor(CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR, out descriptors[created]) != CUDNN_STATUS_SUCCESS)
                    return null;
            }
            long count;
            fixed (nint* dp = descriptors)
            {
                if (cudnnBackendGetAttribute(cfg, CUDNN_ATTR_ENGINECFG_KNOB_CHOICES, CUDNN_TYPE_BACKEND_DESCRIPTOR, maxKnobs,
                        out count, dp) != CUDNN_STATUS_SUCCESS)
                    return null;
            }
            (int Type, long Value)[] knobs = new (int, long)[Math.Min(count, maxKnobs)];
            for (int i = 0; i < knobs.Length; i++)
            {
                int type = 0;
                long value = 0;
                if (cudnnBackendGetAttribute(descriptors[i], CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE, CUDNN_TYPE_KNOB_TYPE, 1, out _, &type)
                        != CUDNN_STATUS_SUCCESS
                    || cudnnBackendGetAttribute(descriptors[i], CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE, CUDNN_TYPE_INT64, 1, out _, &value)
                        != CUDNN_STATUS_SUCCESS)
                    return null;
                knobs[i] = (type, value);
            }
            return knobs;
        }
        finally
        {
            for (int i = 0; i < created; i++)
                cudnnBackendDestroyDescriptor(descriptors[i]);
        }
    }

    private static unsafe (nint exec, long ws, bool ok) TryPlan(nint handle, nint cfg)
    {
        if (cudnnBackendCreateDescriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, out nint p) != CUDNN_STATUS_SUCCESS)
            return (0, 0, false);
        try
        {
            void* hp = (void*)handle;
            void* cp = (void*)cfg;
            SetAttr(p, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, 1, &hp);
            SetAttr(p, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &cp);
        }
        catch
        {
            // SetAttr throws directly on failure (doesn't return a status) — without this, a thrown
            // exception here would leak the just-created execution-plan descriptor `p`.
            cudnnBackendDestroyDescriptor(p);
            throw;
        }
        if (cudnnBackendFinalize(p) != CUDNN_STATUS_SUCCESS)
        {
            cudnnBackendDestroyDescriptor(p);
            return (0, 0, false);
        }
        long ws = 0;
        try
        {
            Check(cudnnBackendGetAttribute(
                p, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 1, out _, &ws),
                "workspace size get");
        }
        catch
        {
            cudnnBackendDestroyDescriptor(p);
            throw;
        }
        return (p, ws, true);
    }
}
