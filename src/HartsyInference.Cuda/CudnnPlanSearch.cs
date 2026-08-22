using static HartsyInference.Cuda.CudnnApi;

namespace HartsyInference.Cuda;

/// <summary>The cuDNN backend-graph engine-config search shared by <see cref="CudnnConv"/> and <see cref="CudnnSdpa"/>.</summary>
internal static class CudnnPlanSearch
{
    /// <summary>Asks heuristics (mode A = recommended runtime-compiled fused engines first, FALLBACK second) for engine configs and finalizes the first that yields a valid plan whose workspace fits <paramref name="maxWorkspaceBytes"/>.</summary>
    internal static unsafe (nint exec, long wsBytes) BuildExecutionPlan(nint handle, nint graph, List<nint> owned, long maxWorkspaceBytes, string what)
    {
        foreach (int mode in new[] { CUDNN_HEUR_MODE_A, CUDNN_HEUR_MODE_FALLBACK })
        {
            nint heur;
            if (cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, out heur) != CUDNN_STATUS_SUCCESS)
                continue;
            owned.Add(heur);
            void* gp = (void*)graph;
            SetAttr(heur, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &gp);
            int m = mode;
            SetAttr(heur, CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, 1, &m);
            if (cudnnBackendFinalize(heur) != CUDNN_STATUS_SUCCESS)
                continue;

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
                        (nint exec, long ws, bool ok) = TryPlan(handle, cfgs[i]);
                        if (ok && ws > maxWorkspaceBytes)
                        {
                            cudnnBackendDestroyDescriptor(exec);
                            ok = false;
                        }
                        if (ok)
                        {
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
