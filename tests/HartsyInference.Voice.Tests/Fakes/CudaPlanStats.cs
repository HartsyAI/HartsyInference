using System.Collections;
using System.Reflection;
using HartsyInference.Engine.Services;

namespace HartsyInference.Voice.Tests.Fakes;

/// <summary>Best-effort read of <c>CudaBackend.LtGemmPlanStats</c> for the LLM slot loaded on one device, through
/// <see cref="TextService"/>'s private per-device slot map — nothing public exposes a loaded slot's backend, and
/// this project has no <c>InternalsVisibleTo</c> into Engine (unlike LLM.Tests), so reflection is the only way to
/// reach it from here. Diagnostic only: reads once per call, never on a hot path, and never throws — a failure
/// (a renamed field, a non-CUDA backend) comes back as a null description rather than failing the measurement it
/// is only there to help explain.</summary>
internal static class CudaPlanStats
{
    /// <summary>cuBLASLt plan-cache counters for the LLM loaded on <paramref name="device"/> (e.g. <c>"cuda:0"</c>),
    /// or null when the slot, its backend, or the stats property could not be reached.</summary>
    public static string? Describe(ITextService text, string device)
    {
        if (text is not TextService textService)
        {
            return null;
        }
        try
        {
            FieldInfo? slotsField = typeof(TextService).GetField("_slots", BindingFlags.NonPublic | BindingFlags.Instance);
            if (slotsField?.GetValue(textService) is not IDictionary slots)
            {
                return null;
            }
            object? slot = null;
            foreach (DictionaryEntry entry in slots)
            {
                if (entry.Key is string key && string.Equals(key, device, StringComparison.OrdinalIgnoreCase))
                {
                    slot = entry.Value;
                    break;
                }
            }
            if (slot is null)
            {
                return null;
            }
            PropertyInfo? backendProperty = slot.GetType().GetProperty("Backend", BindingFlags.Public | BindingFlags.Instance);
            object? backend = backendProperty?.GetValue(slot);
            if (backend is null)
            {
                return null;
            }
            PropertyInfo? statsProperty = backend.GetType().GetProperty("LtGemmPlanStats", BindingFlags.Public | BindingFlags.Instance);
            object? stats = statsProperty?.GetValue(backend);
            return stats is null ? null : $"{backend.GetType().Name} LtGemmPlanStats={stats}";
        }
        catch (Exception ex)
        {
            return $"(plan stats unavailable: {ex.GetType().Name}: {ex.Message})";
        }
    }
}
