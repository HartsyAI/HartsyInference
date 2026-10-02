using System.Collections;
using System.Reflection;
using HartsyInference.Engine.Services;

namespace HartsyInference.Voice.Tests.Fakes;

/// <summary>Best-effort read of a loaded LLM slot's resident VRAM, through <see cref="TextService"/>'s private
/// per-device slot map — same reach as <see cref="CudaPlanStats"/> and for the same reason (nothing public exposes
/// a loaded slot's backend, and this project has no <c>InternalsVisibleTo</c> into Engine). Diagnostic only: never
/// throws, and a failure (a renamed field, a non-CUDA backend) reads as null rather than failing the measurement
/// it is only there to report.</summary>
internal static class VramProbe
{
    /// <summary>Used VRAM (total - free) in bytes on the backend loaded for <paramref name="device"/> (e.g.
    /// <c>"cuda:0"</c>), or null when the slot, its backend, or <c>GetVramInfo</c> could not be reached.</summary>
    public static long? UsedBytes(ITextService text, string device)
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
            PropertyInfo? backendProperty = slot?.GetType().GetProperty("Backend", BindingFlags.Public | BindingFlags.Instance);
            object? backend = backendProperty?.GetValue(slot);
            MethodInfo? method = backend?.GetType().GetMethod("GetVramInfo", BindingFlags.Public | BindingFlags.Instance);
            object? result = method?.Invoke(backend, null);
            if (result is null)
            {
                return null;
            }
            // (long FreeBytes, long TotalBytes) is a ValueTuple<long,long> at the IL level -- the friendly names
            // are compiler sugar, not real field names, so reflection reads Item1/Item2.
            Type tupleType = result.GetType();
            long free = (long)(tupleType.GetField("Item1")?.GetValue(result) ?? 0L);
            long total = (long)(tupleType.GetField("Item2")?.GetValue(result) ?? 0L);
            return total <= 0 ? null : total - free;
        }
        catch (Exception)
        {
            return null;
        }
    }

    public static string Describe(long? usedBytes) => usedBytes is { } b ? $"{b / (1024.0 * 1024):F0} MB" : "(unavailable)";
}
