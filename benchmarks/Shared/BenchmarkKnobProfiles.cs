using System.Globalization;
using HartsyInference.Core.Configuration;

namespace HartsyInference.Benchmarks.Shared;

/// <summary>One benchmark settings freeze shared by request and operation tools.</summary>
public static class BenchmarkKnobProfiles
{
    /// <summary>Pins registry defaults, optionally including explicit host overrides applied by an operation arm.</summary>
    public static KnobProfile Capture(bool includeHostOverrides, out SortedDictionary<string, string> settings)
    {
        settings = new(StringComparer.Ordinal);
        KnobProfile profile = KnobProfile.Create("benchmark-defaults");
        foreach (object knob in KnobRegistry.All)
        {
            (string id, string type, object? value, KnobScope scope, KnobDomain domain, string summary) = KnobRegistry.Describe(knob);
            if (includeHostOverrides && KnobStore.SourceOf(id) == "host") value = KnobRegistry.ValueOf(knob);
            settings[id] = Convert.ToString(value, CultureInfo.InvariantCulture) ?? "<null>";
            profile = knob switch
            {
                Knob<bool> k => profile.With(k, (bool)value!),
                Knob<int> k => profile.With(k, (int)value!),
                Knob<long> k => profile.With(k, (long)value!),
                Knob<float> k => profile.With(k, (float)value!),
                Knob<bool?> k => profile.With(k, (bool?)value),
                Knob<int?> k => profile.With(k, (int?)value),
                Knob<long?> k => profile.With(k, (long?)value),
                Knob<float?> k => profile.With(k, (float?)value),
                Knob<string> k => profile.With(k, (string)value!),
                _ => throw new InvalidDataException("Unsupported knob type: " + type),
            };
        }
        return profile;
    }
}
