using HartsyInference.Audio.Frontends.WeText;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Logging;

namespace HartsyInference.Engine.Audio;

/// <summary>The FST normalizers every IndexTTS version loads with: wetext 0.1.8 over the WeText grammars (upstream
/// parity), or the interim English spoken form when the grammars can neither be found nor fetched, so an offline
/// install still loads the model.</summary>
internal static class IndexTtsTextNormalizers
{
    public static async Task<IndexTtsFstNormalizers> LoadAsync(CancellationToken cancel)
    {
        try
        {
            WeTextNormalizer weText = await WeTextAssets.LoadAsync(WeTextLanguage.Auto, removeErhua: false, WeTextVersion.V0_1_8, cancel)
                .ConfigureAwait(false);
            return IndexTtsFstNormalizers.FromWeText(weText);
        }
        catch (Exception ex) when (!cancel.IsCancellationRequested
            && ex is HttpRequestException or IOException or InvalidDataException or UnauthorizedAccessException or TaskCanceledException)
        {
            Logs.Warning("[Audio][IndexTTS] WeText grammars unavailable (" + ex.Message + "); using the approximate English "
                + "spoken form and leaving Chinese numbers as digits. Text matches upstream again once the grammars can be fetched.");
            return IndexTtsFstNormalizers.Interim;
        }
    }
}
