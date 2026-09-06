using HartsyInference.Engine.Audio.Wake;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>Separating the wake phrase from the command.
///
/// <para>A recogniser writes the same spoken words several ways — with a comma, with an exclamation mark, in
/// any capitalisation — so the cases here are the transcripts actually seen coming off the production box plus
/// the punctuation variants around them. Getting this wrong in the safe direction (leaving the phrase on) sends
/// a small model a greeting to answer instead of a question; getting it wrong in the unsafe direction eats the
/// start of what the user said.</para></summary>
public sealed class WakePhraseTests
{
    [Theory]
    // The literal transcript the production box returned, 2026-09-05.
    [InlineData("Hey Jarvis, what time is it?", "hey_jarvis", "what time is it?")]
    [InlineData("Hey Jarvis, can you tell me what the weather is?", "hey_jarvis", "can you tell me what the weather is?")]
    [InlineData("hey jarvis what time is it", "hey_jarvis", "what time is it")]
    [InlineData("Hey, Jarvis! Turn the lights on.", "hey_jarvis", "Turn the lights on.")]
    [InlineData("HEY JARVIS - set a timer", "hey_jarvis", "set a timer")]
    [InlineData("Hey Jarvis.  What time is it?", "hey_jarvis", "What time is it?")]
    [InlineData("Alexa, play music", "alexa", "play music")]
    [InlineData("hey-hartsy, hello there", "hey-hartsy", "hello there")]
    public void Strip_RemovesLeadingPhrase(string transcript, string word, string expected) =>
        Assert.Equal(expected, WakePhrase.Strip(transcript, word));

    [Theory]
    // No leading match: the transcript is the command, whole. The third case matters most — those words appear
    // in the question, and eating them would change what was asked.
    [InlineData("What time is it?", "hey_jarvis")]
    [InlineData("Jarvis, what time is it?", "hey_jarvis")]
    [InlineData("Tell me about hey jarvis devices", "hey_jarvis")]
    [InlineData("Hey there Jarvis, hello", "hey_jarvis")]
    public void Strip_LeavesTranscriptWithoutLeadingPhrase(string transcript, string word) =>
        Assert.Equal(transcript.Trim(), WakePhrase.Strip(transcript, word));

    /// <summary>A transcript that is only the wake phrase leaves no command. The caller treats empty as "the
    /// user woke the device and said nothing", which is a real thing people do.</summary>
    [Theory]
    [InlineData("Hey Jarvis", "hey_jarvis")]
    [InlineData("Hey Jarvis.", "hey_jarvis")]
    [InlineData("hey jarvis!", "hey_jarvis")]
    public void Strip_PhraseOnly_ReturnsEmpty(string transcript, string word) =>
        Assert.Equal("", WakePhrase.Strip(transcript, word));

    [Theory]
    [InlineData(null, "hey_jarvis", "")]
    [InlineData("", "hey_jarvis", "")]
    [InlineData("   ", "hey_jarvis", "")]
    // An unknown or blank head name must not silently eat the first words of the question.
    [InlineData("Hey Jarvis, what time is it?", "", "Hey Jarvis, what time is it?")]
    [InlineData("Hey Jarvis, what time is it?", null, "Hey Jarvis, what time is it?")]
    public void Strip_HandlesMissingInput(string? transcript, string? word, string expected) =>
        Assert.Equal(expected, WakePhrase.Strip(transcript, word));

    [Theory]
    [InlineData("hey_jarvis", "hey jarvis")]
    [InlineData("hey-hartsy", "hey hartsy")]
    [InlineData("alexa", "alexa")]
    [InlineData("", "")]
    public void FromWordKey_SplitsSeparators(string word, string expected) =>
        Assert.Equal(expected, WakePhrase.FromWordKey(word));
}
