"""Regenerates the reference the Chatterbox frontend parity test diffs against.

Upstream ChatterboxTTS.generate (resemble-ai/chatterbox src/chatterbox/tts.py) prepares text as
punc_norm(text) -> EnTokenizer.text_to_tokens -> pad [START] (255) left and [STOP] (0) right. This runs exactly that
on every line of CORPUS and writes {text, normalized, ids} per line, ids including the start/stop pads.

    pip install --no-deps <chatterbox checkout>      # plus torch (CPU) and tokenizers
    python tools/parity/chatterbox_frontend_dump.py <tokenizer.json> \
        tests/HartsyInference.Audio.Tests/Fixtures/chatterbox_frontend_parity.json
"""
import json
import sys
import unicodedata

from chatterbox.models.tokenizers import EnTokenizer
from chatterbox.tts import punc_norm

START_TEXT_TOKEN = 255
STOP_TEXT_TOKEN = 0

CORPUS = [
    "Yes.",
    "Okay, see you.",
    "The weather is nice today, so we might take a walk.",
    "On March 3rd, 2026 at 4:30 PM, 1,250 people paid $99.95 each.",
    "Wait... what?! No -- really; (yes), 'quoted' & \"double\" -- ok?",
    "No no no no no no no no no no no no.",
    "The quick brown fox jumps over the lazy dog near the river bank.",
    # empty-ish
    "",
    " ",
    "   ",
    "\t\n",
    ".",
    "?",
    "...",
    "…",
    "-",
    "a",
    "I",
    # lowercase start
    "hello there.",
    "yes",
    "okay, see you",
    "the end is near",
    "it's fine, really!",
    "éclair is a pastry.",
    "x",
    "  leading lowercase",
    "\tlower tab",
    "\U0001D44E math italic start",
    "ﬁne ligature start.",
    "ﬀ ligature.",
    "ﬂ ligature.",
    "ﬃ ligature.",
    "ﬄ ligature.",
    "ﬅ ligature.",
    "ﬆ ligature.",
    "ŉ apostrophe n.",
    "ª ordinal start.",
    "ⓐ circled start.",
    "ǅ titlecase start.",
    "ǰ j caron start.",
    "Next\x85line.",
    # no final punctuation
    "Hello world",
    "This sentence has no ending",
    "Numbers like 42",
    "Ends with a quote \"",
    "Ends with a paren (",
    "Ends with colon:",
    "Ends with semicolon;",
    "Ends with a dash -",
    "Ends with an em dash —",
    "Ends with comma,",
    "Ends with space   ",
    # multiple spaces / whitespace runs
    "Too    many     spaces.",
    "  Leading spaces.",
    "Trailing spaces.   ",
    "Tabs\tand\nnewlines\r\nmixed.",
    "Non breaking em　ideographic spaces.",
    "Line separator and paragraph.",
    "Odd \x1c file \x1f separators.",
    # unicode quotes, dashes, ellipsis
    "“Smart double quotes,” she said.",
    "‘Single smart quotes’ are here.",
    "It’s a dog’s life.",
    "Em—dash and en–dash.",
    "Spaced em — dash and spaced en – dash.",
    "Wait… really…",
    "So... then... what?",
    "Four dots.... here.",
    "Spaced hyphen - here.",
    "Double -- hyphen.",
    "Space before comma , here.",
    "Ellipsis then comma..., odd.",
    "Colon: list: items.",
    "Time 10:45 now.",
    "Semi; colon; chain;",
    "Mixed “quotes’ and 'plain' ones.",
    "«Guillemets» are not replaced.",
    "Prime ′ and double prime ″.",
    "Minus − sign.",
    # punctuation and symbols
    "Hello, world!",
    "Really?!",
    "What?",
    "Stop!",
    "Hmm, well, okay.",
    "Email me at someone@example.com today.",
    "Path /usr/local/bin is fine.",
    "100% sure & 50/50 odds.",
    "Price: $5, €10, £15.",
    "Use #hashtag and *stars* and _underscores_.",
    "Brackets [like this] and {braces}.",
    "A [laughter] tag and [sigh] tag.",
    "Tilde ~ caret ^ pipe | backslash \\.",
    "Numbers 1 2 3 and 4.5 and -7.",
    "ALL CAPS SENTENCE HERE.",
    "MiXeD CaSe WoRdS.",
    "Don't, won't, can't, shouldn't.",
    "Rock 'n' roll.",
    "Naïve café résumé.",
    "Straße and ß.",
    "ßharp start.",
    "Greek αβγ letters.",
    "Emoji \U0001F600 here.",
    "Chinese 你好 words.",
    "Combining é accent.",
    "Under_score and hyphen-ated words.",
    "Mr. Smith went to Washington, D.C. yesterday.",
    "First line.\nSecond line.",
    "Question one? Question two?",
    "Wow!!!",
    "Commas,,, everywhere.",
    "Hyphenated-word-chain-here",
    "Okay -",
    "Okay ,",
    "One, two, three, four, five, six, seven, eight, nine, ten.",
    "A very long sentence that keeps going and going with many words so that the tokenizer has to handle a "
    "fairly long input string without any trouble at all, which is the point of this particular line.",
]


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: chatterbox_frontend_dump.py <tokenizer.json> <out.json>")
    tokenizer = EnTokenizer(sys.argv[1])
    lines = []
    for text in CORPUS:
        normalized = punc_norm(text)
        ids = [START_TEXT_TOKEN] + list(tokenizer.text_to_tokens(normalized)[0].tolist()) + [STOP_TEXT_TOKEN]
        lines.append({"text": text, "normalized": normalized, "ids": ids})
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        f.write('{"source": "resemble-ai/chatterbox tts.py punc_norm + EnTokenizer", "lines": [\n')
        f.write(",\n".join(json.dumps(line, ensure_ascii=True) for line in lines))
        # punc_norm's first-character rule over every code point it changes: {code point: text[0].upper()}.
        capitalize = {f"{cp:X}": chr(cp).upper() for cp in range(0x110000)
                      if chr(cp).islower() and chr(cp).upper() != chr(cp)}
        f.write("\n],\n\"unicode\": \"%s\",\n\"capitalize\": %s}\n"
                % (unicodedata.unidata_version, json.dumps(capitalize, ensure_ascii=True)))
    print(f"wrote {len(lines)} lines")


if __name__ == "__main__":
    main()
