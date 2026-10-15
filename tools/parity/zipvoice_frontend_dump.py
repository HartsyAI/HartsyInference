"""Dump ZipVoice's upstream text and audio front end as a parity fixture for the engine port.

Runs k2-fsa/ZipVoice's own code (EmiliaTokenizer and EnglishTextNormalizer, add_punctuation,
chunk_tokens_punctuation, cross_fade_concat and the pydub-based remove_silence) over a fixed corpus and writes, per
line, the segments, normalized text, token symbols and token ids, plus chunker, cross-fade and silence cases. The
engine tests (ZipVoiceFrontendParityTests, ZipVoiceAudioParityTests) assert equality against this file.

Newlines are replaced by spaces before upstream sees them: its segmenter drops them, which would glue words together,
and the engine reads them as whitespace instead.

Usage (reference venv with piper_phonemize, pydub, inflect, cn2an, jieba, pypinyin, torch, lhotse):
    python tools/parity/zipvoice_frontend_dump.py --upstream /path/to/ZipVoice --tokens /path/to/tokens.txt \
        --out tests/HartsyInference.Audio.Tests/Fixtures/zipvoice_frontend_parity.json
"""

import argparse
import hashlib
import importlib.metadata
import json
import logging
import random
import subprocess
import sys

import numpy as np

CORPUS = [
    "Yes.",
    "Okay, see you.",
    "The weather is nice today, so we might take a walk.",
    "After the meeting ended, the team gathered in the kitchen to talk about the new product launch, the budget, and the schedule for the coming quarter.",
    "Our water bill arrived late this month, so I called the office to ask why. The clerk said a meter reading was missed in March. She promised to fix the record and send a corrected statement within two weeks. I thanked her, hung up, and wrote the date on my calendar so I would remember to check. Last time this happened it took three phone calls, so I am hoping this one goes faster.",
    "On March 3rd, 2026 at 4:30 PM, 1,250 people paid $99.95 each.",
    "Wait... what?! No -- really; (yes), 'quoted' & \"double\" -- ok?",
    "No no no no no no no no no no no no.",
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "I have 3 cats and 12 dogs.",
    "There were 0 errors and 1 warning.",
    "The population is 1234567.",
    "He was born in 1999 and moved in 2005.",
    "In 2000 the company had 2010 employees.",
    "The year 1066 changed everything; so did 1900.",
    "Chapter 7 starts on page 101.",
    "It costs $5 or $1.50 or $0.99 today.",
    "The book costs 12$ and the pen 3.5$.",
    "We paid £7.14 for tea and £1 for milk.",
    "Meet me at 13:30 or 4:00 tomorrow.",
    "The train leaves at 12:05 and arrives at 23:59.",
    "Run 5km then walk 2.5m and rest 10min.",
    "The car drove 60km/h for 1h.",
    "Add 2tbsp of sugar and 1tsp of salt.",
    "It was -19°C outside, or 4°F.",
    "Pi is about 3.14159 and e is 2.718.",
    "Mix 1/2 cup with 3/4 cup and 2/3 spoon.",
    "Take 5/6 of it, or 3/2 of the rest, or 6/1.",
    "She finished 1st, he was 22nd, and they were 103rd.",
    "Lists: 1. apples 2. pears 3. plums",
    "The score was 3 - 2 and 5 + 5 = 10.",
    "Growth hit 45% this year, up from 7%.",
    "Call 555 0199 or dial 911.",
    "The 1990s were loud and the 80s were louder.",
    "Version 2.0 beats version 10.5 easily.",
    "He scored 100 points in 48 minutes.",
    "The distance is 3,000,000 meters.",
    "Room 404 is on floor 4.",
    "We need 7 tables for 45 guests at 6 pm.",
    "Dr. Smith met Mr. Jones and Mrs. Brown on St. James St.",
    "The NASA and FBI agents spoke to the CEO of IBM.",
    "I used an AI tool on my iOS device and my PC.",
    "The USA, the UK, and the EU signed the deal.",
    "e.g. this one, i.e. that one.",
    "Use JSON with conda and ChatGPT.",
    "McDonald's and iPhone and YouTube are brands.",
    "OK, the ASAP request came from HR via email.",
    "A is the first letter and a is an article.",
    "I am here, and I said A, B, and C.",
    "The TV and the DVD were in the den.",
    "I don't think you're right, but we'll see.",
    "She can't go because they've already left.",
    "It's the dog's bone and the boss's chair.",
    "The church's bell and the judge's ruling.",
    "Chris's car and James's house and the cat's toy.",
    "Let's go, y'all, it's o'clock time.",
    "The kids' toys and the students' books.",
    "Microsoft's revenue and Zelenskyy's speech.",
    "I read the book yesterday and I will read it again.",
    "Please record the record for me.",
    "We live here, and the show is live.",
    "Lead the team to the lead mine.",
    "Close the door, the store is close.",
    "I refuse to collect the refuse around here.",
    "The wind will wind the clock.",
    "Present the present to the president.",
    "Does he like the does in the forest?",
    "The bass player caught a bass.",
    "This is a complex problem in a complex.",
    "Use the tool; the use is clear.",
    "They object to the object.",
    "He wound the wound with a bandage.",
    "I will permit you to get a permit.",
    "The flibbertigibbet quizzed the zorblax.",
    "Supercalifragilisticexpialidocious is long.",
    "Kubernetes and Tensorflow run on Nvidia hardware.",
    "The xylophonist played arpeggios.",
    "Grzegorz and Siobhan visited Reykjavik.",
    "We bought a smartwatch and a dashcam.",
    "Blorptastic snazzlefrump wibblewobble.",
    "Her name is Anneliese and his is Thaddeus.",
    "This is a state-of-the-art, well-known, up-to-date tool.",
    "My mother-in-law has a twenty-one-year-old cat.",
    "It was a long-term, high-quality, cost-effective plan.",
    "The e-mail was re-sent by the co-op.",
    "A self-driving car took the x-ray.",
    "Hello!!! Is anyone there???",
    "What?! Really?! No way!!",
    "Well... I guess... maybe.",
    "Stop -- please -- stop.",
    "He said, \"Go home.\"",
    "She whispered: 'leave now'.",
    "Items: one; two; three.",
    "(Parenthetical remark) then more text.",
    "Rock & roll & blues.",
    "Hi, there , friend .",
    "“Smart quotes” and ‘single ones’ and don’t.",
    "The café served crème brûlée naïvely.",
    "Email me at test@example.com or #hashtag.",
    "Wow… that’s amazing—truly.",
    "Tabs\tand\nnewlines here.",
    "   leading and trailing spaces   ",
    "Hi.",
    "Go!",
    "Yes",
    "Oh?",
    "A.",
    "Hmm.",
    "Ok",
    "When the sun rose over the hills, the farmers walked out to the fields and began their work.",
    "Programming languages like Python, Java, and Rust each have their strengths.",
    "The museum, which opened in 1887, holds over 2500 paintings and 300 sculptures.",
    "If you can't find the answer, ask a friend, a teacher, or a librarian.",
    "Thunderstorms are expected tonight, with winds up to 40 mph.",
    "He looked at me and said, it's never too late to start again.",
    "Our flight was delayed by 2 hours, so we read books at gate B12.",
    "The quick brown fox jumps over the lazy dog, again and again and again.",
    "Is this the real life? Is this just fantasy?",
    "Doctors recommend 8 hours of sleep and 2 liters of water a day.",
    "The button on the little bottle is a cache of garçon food at the restaurant.",
    "He ate a croissant and some fiancé-style entrée, then called the human.",
    "Please give me a piece of cake, a bottle of water, and a kitten.",
    "Hello [laughs] there, bold text.",
    "Hello [S1] how are you? [S2] Fine, thanks.",
    "Is 42 the answer? 3.14 is pi, and I am 100% sure about $1,234.56 and £5.",
    "On 12/25 we had 1/2 a cake, 3/4 of a pie, and the 2nd, 3rd, 4th, 21st and 100th place.",
    "Mrs. Smith and Mr. Jones met Dr. Brown, Capt. Hook, Lt. Dan, Gen. Lee, Col. Mustard, Rev. Green and Sgt. Pepper.",
    "The Hon. Judge, Esq., of Acme Co. Ltd., Jr., met St. Louis, etc. btw.",
    "The years 2005, 2000, 1900, 1066 and 1999 were different; 2010 too.",
    "I have 1,000 apples and 2,500,000 oranges, 12345 pears, and 0 bananas.",
    "42",
    "!!!",
    "Ellipsis… and three dots... and a dash — and an en dash – here.",
    "He said (quietly) that \"it works\", didn't he?",
    "Hello, world. How are you today? I'm fine! Really; truly: yes.",
    "Wait.  Two  spaces   and a final one. ",
    "A very long sentence without any punctuation at all that just keeps going and going so that the chunker has to deal with a budget smaller than the sentence itself and keep it whole",
]

PUNCTUATION_CASES = ["Hello", "Hello.", "Hello…", "Hi!  ", "  Hi ", "What?!", "wow;", "done:", "", "   "]

# The chunker runs over the tokens of the most punctuated corpus lines at each of these budgets.
CHUNK_LINE_COUNT = 16
CHUNK_BUDGETS = [-3, 0, 1, 7, 25, 60, 150]

# Clause cases: realistic punctuation between words whose phonemes do not depend on their neighbours, so the engine
# test can phonemize them from a small lexicon and compare only the clause structure and marks.
CLAUSE_WORDS = ["cat", "Dog", "dog", "fox", "Fox", "bird", "Tree", "tree"]
CLAUSE_SEPS = [
    " ", " ", " ", ", ", ". ", "; ", ": ", "? ", "! ", "… ", "... ", " — ", " - ", " – ", "?! ", "!! ", ") ", " (",
    ", \"", ".\" ", "?\" ", "!\" ", ",\" ", " -- ", "; \"", "), ",
]
CLAUSE_ENDS = ["", ".", "?", "!", "…", "...", ",", ".\"", "?\"", ")", ".)"]
CLAUSE_CASES = 120

SAMPLE_RATE = 24000
BLOCK = SAMPLE_RATE // 100

# name, spec, only_edge, trail_sil. A spec is blocks of 10 ms (S silent, L loud, T just under the -50 dBFS threshold,
# U just over it, V a mix whose float rms is over it but whose integer rms is under it) or raw loud samples (R).
SILENCE_CASES = [
    ("edge_basic", "S50 L100 S50", True, 0),
    ("edge_short_kept", "S5 L100 S5", True, 0),
    ("edge_asymmetric", "S13 L40 S77", True, 0),
    ("prompt_basic", "S50 L300 S50", False, 200),
    ("pause_1500", "S20 L100 S150 L100 S20", False, 200),
    ("pause_3500", "S20 L100 S350 L100 S20", False, 0),
    ("pause_2000", "L100 S200 L100", False, 0),
    ("pause_1000", "L100 S100 L100", False, 0),
    ("pause_990", "L100 S99 L100", False, 0),
    ("pause_1010", "L100 S101 L100", False, 0),
    ("pause_1990", "L100 S199 L100", False, 0),
    ("two_pauses", "S30 L50 S300 L50 S120 L50 S30", False, 0),
    ("threshold_edges", "S30 U10 L40 T10 V10 S20", True, 0),
    ("threshold_integer_rms", "V40 L50 V40", True, 0),
    ("all_silent", "S300", True, 0),
    ("odd_length_down", "S20 L100 R5", True, 0),
    ("odd_length_up", "S20 L100 R15", True, 0),
    ("trail_only", "L100", True, 333),
]

# name, wave lengths, fade seconds, sample rate
CROSSFADE_CASES = [
    ("three", [40, 6, 25], 0.1, 100),
    ("single", [50], 0.1, 100),
    ("short_pair", [3, 3], 0.1, 100),
    ("exact_fade", [10, 5], 0.1, 100),
    ("empty_first", [0, 7, 4], 0.1, 100),
    ("no_fade", [8, 8], 0.0, 100),
]


def build_clause_text(rng):
    parts = []
    for _ in range(rng.randint(3, 9)):
        parts.append(rng.choice(CLAUSE_WORDS))
        parts.append(rng.choice(CLAUSE_SEPS))
    text = "".join(parts).rstrip()
    end = rng.choice(CLAUSE_ENDS)
    return text + end if text[-1].isalnum() else text


def build_signal(spec):
    parts = []
    n = 0
    for token in spec.split():
        kind, count = token[0], int(token[1:])
        size = count if kind == "R" else count * BLOCK
        idx = np.arange(n, n + size, dtype=np.int64)
        if kind in ("L", "R"):
            v = (idx * 7919) % 2001 + 2000
        elif kind == "S":
            v = (idx * 131) % 61 - 30
        elif kind == "T":
            v = np.full(size, 103, dtype=np.int64)
        elif kind == "U":
            v = np.full(size, 104, dtype=np.int64)
        elif kind == "V":
            v = np.where((idx - n) % BLOCK < 192, 104, 103)
        else:
            raise ValueError(token)
        parts.append(v)
        n += size
    return np.concatenate(parts).astype(np.int16)


def build_wave(index, length):
    i = np.arange(length, dtype=np.int64)
    return ((i * 37 + index * 101) % 211).astype(np.float32) / np.float32(211) - np.float32(0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    logging.disable(logging.CRITICAL)
    sys.path.insert(0, args.upstream)
    import torch
    from zipvoice.tokenizer.normalizer import EnglishTextNormalizer
    from zipvoice.tokenizer.tokenizer import EmiliaTokenizer
    from zipvoice.utils.infer import add_punctuation, chunk_tokens_punctuation, cross_fade_concat, remove_silence

    tokenizer = EmiliaTokenizer(token_file=args.tokens)
    normalizer = EnglishTextNormalizer()

    lines = []
    for text in CORPUS:
        spaced = text.replace("\n", " ")
        try:
            punct = add_punctuation(spaced)
        except IndexError:
            punct = None
        entry = {"text": text, "punct": punct}
        if punct is not None:
            mapped = tokenizer.preprocess_text(punct)
            segments = tokenizer.get_segment(mapped)
            tokens = tokenizer.texts_to_tokens([punct])[0]
            ids = tokenizer.tokens_to_token_ids([tokens])[0]
            entry["segments"] = [[seg, lang] for seg, lang in segments]
            entry["norm"] = normalizer.normalize(segments[0][0]) if len(segments) == 1 and segments[0][1] == "en" else None
            entry["tokens"] = "".join(tokens)
            entry["ntok"] = len(tokens)
            entry["ids"] = ",".join(str(i) for i in ids)
        lines.append(entry)

    punctuation = []
    for text in PUNCTUATION_CASES:
        try:
            out = add_punctuation(text)
        except IndexError:
            out = None
        punctuation.append({"in": text, "out": out})

    marks = ".!?;:,"
    candidates = [i for i, e in enumerate(lines) if e.get("tokens") and sum(e["tokens"].count(m) for m in marks) >= 3]
    chunk_lines = candidates[:: max(1, len(candidates) // CHUNK_LINE_COUNT)][:CHUNK_LINE_COUNT]
    chunks = []
    for line in chunk_lines:
        tokens = list(lines[line]["tokens"])
        for budget in CHUNK_BUDGETS:
            chunks.append({
                "line": line,
                "max": budget,
                "chunks": ["".join(c) for c in chunk_tokens_punctuation(tokens, max_tokens=budget)],
            })

    crossfade = []
    for name, lens, fade, sr in CROSSFADE_CASES:
        waves = [torch.from_numpy(build_wave(j, n))[None] for j, n in enumerate(lens)]
        out = cross_fade_concat(waves, fade_duration=fade, sample_rate=sr)
        crossfade.append({"name": name, "lens": lens, "fade": fade, "sr": sr, "out": out[0].tolist()})

    silence = []
    for name, spec, only_edge, trail in SILENCE_CASES:
        pcm = build_signal(spec)
        wav = torch.from_numpy(pcm.astype(np.float32) / 32768.0)[None]
        out = remove_silence(wav, SAMPLE_RATE, only_edge=only_edge, trail_sil=trail)
        out_pcm = np.rint(out[0].numpy() * 32768.0).astype(np.int16)
        silence.append({
            "name": name, "spec": spec, "only_edge": only_edge, "trail": trail,
            "in_len": int(pcm.shape[0]), "out_len": int(out_pcm.shape[0]),
            "sha1": hashlib.sha1(out_pcm.tobytes()).hexdigest(),
        })

    from piper_phonemize import phonemize_espeak

    lexicon = {}
    for word in CLAUSE_WORDS:
        lexicon[word.lower()] = "".join("".join(s) for s in phonemize_espeak(word.lower(), "en-us"))
    rng = random.Random(20261015)
    clauses = []
    while len(clauses) < CLAUSE_CASES:
        text = build_clause_text(rng)
        tokens = tokenizer.texts_to_tokens([text])[0]
        clauses.append({"text": text, "tokens": "".join(tokens)})

    rev = subprocess.run(["git", "-C", args.upstream, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    fixture = {
        "source": {
            "zipvoice": rev,
            "piper_phonemize": importlib.metadata.version("piper_phonemize"),
            "pydub": importlib.metadata.version("pydub"),
            "inflect": importlib.metadata.version("inflect"),
        },
        "lines": lines,
        "lexicon": lexicon,
        "clauses": clauses,
        "add_punctuation": punctuation,
        "chunks": chunks,
        "crossfade": crossfade,
        "silence": silence,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(fixture, f, ensure_ascii=False, indent=1)
    print(f"{len(lines)} lines, {len(clauses)} clause cases, {len(chunks)} chunk cases, {len(crossfade)} cross-fades, "
          f"{len(silence)} silence cases")


if __name__ == "__main__":
    main()
