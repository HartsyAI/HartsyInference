"""Dump the upstream GPT-SoVITS English text frontend as a parity fixture for the engine port.

Runs the exact upstream code path (GPT_SoVITS/text/english.py + cleaner.py + en_normalization/expend.py, and the
TTS_infer_pack text preprocessing with its cut methods) over a fixed corpus and writes, per line, the normalized
text, the phones and the v2 symbol ids. The engine test (GptSoVitsEnglishFrontendParityTests) asserts exact
equality against this file.

Usage (reference venv with g2p_en, nltk + averaged_perceptron_tagger_eng, inflect, wordsegment):
    python tools/parity/gptsovits_frontend_dump.py \
        --upstream /path/to/GPT-SoVITS --out tests/HartsyInference.Audio.Tests/Fixtures/gptsovits_en_frontend_parity.json

Also used offline to export the dictionaries and models the engine embeds (--export-resources DIR).
"""

import argparse
import collections
import contextlib
import importlib.metadata
import io
import json
import os
import sys
import types

MATRIX = [
    "Yes.",
    "Okay, see you.",
    "The weather is nice today, so we might take a walk.",
    "After the meeting ended, the team gathered in the kitchen to talk about the new product launch, the budget, and the schedule for the coming quarter.",
    "Our water bill arrived late this month, so I called the office to ask why. The clerk said a meter reading was missed in March. She promised to fix the record and send a corrected statement within two weeks. I thanked her, hung up, and wrote the date on my calendar so I would remember to check. Last time this happened it took three phone calls, so I am hoping this one goes faster.",
    "On March 3rd, 2026 at 4:30 PM, 1,250 people paid $99.95 each.",
    "Wait... what?! No -- really; (yes), 'quoted' & \"double\" -- ok?",
    "No no no no no no no no no no no no.",
]

REFERENCE = ["The quick brown fox jumps over the lazy dog near the river bank."]

EXTRA = [
    # numbers, years, money, times, measurements, fractions, ordinals
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
    # abbreviations, acronyms, casing
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
    # contractions and possessives
    "I don't think you're right, but we'll see.",
    "She can't go because they've already left.",
    "It's the dog's bone and the boss's chair.",
    "The church's bell and the judge's ruling.",
    "Chris's car and James's house and the cat's toy.",
    "Let's go, y'all, it's o'clock time.",
    "The kids' toys and the students' books.",
    "Microsoft's revenue and Zelenskyy's speech.",
    # homographs
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
    # OOV words and compounds
    "The flibbertigibbet quizzed the zorblax.",
    "Supercalifragilisticexpialidocious is long.",
    "Kubernetes and Tensorflow run on Nvidia hardware.",
    "The xylophonist played arpeggios.",
    "Grzegorz and Siobhan visited Reykjavik.",
    "We bought a smartwatch and a dashcam.",
    "Blorptastic snazzlefrump wibblewobble.",
    "Her name is Anneliese and his is Thaddeus.",
    # hyphenated words
    "This is a state-of-the-art, well-known, up-to-date tool.",
    "My mother-in-law has a twenty-one-year-old cat.",
    "It was a long-term, high-quality, cost-effective plan.",
    "The e-mail was re-sent by the co-op.",
    "A self-driving car took the x-ray.",
    # punctuation runs and quotes
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
    # short lines
    "Hi.",
    "Go!",
    "Yes",
    "Oh?",
    "A.",
    "Hmm.",
    "Ok",
    # longer prose
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
]

LTS_WORDS = [
    "zorblax", "flibbertigibbet", "snazzlefrump", "blorptastic", "wibblewobble", "kubernetes", "grzegorz", "siobhan",
    "reykjavik", "zelenskyy", "anneliese", "thaddeus", "xylophonist", "arpeggios", "dashcam", "smartwatch",
    "chatgpt", "huggingface", "pytorch", "numpy", "tensorflow", "nvidia", "qwerty", "zxcvbn", "aaaa", "mmmm",
    "brrr", "pfft", "schmidt", "tschüss", "o'neill", "co-op", "vowelless", "quixotic", "ptolemy", "gnocchi",
    "worcestershire", "llanfair", "przemysl", "uzbekistan", "kazakhstani", "tsunamis", "yttrium", "zyzzyva",
    "floccinaucinihilipilification", "antidisestablishmentarianism", "a", "ab", "abc", "abcdefghijklmnopqrstuvwxyz",
]

CUT_TEXTS = MATRIX + REFERENCE + [
    "Hi. This is a short one. Another sentence here. And one more. Then a fifth. And a sixth!",
    "Go. Now. Please, quickly, before it is too late, because the rain is coming and the roads will flood.",
    "Okay",
    "This line has no final punctuation",
    "Price is 3.50 dollars. Next.",
]


def stub_heavy_modules():
    """TextPreprocessor imports torch/transformers/LangSegmenter at module load; the English path uses none of them."""
    torch = types.ModuleType("torch")
    torch.device = object
    torch.Tensor = object
    sys.modules.setdefault("torch", torch)
    tf = types.ModuleType("transformers")
    tf.AutoModelForMaskedLM = object
    tf.AutoTokenizer = object
    sys.modules["transformers"] = tf
    ls = types.ModuleType("text.LangSegmenter")
    ls.LangSegmenter = object
    sys.modules["text.LangSegmenter"] = ls
    sys.modules["text.chinese"] = types.ModuleType("text.chinese")
    i18n_pkg = types.ModuleType("tools.i18n.i18n")
    i18n_pkg.I18nAuto = lambda **kw: (lambda s: s)
    i18n_pkg.scan_language_list = lambda: []
    sys.modules["tools"] = types.ModuleType("tools")
    sys.modules["tools.i18n"] = types.ModuleType("tools.i18n")
    sys.modules["tools.i18n.i18n"] = i18n_pkg
    tqdm = types.ModuleType("tqdm")
    tqdm.tqdm = lambda x: x
    sys.modules.setdefault("tqdm", tqdm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default="/mnt/model-storage/scratch/tts-ref/src/GPT-SoVITS")
    ap.add_argument("--out", required=False)
    ap.add_argument("--export-resources", default=None, help="Write the engine's embedded resources here.")
    ap.add_argument("--stats", action="store_true", help="Print per-branch hit counts for the corpus.")
    ap.add_argument("--stress", type=int, default=0, help="Instead, write N random lines (JSONL) to --stress-out.")
    ap.add_argument("--stress-out", default=None)
    args = ap.parse_args()

    root = os.path.join(args.upstream, "GPT_SoVITS")
    sys.path.insert(0, root)
    os.chdir(root)
    os.environ["version"] = "v2"
    venv_nltk = os.path.join(sys.prefix, "nltk_data")
    import nltk

    if os.path.isdir(venv_nltk):
        nltk.data.path.insert(0, venv_nltk)

    with contextlib.redirect_stdout(sys.stderr):
        from text import cleaner, cleaned_text_to_sequence
        import text.english as english

    hits = collections.Counter()
    g = english._g2p
    orig_qry = g.qryword
    orig_predict = g.predict

    def predict(word):
        hits["predict"] += 1
        return orig_predict(word)

    g.predict = predict
    orig_segment = english.wordsegment.segment

    def segment(word):
        comps = orig_segment(word)
        hits["wordsegment_split" if len(comps) > 1 else "wordsegment_single"] += 1
        return comps

    english.wordsegment.segment = segment

    def qryword(o_word):
        word = o_word.lower()
        if len(word) > 1 and word in g.cmu:
            hits["cmu"] += 1
        elif o_word.istitle() and word in g.namedict:
            hits["namedict"] += 1
        return orig_qry(o_word)

    g.qryword = qryword

    if args.export_resources:
        export_resources(args.export_resources, english, g)
        return

    def clean(t):
        with contextlib.redirect_stdout(sys.stderr):
            phones, _, norm = cleaner.clean_text(t, "en", "v2")
        return phones, norm

    if args.stress:
        corpus = stress_corpus(args.stress)
        with io.open(args.stress_out, "w", encoding="utf-8") as f:
            for t in corpus:
                phones, norm = clean(t)
                f.write(json.dumps({"text": t, "norm": norm, "phones": " ".join(phones)}, ensure_ascii=False) + "\n")
        print(f"wrote {len(corpus)} stress lines to {args.stress_out}")
        return

    lines = []
    for t in MATRIX + REFERENCE + EXTRA:
        phones, norm = clean(t)
        ids = cleaned_text_to_sequence(phones, "v2")
        lines.append({"text": t, "norm": norm, "phones": " ".join(phones), "ids": ids})

    stub_heavy_modules()
    # TTS_infer_pack/__init__.py imports the whole TTS stack; load just the two text modules from source.
    import importlib.util

    pkg = types.ModuleType("TTS_infer_pack")
    pkg.__path__ = [os.path.join(root, "TTS_infer_pack")]
    sys.modules["TTS_infer_pack"] = pkg

    def load(name):
        spec = importlib.util.spec_from_file_location(f"TTS_infer_pack.{name}", os.path.join(root, "TTS_infer_pack", f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod

    with contextlib.redirect_stdout(sys.stderr):
        seg = load("text_segmentation_method")
        TextPreprocessor = load("TextPreprocessor").TextPreprocessor

    tp = TextPreprocessor.__new__(TextPreprocessor)

    def phones_for_segment(t, final=False):
        # TextPreprocessor.get_phones_and_bert for language "en", minus the (all-zero) BERT tensor.
        import re

        t2 = re.sub(r" {2,}", " ", t)
        with contextlib.redirect_stdout(sys.stderr):
            ph, _, norm = cleaner.clean_text(t2, "en", "v2")
        ids = cleaned_text_to_sequence(ph, "v2")
        if not final and len(ids) < 6:
            return phones_for_segment("." + t2, final=True)
        return ids, norm

    segments = []
    for method in ["cut0", "cut1", "cut5"]:
        for t in CUT_TEXTS:
            with contextlib.redirect_stdout(sys.stderr):
                pre = tp.replace_consecutive_punctuation(t)
                segs = tp.pre_seg_text(pre, "en", method)
            out = []
            for s in segs:
                ids, norm = phones_for_segment(s)
                if norm == "":
                    continue
                out.append({"text": s, "norm": norm, "ids": ids})
            segments.append({"method": method, "text": t, "segments": out})

    prompts = []
    for t in REFERENCE + ["Hello there", "Okay"]:
        p = t.strip("\n")
        if p[-1] not in seg.splits:
            p += "."
        ids, norm = phones_for_segment(p)  # TTS.run: segment_and_extract_feature_for_text, short-text retry included
        prompts.append({"text": t, "prompt": p, "norm": norm, "ids": ids})

    numbers = []
    from text.en_normalization.expend import normalize

    for n in list(range(0, 130)) + list(range(130, 3200, 37)) + [1000, 1001, 1100, 1999, 2000, 2001, 2009, 2010,
                                                                   2099, 2100, 2999, 3000, 10000, 12345, 100000,
                                                                   1000000, 1234567890]:
        numbers.append({"text": f"n {n} x", "norm": normalize(f"n {n} x")})

    # The neural letter-to-sound model on its own, so float drift shows up as a named word rather than a sentence.
    lts = [{"word": w, "phones": " ".join(orig_predict(w))} for w in LTS_WORDS]

    fixture = {
        "source": "GPT-SoVITS " + git_head(args.upstream),
        "lts": lts,
        "versions": {p: importlib.metadata.version(p) for p in ["g2p_en", "nltk", "inflect", "wordsegment"]},
        "lines": lines,
        "numbers": numbers,
        "segments": segments,
        "prompts": prompts,
    }
    if args.out:
        # One record per line keeps the fixture reviewable in diffs.
        with io.open(args.out, "w", encoding="utf-8") as f:
            f.write("{\n")
            keys = list(fixture)
            for k_i, k in enumerate(keys):
                v = fixture[k]
                f.write(f" {json.dumps(k)}: ")
                if isinstance(v, list):
                    f.write("[\n" + ",\n".join("  " + json.dumps(r, ensure_ascii=False) for r in v) + "\n ]")
                else:
                    f.write(json.dumps(v, ensure_ascii=False))
                f.write(",\n" if k_i < len(keys) - 1 else "\n")
            f.write("}\n")
        print(f"wrote {len(lines)} lines, {len(numbers)} numbers, {len(segments)} segment cases to {args.out}")
    if args.stats:
        print(dict(hits))


def stress_corpus(n, seed=1234):
    """Seeded random lines mixing frequent words, homographs, capitals, invented words, numbers, money, times and
    punctuation: a broad sweep beyond the committed fixture (run with GPTSOVITS_EN_STRESS_FIXTURE)."""
    import random
    import wordsegment
    from g2p_en.g2p import construct_homograph_dictionary

    rng = random.Random(seed)
    common = [w for w, _ in sorted(wordsegment.UNIGRAMS.items(), key=lambda kv: -kv[1])[:6000] if w.isalpha()]
    homographs = sorted(construct_homograph_dictionary())
    letters = "abcdefghijklmnopqrstuvwxyz"
    puncts = [",", ".", "!", "?", ";", ":", "...", " -", " --", "!?", '"', "'", "(", ")", "&", "%"]

    def token():
        r = rng.random()
        if r < 0.55:
            w = rng.choice(common)
        elif r < 0.68:
            w = rng.choice(homographs)
        elif r < 0.76:
            w = "".join(rng.choice(letters) for _ in range(rng.randint(4, 12)))
        elif r < 0.80:
            w = rng.choice(common) + "'s"
        elif r < 0.83:
            w = rng.choice(common) + "-" + rng.choice(common)
        elif r < 0.86:
            w = "".join(rng.choice(letters.upper()) for _ in range(rng.randint(2, 5)))
        elif r < 0.90:
            w = str(rng.choice([rng.randint(0, 99), rng.randint(100, 9999), rng.randint(10000, 10**7)]))
        elif r < 0.92:
            w = rng.choice(["$", "£"]) + str(rng.randint(0, 999)) + rng.choice(["", "." + str(rng.randint(0, 99))])
        elif r < 0.94:
            w = f"{rng.randint(0, 23)}:{rng.randint(0, 59):02d}"
        elif r < 0.96:
            w = str(rng.randint(1, 200)) + rng.choice(["st", "nd", "rd", "th", "km", "m", "s", "h", "%", "/4"])
        else:
            w = str(rng.randint(0, 99)) + "." + str(rng.randint(0, 999))
        if rng.random() < 0.15:
            w = w.capitalize()
        return w

    lines = []
    for _ in range(n):
        parts = []
        for _ in range(rng.randint(1, 18)):
            parts.append(token())
            if rng.random() < 0.15:
                parts[-1] += rng.choice(puncts)
        line = " ".join(parts)
        if rng.random() < 0.8:
            line = line[0].upper() + line[1:] + rng.choice([".", "?", "!", ""])
        lines.append(line)
    return lines


def git_head(path):
    try:
        import subprocess

        return subprocess.check_output(["git", "-C", path, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def export_resources(out_dir, english, g):
    """Exports what the engine embeds: the merged CMU dict exactly as en_G2p sees it, the name dict, the g2p_en
    homograph table, the g2p_en GRU weights and the nltk tagger tables. Each is plain text or raw float32 so the
    C# side needs no pickle/npz reader."""
    import gzip
    import numpy as np

    os.makedirs(out_dir, exist_ok=True)

    def wtext(name, rows):
        with gzip.open(os.path.join(out_dir, name), "wt", encoding="utf-8", compresslevel=9) as f:
            for r in rows:
                f.write(r + "\n")

    # Merged dict (cmudict.rep + cmudict-fast.rep + hot + deletions), first pronunciation only (all that is used).
    # "word<TAB>phones"; phones keep upstream's empty entries (cmudict-fast lines with a double space), which the
    # symbol filter drops later.
    wtext("gptsovits_en_dict.txt.gz", (w + "\t" + " ".join(p[0]) for w, p in sorted(g.cmu.items())))
    wtext("gptsovits_en_names.txt.gz", (w + "\t" + " ".join(p[0]) for w, p in sorted(g.namedict.items())))
    wtext("gptsovits_en_homographs.txt.gz",
          (w + "|" + " ".join(a) + "|" + " ".join(b) + "|" + pos for w, (a, b, pos) in sorted(g.homograph2features.items())))
    # g2p_en GRU: graphemes, phonemes, then each tensor as name, shape, float32 little-endian.
    with gzip.open(os.path.join(out_dir, "g2p_en_lts.bin.gz"), "wb", compresslevel=9) as f:
        def wstr(s):
            b = s.encode("utf-8")
            f.write(len(b).to_bytes(4, "little"))
            f.write(b)

        wstr("\t".join(g.graphemes))
        wstr("\t".join(g.phonemes))
        names = ["enc_emb", "enc_w_ih", "enc_w_hh", "enc_b_ih", "enc_b_hh", "dec_emb", "dec_w_ih", "dec_w_hh",
                 "dec_b_ih", "dec_b_hh", "fc_w", "fc_b"]
        for n in names:
            a = np.ascontiguousarray(g.variables[n], dtype="<f4")
            wstr(n)
            f.write(a.ndim.to_bytes(4, "little"))
            for d in a.shape:
                f.write(int(d).to_bytes(4, "little"))
            f.write(a.tobytes())
    # nltk averaged perceptron (eng): weights as "feature\ttag\tweight" with repr floats, tagdict, classes.
    from nltk.tag.perceptron import PerceptronTagger

    tagger = PerceptronTagger()
    rows = []
    for feat, wts in sorted(tagger.model.weights.items()):
        for tag, w in sorted(wts.items()):
            rows.append(f"{feat}\t{tag}\t{w!r}")
    wtext("nltk_perceptron_weights.txt.gz", rows)
    wtext("nltk_perceptron_tagdict.txt.gz", (f"{w}\t{t}" for w, t in sorted(tagger.tagdict.items())))
    wtext("nltk_perceptron_classes.txt.gz", sorted(tagger.classes))
    # wordsegment unigram/bigram counts as shipped.
    import wordsegment

    wtext("wordsegment_unigrams.txt.gz", (f"{w}\t{c!r}" for w, c in wordsegment.UNIGRAMS.items()))
    wtext("wordsegment_bigrams.txt.gz", (f"{w}\t{c!r}" for w, c in wordsegment.BIGRAMS.items()))
    for n in sorted(os.listdir(out_dir)):
        print(n, os.path.getsize(os.path.join(out_dir, n)))


if __name__ == "__main__":
    main()
