"""Records the upstream IndexTTS text frontend (1.5, 2.0 and 2.5) for the C# parity tests.

For every corpus line and version this runs the exact upstream text path and writes, per stage, what upstream
produced: the normalizer output, every string upstream handed to the wetext English/Chinese FST normalizer together
with the FST's answer, the tokenizer pieces, the segments and the token ids the GPT finally embeds.

    PYTHONHASHSEED=0 fe-venv/bin/python tools/parity/indextts_frontend_dump.py \
        <index-tts checkout> <models root> tests/HartsyInference.Audio.Tests/Fixtures/Frontend/indextts_frontend.json

<index-tts checkout> is github.com/index-tts/index-tts; <models root> holds IndexTeam--IndexTTS-1.5, -2 and -2.5.

Upstream paths reproduced (file:line in the checkout used for the fixture, recorded in the output):
- 1.5 (indextts/infer.py): TextNormalizer() -> TextTokenizer(bpe.model).tokenize(text) -> split_segments(120)
  -> convert_tokens_to_ids per segment.
- 2.0 (indextts/infer_v2.py): the same with TextNormalizer(enable_glossary=True) (no glossary.yaml ships, so empty).
- 2.5 (indextts/infer_v2_5.py, infer_generator): clean_pattern pre-pass, normalize (en/zh/zhen), lower(),
  apply_pronunciation_annotations, special-token uppercasing, split_text_by_tokens, encode("<|lang|> " + segment)
  and the stop pad. Those lines are copied below verbatim because infer_v2_5 cannot be imported without the
  acoustic model stack.
All versions then go through UnifiedVoice.prepare_gpt_inputs, whose valid_mask drops every id equal to the start (0)
or stop (1) text token before re-wrapping; "gpt_ids" records that masked list.

Upstream selects WeTextProcessing on Linux and wetext elsewhere; this script uses wetext (the FSTs the C# runtime
loads) with the exact non-Linux constructor arguments from TextNormalizer.load().
"""
import json
import os
import re
import subprocess
import sys
import types
from importlib.metadata import version

CORPUS_EN = [
    # Required lines.
    "Yes.",
    "Okay, see you.",
    "The weather is nice today, so we might take a walk.",
    "After the meeting ended, the team gathered in the kitchen to talk about the new product launch, the budget, "
    "and the schedule for the coming quarter.",
    "On March 3rd, 2026 at 4:30 PM, 1,250 people paid $99.95 each.",
    "Wait... what?! No -- really; (yes), 'quoted' & \"double\" -- ok?",
    "No no no no no no no no no no no no.",
    "Every morning before the city wakes, Maria walks her old dog along the river, past the shuttered bakery and "
    "the quiet docks. She likes the cold air, the gulls arguing overhead, and the way the light slowly turns the "
    "water from grey to gold. By the time she returns, her coffee is waiting.",
    "The quick brown fox jumps over the lazy dog near the river bank.",
    # Short openings.
    "No.",
    "Hi!",
    "Hello there.",
    "Sure, why not?",
    "Well, that's that.",
    "Thanks!",
    "Really?",
    "Good morning, everyone.",
    "OK.",
    "a",
    "I",
    # Contractions.
    "It's raining, and that's why he's late.",
    "What's the plan? Where's the money? Who's there?",
    "I can't, won't, and shouldn't do it.",
    "They're sure we've met, but I'd say you'll forget.",
    "How's it going? Which's the best?",
    "There's a cat; here's a dog.",
    "She’s here and it’s fine.",
    "This's odd, and she's right.",
    # Tech terms and hyphens.
    "GPT-5-nano is the smallest and fastest variant in the GPT-5 model family.",
    "such as XTTS, CosyVoice2, Fish-Speech, and F5-TTS",
    "We compared F5-TTS with CosyVoice-2 and IndexTTS-2.",
    "The well-known state-of-the-art model is fast.",
    "Use Wi-Fi 6E or USB-C 3.2 ports.",
    "Version 2.0 of the API is out.",
    "Here are some highly-rated M.2 NVMe SSDs: Samsung 9100 PRO PCIe 5.0 SSD M.2, $139.99",
    "we dive deep into the showdown between DisplayPort 1.4 and HDMI 2.1 to determine which is the best choice for "
    "gaming enthusiasts",
    "Run C++ and C# builds with CMake.",
    "Visit https://example.com or email me at john@example.com.",
    # Names.
    "Joseph Gordon-Levitt is an American actor.",
    "Dr. Smith met Mr. Jones and Mrs. Brown at St. Mary's.",
    "Tim Cook and Satya Nadella spoke at the event.",
    # Numbers, dates, times, money.
    "See you at 8:00 AM",
    "Couting down 3, 2, 1, go!",
    "This sales for 2.5% off, only $12.5.",
    "We sold 1,234,567 units, up 15%.",
    "I'm 25 years old and I've been here since 2010.",
    "Call 555-123-4567 before 9 p.m.",
    "The meeting is on 2024-05-06.",
    "She was born on 12/25/1990.",
    "He finished 1st, she finished 2nd and they finished 3rd.",
    "It costs €45 or £30.",
    "The temperature is -3.5 degrees.",
    "Pi is about 3.14159.",
    "Add 1/2 cup of sugar.",
    "In 1066 the Normans arrived.",
    "World War 2 ended in 1945.",
    "Chapter 11 starts on page 304.",
    "The 21st century began in 2001.",
    "Room 101, floor 7.",
    "100",
    "1,250.",
    "3.14",
    # Abbreviations and acronyms.
    "The U.S. and the U.K. signed the deal.",
    "NASA and the FBI work with the CIA.",
    "e.g. apples, i.e. fruit, etc.",
    "Mt. Everest is 8,849 m tall.",
    "The CEO of Apple Inc. spoke.",
    # Punctuation.
    "Hello\nworld.",
    "Wait—what?",
    "She said “hello” and left.",
    "(Parenthetical) remarks [like this] matter.",
    "Hmm... let me think... okay.",
    "Stop! Go! Stop!",
    "Is it 5:30? Yes: it is.",
    "Approx ~10 items.",
    "Emojis \U0001F600 are fun.",
    # Pinyin with tones (letters plus tone digit route to the Chinese branch).
    "xuan4 is pinyin",
    "have you ever been to beta2?",
    "The word is zhong4 yao4.",
    # Pronunciation annotations.
    "I am <going|G OW1 . IH0 NG> home.",
    "<live|L IH1 V> music is fun.",
    # Long line (forces 1.5/2.0 and 2.5 segmentation).
    "This is a very long sentence that keeps going and going without ever really stopping because we want to "
    "exercise the segmentation logic, and it has commas, plenty of them, so that the splitter has somewhere to cut, "
    "but it also has many words with no punctuation at all which forces the hard limit to kick in eventually when "
    "the clause grows past the maximum number of tokens allowed per segment in a single generation pass of the model "
    "which is one hundred and twenty tokens by default so we need quite a lot of text here to get there safely.",
    "One. Two. Three. Four. Five. Six. Seven. Eight. Nine. Ten. Eleven. Twelve. Thirteen. Fourteen. Fifteen. Sixteen. "
    "Seventeen. Eighteen. Nineteen. Twenty. Twenty-one. Twenty-two. Twenty-three. Twenty-four. Twenty-five. "
    "Twenty-six. Twenty-seven. Twenty-eight. Twenty-nine. Thirty. Thirty-one. Thirty-two. Thirty-three. Thirty-four.",
    "<going|G OW1 . IH0 NG> " + "word " * 130 + "end.",
]

CORPUS_ZH = [
    "IndexTTS 正式发布1.0版本了，效果666",
    "晕XUAN4是一种GAN3觉",
    "我爱你！",
    "“我爱你”的英语是“I love you”",
    "2.5平方电线",
    "共465篇，约315万字",
    "2002年的第一场雪，下在了2003年",
    "速度是10km/h",
    "现在是北京时间2025年01月11日 20:00",
    "电话：135-4567-8900",
    "他这条视频点赞3000+，评论1000+，收藏500+",
    "受不liao3你了",
    "“衣裳”不读衣chang2，而是读衣shang5",
    "最zhong4要的是：不要chong2蹈覆辙",
    "不zuo1死就不会死",
    "8:00 AM 开会",
    "数到3就开始：1、2、3",
    "5G网络是4G网络的升级版，2G网络是3G网络的前身",
    "苹果于2030/1/2发布新 iPhone 2X 系列手机，最低售价仅 ¥12999",
    "这酒...里...有毒...",
    "只有,,,才是最好的",
    "babala2是什么？",
    "用beta1测试",
    "今天是个好日子 it's a good day",
    "GPT-5-Nano 是 GPT-5 模型家族中最小且速度最快的变体",
    "2025/09/08 IndexTTS-2 全球发布",
    "约瑟夫·高登-莱维特（Joseph Gordon-Levitt is an American actor）",
    "克里斯托弗·诺兰执导了《盗梦空间》。",
    "蒲蒬西·唐纳德·库克（英文名：Timothy Donald Cook），"
    "通称蒂姆·库克（Tim Cook），现任苹果公司首席"
    "执行官。",
    "我要去<行|XING2>动。",
    "你好，世界！今天天气很好。",
    "他说：“明天见。”",
    "这个产品售价$99，特价～",
    "一二三，，，四五六……",
    "注意；分号、顿号【括号】「引号」",
    "我的邮箱是 john@example.com",
    "user123@mail.com",
    "在 2023 年，我们卖了 3.5 万台。",
    "中文和English混合的句子, with numbers 42.",
    "Hello 你好 world 世界",
    "そうですね、ほんとに。",
    "안녕하세요 세계",
    "张三和李四在北京见面。",
    "今天是2024年12月25日，温度-5℃。",
    "《盗梦空间》是由美国华纳兄弟影片公司出品的"
    "电影，由克里斯托弗·诺兰执导并编剧，莱昂"
    "纳多·迪卡普里奥、玛丽昂·歌迪亚、约瑟夫"
    "·高登-莱维特、艾利奥特·佩吉、汤姆·哈迪"
    "等联袂主演，2010年7月16日在美国上映，2010年9月1日"
    "在中国内地上映，2020年8月28日在中国内地重映。"
    "影片剧情游走于梦境与现实之间，被定义为“"
    "发生在意识结构内的当代动作科幻片”，讲述"
    "了由莱昂纳多·迪卡普里奥扮演的造梦师，带"
    "领特工团队进入他人梦境，从他人的潜意识中"
    "盗取机密，并重塑他人梦境的故事。",
]

PRONUNCIATION_ANNOTATION_PATTERN = re.compile(r'<([^|>\n]+)\|([^>\n]+)>')  # infer_v2_5.py:39
SPLIT_PROTECTED_PATTERN = re.compile(r'<\|SPECIAL_TOKEN_\d+\|>.*?<\|SPECIAL_TOKEN_\d+\|>')  # infer_v2_5.py:411


def is_kana(s):  # infer_v2_5.py:44
    hira = re.compile(r'^[぀-ゟ]+$')
    kata = re.compile(r'^[゠-ヿ]+$')
    if hira.fullmatch(s):
        return True
    elif kata.fullmatch(s):
        return True
    return False


def apply_pronunciation_annotations(text):  # infer_v2_5.py:53
    def _replace(match):
        word = match.group(1)
        pronunciation = match.group(2).upper()
        has_chinese = bool(re.search(r"[一-鿿]", word))
        if is_kana(pronunciation):
            return f' {pronunciation} '
        else:
            token = 'SPECIAL_TOKEN_2' if has_chinese else 'SPECIAL_TOKEN_1'
            return f'<|{token}|>{pronunciation}<|{token}|>'
    return PRONUNCIATION_ANNOTATION_PATTERN.sub(_replace, text)


class SplitV25:
    """infer_v2_5.py:413-464 (_token_len, _split_atomic_pieces, split_text_by_tokens), self.gpt replaced by capacity."""

    def __init__(self, tokenizer, capacity):
        self.tokenizer = tokenizer
        self.capacity = capacity

    def _token_len(self, text):
        return len(self.tokenizer.encode(text, allowed_special='all'))

    def _split_atomic_pieces(self, text):
        pieces, pos = [], 0
        for m in SPLIT_PROTECTED_PATTERN.finditer(text):
            if m.start() > pos:
                pieces.append((text[pos:m.start()], False))
            pieces.append((m.group(0), True))
            pos = m.end()
        if pos < len(text):
            pieces.append((text[pos:], False))
        return pieces

    def split_text_by_tokens(self, text, max_tokens, lang_prefix=""):
        capacity = self.capacity
        budget = min(max_tokens, capacity - 2) - self._token_len(lang_prefix)
        budget = max(1, budget)
        if self._token_len(text) <= budget:
            return [text]

        chunks = []
        for piece, atomic in self._split_atomic_pieces(text):
            if atomic:
                chunks.append(piece)
                continue
            for part in re.split(r'(?<=[，。！？、；：,\.!\?;:\n])', piece):
                if not part:
                    continue
                if self._token_len(part) <= budget:
                    chunks.append(part)
                    continue
                current = ""
                for ch in part:
                    if current and self._token_len(current + ch) > budget:
                        chunks.append(current)
                        current = ch
                    else:
                        current += ch
                if current:
                    chunks.append(current)

        segments, current = [], ""
        for chunk in chunks:
            if current and self._token_len(current + chunk) > budget:
                segments.append(current)
                current = chunk
            else:
                current += chunk
        if current:
            segments.append(current)
        return segments or [text]


class RecordingFst:
    """Wraps a wetext Normalizer and records every call as {lang, input, output|error}."""

    def __init__(self, inner, lang, log):
        self.inner = inner
        self.lang = lang
        self.log = log

    def normalize(self, text):
        try:
            out = self.inner.normalize(text)
        except Exception as e:  # upstream catches and falls back; the C# lookup rethrows from the record
            self.log.append({"lang": self.lang, "input": text, "error": type(e).__name__})
            raise
        self.log.append({"lang": self.lang, "input": text, "output": out})
        return out


def install_whisper_stub():
    # indextts/utils/tokenizer.py imports whisper.tokenizer.Tokenizer only as a base class whose encode/decode forward
    # to the tiktoken Encoding (openai-whisper's own implementation); this stub is that forwarding and nothing else.
    whisper = types.ModuleType("whisper")
    whisper_tokenizer = types.ModuleType("whisper.tokenizer")

    class Tokenizer:
        def __init__(self, encoding, num_languages=99, language=None, task=None):
            self.encoding = encoding
            self.num_languages = num_languages
            self.language = language
            self.task = task

        def encode(self, text, **kwargs):
            return self.encoding.encode(text, **kwargs)

        def decode(self, token_ids, **kwargs):
            return self.encoding.decode(token_ids, **kwargs)

    whisper_tokenizer.Tokenizer = Tokenizer
    whisper.tokenizer = whisper_tokenizer
    sys.modules["whisper"] = whisper
    sys.modules["whisper.tokenizer"] = whisper_tokenizer


def make_normalizer(TextNormalizer, Normalizer, enable_glossary, log):
    tn = TextNormalizer(enable_glossary=enable_glossary)
    # TextNormalizer.load(), non-Linux branch, verbatim constructor arguments; load() then returns early.
    tn.zh_normalizer = RecordingFst(Normalizer(remove_erhua=False, lang="zh", operator="tn"), "zh", log)
    tn.en_normalizer = RecordingFst(Normalizer(lang="en", operator="tn"), "en", log)
    return tn


def gpt_ids(ids):
    # UnifiedVoice.prepare_gpt_inputs valid_mask: drop start (0) and stop (1) text tokens anywhere in the row.
    return [i for i in ids if i != 0 and i != 1]


def dump_v1_v2(text, tokenizer, log):
    log.clear()
    pieces = tokenizer.tokenize(text)
    fst_calls = list(log)
    stripped_single = len(text.strip()) == 1
    log.clear()
    normalized = None if stripped_single or len(text) == 0 else tokenizer.normalizer.normalize(text)
    log.clear()
    cjk = None
    if normalized is not None:
        from indextts.utils.common import tokenize_by_CJK_char
        cjk = tokenize_by_CJK_char(normalized)
    segments = tokenizer.split_segments(pieces, max_text_tokens_per_segment=120)
    segment_ids = [tokenizer.convert_tokens_to_ids(s) for s in segments]
    return {
        "normalized": normalized,
        "cjk_tokenized": cjk,
        "fst_calls": fst_calls,
        "pieces": pieces,
        "piece_ids": tokenizer.convert_tokens_to_ids(pieces),
        "segments": segments,
        "segment_ids": segment_ids,
        "gpt_ids": [gpt_ids(s) for s in segment_ids],
    }


def dump_v25(text, lang, text_process, tokenizer, splitter, log):
    """infer_v2_5.py:696-722 verbatim apart from recording."""
    log.clear()
    lang_prefix = f'<|{lang.lower()}|> '
    cleaned = text_process.clean_pattern.sub(lambda x: text_process.char_rep_map[x.group()], text)
    text = cleaned
    normalized = None
    if lang.lower() in ['zh', 'zhen', 'en']:
        text = text_process.normalize(text)
        normalized = text
    if lang.lower() in ['ja', 'zh', 'zhen', 'en']:
        text = text.lower()
    if lang.lower() == 'es':
        text = text.upper()
    lowered = text
    text = apply_pronunciation_annotations(text)
    annotated = text
    text = re.sub(r'<\|([^|]+)\|>', lambda m: f'<|{m.group(1).upper()}|>', text)
    prepared = text
    fst_calls = list(log)
    segments = splitter.split_text_by_tokens(text, 120, lang_prefix)
    segment_ids = []
    for seg_text in segments:
        toks = tokenizer.encode(lang_prefix + seg_text, allowed_special='all')
        segment_ids.append(toks + [1])  # F.pad(toks, (0, 1), value=1)
    log.clear()
    return {
        "lang": lang,
        "cleaned": cleaned,
        "normalized": normalized,
        "lowered": lowered,
        "annotated": annotated,
        "prepared": prepared,
        "fst_calls": fst_calls,
        "segments": segments,
        "segment_ids": segment_ids,
        "gpt_ids": [gpt_ids(s) for s in segment_ids],
    }


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    repo, models, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    if os.environ.get("PYTHONHASHSEED") != "0":
        sys.exit("Run with PYTHONHASHSEED=0: save_names/save_pinyin_tones order placeholders through set().")
    sys.path.insert(0, repo)
    install_whisper_stub()
    from wetext import Normalizer
    from indextts.utils.front import TextNormalizer, TextTokenizer
    from indextts.utils.tokenizer import get_tokenizer
    import torch

    dir15 = os.path.join(models, "IndexTeam--IndexTTS-1.5")
    dir20 = os.path.join(models, "IndexTeam--IndexTTS-2")
    dir25 = os.path.join(models, "IndexTeam--IndexTTS-2.5")
    log = []
    tok15 = TextTokenizer(os.path.join(dir15, "bpe.model"), make_normalizer(TextNormalizer, Normalizer, False, log))
    tok20 = TextTokenizer(os.path.join(dir20, "bpe.model"), make_normalizer(TextNormalizer, Normalizer, True, log))
    if os.path.exists(os.path.join(dir20, "glossary.yaml")):
        tok20.normalizer.load_glossary_from_yaml(os.path.join(dir20, "glossary.yaml"))
    tp25 = make_normalizer(TextNormalizer, Normalizer, True, log)
    if os.path.exists(os.path.join(dir25, "glossary.yaml")):
        tp25.load_glossary_from_yaml(os.path.join(dir25, "glossary.yaml"))
    tok25 = get_tokenizer(multilingual=True, model_dir=dir25)
    sd = torch.load(os.path.join(dir25, "gpt.pth"), map_location="cpu", mmap=True, weights_only=True)
    sd = sd.get("model", sd)
    capacity = int(sd["text_pos_embedding.emb.weight"].shape[0])
    splitter = SplitV25(tok25, capacity)

    lines = [(t, "en") for t in CORPUS_EN] + [(t, "zh") for t in CORPUS_ZH]
    cases = []
    for text, lang in lines:
        cases.append({
            "text": text,
            "v1_5": dump_v1_v2(text, tok15, log),
            "v2_0": dump_v1_v2(text, tok20, log),
            "v2_5": dump_v25(text, lang, tp25, tok25, splitter, log),
        })

    head = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = {
        "upstream": {"repo": "index-tts/index-tts", "commit": head},
        "wetext": version("wetext"),
        "sentencepiece": version("sentencepiece"),
        "tiktoken": version("tiktoken"),
        "v2_5_text_pos_capacity": capacity,
        "max_text_tokens_per_segment": 120,
        "spm_unk_id": tok15.sp_model.unk_id(),
        "cases": cases,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # One case per line keeps the fixture small and its diffs readable.
    header = json.dumps({k: v for k, v in out.items() if k != "cases"}, ensure_ascii=False)
    lines = [json.dumps(c, ensure_ascii=False, separators=(",", ":")) for c in cases]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header[:-1] + ',\n"cases":[\n' + ",\n".join(lines) + "\n]}\n")
    print(f"{len(cases)} lines -> {out_path}")


if __name__ == "__main__":
    main()
