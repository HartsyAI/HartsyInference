"""Regenerates the reference the WeText parity test diffs against.

CosyVoice 2 pins wetext==0.0.4 and builds `Normalizer()` / `Normalizer(remove_erhua=False)` with lang="auto". 0.0.4
ships no FSTs; it fetches ModelScope `pengzhendong/wetext` at runtime. This script runs the unmodified 0.0.4
`Normalizer` over the corpus below with `snapshot_download` redirected to a local copy of those FSTs (the wetext
0.1.8 wheel bundles byte-identical ones under `wetext/fsts`; ModelScope commit b04bc075 serves the same files), and
writes `wetext_parity.json`: per line, the language auto-detection picked, the tagger output (null when 0.0.4 returns
the text untouched because it has no digit), and the final output (null with `error` when 0.0.4 raises).

    python -m venv wetext004-venv && wetext004-venv/bin/pip install wetext==0.0.4
    wetext004-venv/bin/python tools/parity/wetext_parity_reference.py <fst-dir> \
        tests/HartsyInference.Audio.Tests/Fixtures/wetext_parity.json

<fst-dir> holds en/tn/{tagger,verbalizer}.fst and zh/tn/{tagger,verbalizer}.fst, e.g. the 0.1.8 wheel's wetext/fsts.
"""
import hashlib
import json
import re
import sys
from importlib.metadata import version

import wetext.wetext as wetext_module

ENGLISH = [
    # Dates, years and ordinals.
    "On March 3rd, 2026 at 4:30 PM, 1,250 people paid $99.95 each.",
    "Dr. Smith lives at 221B Baker St. and earns 12% more in 1999.",
    "The meeting is on 2024-05-06.",
    "She was born on 12/25/1990.",
    "Today is Jan. 1, 2000.",
    "The treaty was signed on 4 July 1776.",
    "We moved here in the 1980s.",
    "It happened in the '90s.",
    "The 21st century began in 2001.",
    "He finished 1st, she finished 2nd and they finished 3rd.",
    "This is the 100th anniversary.",
    "The 4th of July is a holiday.",
    "Queen Elizabeth II reigned for 70 years.",
    "Chapter 11 starts on page 304.",
    "World War 2 ended in 1945.",
    "The year 2000 problem was overblown.",
    "In 1066 the Normans arrived.",
    "The company was founded in 1905 and sold in 2010.",
    "The deadline is Friday, October 13, 2023.",
    "Sept 11 2001 changed everything.",
    # Times.
    "Meet me at 10am tomorrow.",
    "The train leaves at 7:45 a.m. sharp.",
    "Dinner is at 6:30 pm.",
    "The store opens at 09:00 and closes at 21:30.",
    "It is 12:00 noon.",
    "The launch is at 3:15:20 PM.",
    "Wake up at 5 o'clock.",
    "The show runs from 8 to 10 PM.",
    # Money.
    "It costs $5.",
    "The house sold for $1,250,000.",
    "A coffee is $3.50 here.",
    "They raised $2 million last year.",
    "The budget is $4.5 billion.",
    "It costs \u20ac45 or \u00a330.",
    "The fee is 50 cents.",
    "He owes me $0.99.",
    "Tickets cost $12.00 each.",
    "The prize was \u00a51000.",
    "Revenue hit $3.2B in Q4.",
    "I paid USD 20 for it.",
    # Percentages, decimals, fractions.
    "Sales grew by 15%.",
    "Only 0.5% of users clicked.",
    "The rate is 3.75 percent.",
    "Pi is about 3.14159.",
    "Add 1/2 cup of sugar.",
    "About 2/3 of the class passed.",
    "The value is -42.",
    "The temperature is -3.5 degrees.",
    "Multiply by 1.5 and divide by 0.25.",
    "The score was 7 to 3.",
    "The odds are 1 in 1,000,000.",
    "A ratio of 16:9 is common.",
    # Plain cardinals.
    "I have 2 cats and 3 dogs.",
    "There are 7 days in a week.",
    "He counted to 100.",
    "The population is 8,000,000,000.",
    "Room 101 is empty.",
    "Bus 42 is late.",
    "She scored 0 points.",
    "It weighs 1000 kilograms.",
    "We need 12 eggs, 3 apples and 1 loaf.",
    "The answer is 42.",
    "Flight 815 departed late.",
    "There were 365 entries.",
    "Version 2 is better than version 1.",
    "Number 9, number 9.",
    "The code is 007.",
    "Add 15 and 27 to get 42.",
    # Ranges.
    "Read pages 10-20 tonight.",
    "The years 1914-1918 were brutal.",
    "Ages 5 to 12 are welcome.",
    "Open 9-5 Monday to Friday.",
    "It lasted 2-3 weeks.",
    # Phone numbers and identifiers.
    "Call me at 555-123-4567 tomorrow.",
    "My number is (555) 987-6543.",
    "Dial +1 800 555 0199 for help.",
    "Call 911 in an emergency.",
    "The zip code is 90210.",
    "My PIN is 4821.",
    "Order #12345 has shipped.",
    "Use code ABC123 at checkout.",
    "The IP is 192.168.1.1.",
    "Visit www.example.com for 20 tips.",
    "Email me at john.doe99@example.com today.",
    "The ISBN is 978-3-16-148410-0.",
    # Measurements.
    "The box is 5 cm wide.",
    "He ran 10 km in 45 minutes.",
    "The bottle holds 500 ml.",
    "It is 6 ft 2 in tall.",
    "The car does 60 mph.",
    "Set the oven to 350\u00b0F.",
    "It was 25\u00b0C outside.",
    "The file is 4 GB.",
    "The CPU runs at 3.6 GHz.",
    "Add 200 g of flour.",
    "The road is 3 mi long.",
    "It weighs 150 lbs.",
    "The screen is 27 inches.",
    "The pool is 50 m long.",
    "Download speed is 100 Mbps.",
    "The dose is 5 mg per day.",
    "The plot is 2 ha.",
    "He is 180cm tall.",
    # Abbreviations and titles.
    "Mr. Jones met Mrs. Brown at 5.",
    "Ms. Lee has 3 kids.",
    "Dr. Who has 13 doctors.",
    "St. Louis has 300,000 people.",
    "Prof. Adams teaches 4 classes.",
    "Lt. Dan lost 2 legs.",
    "Mt. Everest is 8,849 m high.",
    "Apt. 4B is upstairs.",
    "The U.S. has 50 states.",
    "The U.K. joined in 1973.",
    "Jr. and Sr. are 2 suffixes.",
    "Ave. 5 is closed.",
    "Gen. Grant won in 1865.",
    "Capt. Hook lost 1 hand.",
    "See fig. 3 and vol. 2.",
    "Etc. and e.g. are 2 abbreviations.",
    "NASA launched 3 rockets.",
    "The FBI and CIA employ 1000s.",
    # Symbols.
    "Tom & Jerry aired in 1940.",
    "Use # 5 for this.",
    "The cost is 5 + 3 = 8.",
    "10 > 5 and 3 < 4.",
    "Rate: 5/10 stars.",
    "Win 100% of the time.",
    "Item @ $5 each.",
    "Mix A & B at 50:50.",
    "Wait... what?! No -- really; (yes), 'quoted' & \"double\" -- ok? 3",
    "He said \"I have 3 cats\" twice.",
    "It's 'version 2' now.",
    "The 'big' 5 companies.",
    "*Note: 3 items left.*",
    "Path is C:\\Users\\5.",
    "Use_snake_case 5 times.",
    "a/b testing with 2 groups.",
    "~10 people came.",
    "The price went from $5 to $7 (+40%).",
    # Mixed case and formatting.
    "THE YEAR 2024 IS HERE.",
    "Top 10 Tips For 2025!",
    "iPhone 15 Pro Max costs $1199.",
    "COVID-19 started in 2019.",
    "Windows 11 replaced Windows 10.",
    "MP3 files and MP4 videos.",
    "The F-35 flies at Mach 1.6.",
    "Area 51 is in Nevada.",
    "Agent 47 is fictional.",
    "R2-D2 and C-3PO are droids.",
    "The 3D printer made 5 parts.",
    "Use 4K resolution at 60fps.",
    "Formula 1 races are fast.",
    "B2B and B2C markets grew 12%.",
    "Studio 54 opened in 1977.",
    # Plain sentences without digits (0.0.4 returns these unchanged).
    "Yes.",
    "Okay, see you.",
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "Dr. Smith is here.",
    "Wait... what?! No -- really; (yes), 'quoted' & \"double\" -- ok?",
    "No no no no no no no no no no no no.",
    "Mr. and Mrs. Smith went to St. Paul.",
    "I can't believe it's not butter!",
    # Whitespace edges (0.0.4 never strips the input).
    "  3 cats  ",
    "Line one 1\tand tab 2.",
    "Ends with a number 7",
    "8",
    "12345",
    "1.5",
    "$100",
    "50%",
    "2nd",
    # Numbers inside longer prose.
    "In 2023, about 4.5 million visitors spent an average of $250 over 3.2 days.",
    "The recipe needs 2 eggs, 1/4 cup milk, and 350 g of flour at 180\u00b0C for 25 minutes.",
    "Between 9:00 and 17:00, 1,024 calls came in, 12% more than on 3/14/2023.",
    "Unit 7, 1600 Pennsylvania Ave., Washington, DC 20500.",
    "Patients aged 65+ make up 18.5% of the 2.1 million total.",
]

CHINESE = [
    "今天是2024年5月6日，气温25℃。",
    "这件衣服售价¥199.5元。",
    "我的电话是13812345678。",
    "比分是3:2，时间是下午3点15分。",
    "他跑了1.5公里，用了12分30秒。",
    "大约有50%的人同意，1/3的人反对。",
    "第1名获得了10000元奖金。",
    "会议定于2023年12月31日举行。",
    "现在是早上8:30。",
    "晚上10点见。",
    "1949年10月1日，中华人民共和国成立。",
    "他今年18岁。",
    "这本书有365页。",
    "价格是3.5元一斤。",
    "我们班有45个学生。",
    "温度降到了-5℃。",
    "增长了12.5%。",
    "身高175cm，体重70kg。",
    "距离是100km。",
    "请拨打110报警。",
    "邮编是100084。",
    "他住在3楼302室。",
    "第2章第3节。",
    "2/5的人选择了A。",
    "这台电脑重2.3千克。",
    "股价上涨了0.75元。",
    "我花了1,000元。",
    "全国有14亿人口。",
    "他在2008年北京奥运会上获得了3枚金牌。",
    "下午2:45开会。",
    "时间是23:59:59。",
    "1+1=2。",
    "5-3=2。",
    "3×4=12。",
    "10÷2=5。",
    "这辆车时速120公里。",
    "约有1万人参加。",
    "第100位顾客。",
    "这是第3次了。",
    "门牌号是88号。",
    "密码是0000。",
    "房间号是0815。",
    "我买了2斤苹果。",
    "这瓶水500毫升。",
    "他考了98.5分。",
    "利率是4.35%。",
    "价格从100元涨到120元。",
    "2020-2022年疫情严重。",
    "年龄在18-25岁之间。",
    "请在3-5个工作日内完成。",
    "QQ号是123456789。",
    "身份证号是110101199003071234。",
    "车牌号是京A12345。",
    "我有3个苹果和2个香蕉。",
    "一共是¥88.88。",
    "这件事发生在1998年。",
    "她出生于1995年3月8日。",
    "今天星期3。",
    "2点了，该走了。",
    "12点整。",
    "5分钟后出发。",
    "等了30秒。",
    "他跑了42.195公里。",
    "电压是220V。",
    "功率是1500W。",
    "屏幕是6.1英寸。",
    "内存是16GB。",
    "网速是100Mbps。",
    "这里海拔3000米。",
    "面积是120平方米。",
    "体积是2立方米。",
    "他喝了3杯咖啡。",
    "买了10本书。",
    "第一名是007号选手。",
    "比赛结果是2:1。",
    "零下10度。",
    "这台机器有99.9%的准确率。",
    "预计2030年完成。",
    "公元前221年秦统一六国。",
    "他有1/4的英国血统。",
    "增加了3倍。",
    "约0.5公斤。",
    "每人200元。",
    "共计12,345,678元。",
    "第3季度营收增长15%。",
    "2024年第1季度。",
    "我在2楼等你。",
    "这是iPhone 15。",
    "他的成绩是A+，排名第5。",
    "Windows 11系统。",
    "5G网络覆盖了90%的城市。",
    "COVID-19疫情始于2019年。",
    "开了2个小时的会。",
    "早上6点起床，晚上11点睡觉。",
    "两个人花了300块钱。",
    "儿子今年3岁了，喜欢玩儿。",
    "哪儿有5块钱？",
    "我有一个苹果。",
    "今天天气很好。",
    "你好，世界！",
    "  我有3本书  ",
    "1998",
    "3.14",
    "50%",
    "100元",
    "Python3.12版本发布了。",
    "我们约在3:00 PM见面。",
    "价格是$99。",
    "温度是36.5°C。",
]


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    fst_dir, out_path = sys.argv[1], sys.argv[2]
    if version("wetext") != "0.0.4":
        sys.exit(f"wetext {version('wetext')} is installed; CosyVoice pins 0.0.4")
    wetext_module.snapshot_download = lambda *args, **kwargs: fst_dir
    normalizer = wetext_module.Normalizer(lang="auto", operator="tn", remove_erhua=False)

    entries = []
    for text in ENGLISH + CHINESE:
        entry = {"text": text}
        has_digit = re.search(r"\d", text) is not None
        lang = "zh" if wetext_module.Normalizer.contains_chinese(text) else "en"
        entry["lang"] = lang
        try:
            entry["tagged"] = normalizer.tag(text, lang) if has_digit else None
            entry["output"] = normalizer.normalize(text)
        except Exception as error:  # 0.0.4 raises on inputs its tagger cannot accept
            entry["tagged"] = entry.get("tagged")
            entry["output"] = None
            entry["error"] = type(error).__name__
        entries.append(entry)

    hashes = {}
    for lang in ("en", "zh"):
        for name in ("tagger", "verbalizer"):
            with open(f"{fst_dir}/{lang}/tn/{name}.fst", "rb") as handle:
                hashes[f"{lang}/tn/{name}.fst"] = hashlib.sha256(handle.read()).hexdigest()
    fixture = {"wetext": version("wetext"), "fst_sha256": hashes, "entries": entries}
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(fixture, handle, ensure_ascii=False, indent=1)
        handle.write("\n")
    print(f"{len(entries)} entries ({len(ENGLISH)} English, {len(CHINESE)} Chinese) -> {out_path}")


if __name__ == "__main__":
    main()
