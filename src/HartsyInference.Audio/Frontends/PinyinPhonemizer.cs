using System.Collections.Generic;

namespace HartsyInference.Audio.Frontends;

/// <summary>The deterministic core of the GPT-SoVITS Chinese G2P: converts per-syllable pinyin (initials +
/// tone-3 finals, as produced by pypinyin / g2pW) into the model's phoneme symbols plus the per-character
/// phoneme counts (<c>word2ph</c>), via the upstream <c>opencpop-strict</c> map and the pinyin-normalization
/// rep-maps. Verbatim from <c>chinese2._g2p</c>'s syllable→phoneme assembly. Any pinyin source (the ported
/// pypinyin path or the g2pW model) feeds this.</summary>
public static class PinyinPhonemizer
{
    // opencpop-strict.txt: normalized pinyin syllable → "initial final" phoneme pair (or a single final).
    private static readonly Dictionary<string, string> _pinyinToSymbol = new()
    {
        ["a"]="AA a", ["ai"]="AA ai", ["an"]="AA an", ["ang"]="AA ang", ["ao"]="AA ao", ["ba"]="b a",
        ["bai"]="b ai", ["ban"]="b an", ["bang"]="b ang", ["bao"]="b ao", ["bei"]="b ei", ["ben"]="b en",
        ["beng"]="b eng", ["bi"]="b i", ["bian"]="b ian", ["biao"]="b iao", ["bie"]="b ie", ["bin"]="b in",
        ["bing"]="b ing", ["bo"]="b o", ["bu"]="b u", ["ca"]="c a", ["cai"]="c ai", ["can"]="c an",
        ["cang"]="c ang", ["cao"]="c ao", ["ce"]="c e", ["cei"]="c ei", ["cen"]="c en", ["ceng"]="c eng",
        ["cha"]="ch a", ["chai"]="ch ai", ["chan"]="ch an", ["chang"]="ch ang", ["chao"]="ch ao", ["che"]="ch e",
        ["chen"]="ch en", ["cheng"]="ch eng", ["chi"]="ch ir", ["chong"]="ch ong", ["chou"]="ch ou", ["chu"]="ch u",
        ["chua"]="ch ua", ["chuai"]="ch uai", ["chuan"]="ch uan", ["chuang"]="ch uang", ["chui"]="ch ui", ["chun"]="ch un",
        ["chuo"]="ch uo", ["ci"]="c i0", ["cong"]="c ong", ["cou"]="c ou", ["cu"]="c u", ["cuan"]="c uan",
        ["cui"]="c ui", ["cun"]="c un", ["cuo"]="c uo", ["da"]="d a", ["dai"]="d ai", ["dan"]="d an",
        ["dang"]="d ang", ["dao"]="d ao", ["de"]="d e", ["dei"]="d ei", ["den"]="d en", ["deng"]="d eng",
        ["di"]="d i", ["dia"]="d ia", ["dian"]="d ian", ["diao"]="d iao", ["die"]="d ie", ["ding"]="d ing",
        ["diu"]="d iu", ["dong"]="d ong", ["dou"]="d ou", ["du"]="d u", ["duan"]="d uan", ["dui"]="d ui",
        ["dun"]="d un", ["duo"]="d uo", ["e"]="EE e", ["ei"]="EE ei", ["en"]="EE en", ["eng"]="EE eng",
        ["er"]="EE er", ["fa"]="f a", ["fan"]="f an", ["fang"]="f ang", ["fei"]="f ei", ["fen"]="f en",
        ["feng"]="f eng", ["fo"]="f o", ["fou"]="f ou", ["fu"]="f u", ["ga"]="g a", ["gai"]="g ai",
        ["gan"]="g an", ["gang"]="g ang", ["gao"]="g ao", ["ge"]="g e", ["gei"]="g ei", ["gen"]="g en",
        ["geng"]="g eng", ["gong"]="g ong", ["gou"]="g ou", ["gu"]="g u", ["gua"]="g ua", ["guai"]="g uai",
        ["guan"]="g uan", ["guang"]="g uang", ["gui"]="g ui", ["gun"]="g un", ["guo"]="g uo", ["ha"]="h a",
        ["hai"]="h ai", ["han"]="h an", ["hang"]="h ang", ["hao"]="h ao", ["he"]="h e", ["hei"]="h ei",
        ["hen"]="h en", ["heng"]="h eng", ["hong"]="h ong", ["hou"]="h ou", ["hu"]="h u", ["hua"]="h ua",
        ["huai"]="h uai", ["huan"]="h uan", ["huang"]="h uang", ["hui"]="h ui", ["hun"]="h un", ["huo"]="h uo",
        ["ji"]="j i", ["jia"]="j ia", ["jian"]="j ian", ["jiang"]="j iang", ["jiao"]="j iao", ["jie"]="j ie",
        ["jin"]="j in", ["jing"]="j ing", ["jiong"]="j iong", ["jiu"]="j iu", ["ju"]="j v", ["jv"]="j v",
        ["juan"]="j van", ["jvan"]="j van", ["jue"]="j ve", ["jve"]="j ve", ["jun"]="j vn", ["jvn"]="j vn",
        ["ka"]="k a", ["kai"]="k ai", ["kan"]="k an", ["kang"]="k ang", ["kao"]="k ao", ["ke"]="k e",
        ["kei"]="k ei", ["ken"]="k en", ["keng"]="k eng", ["kong"]="k ong", ["kou"]="k ou", ["ku"]="k u",
        ["kua"]="k ua", ["kuai"]="k uai", ["kuan"]="k uan", ["kuang"]="k uang", ["kui"]="k ui", ["kun"]="k un",
        ["kuo"]="k uo", ["la"]="l a", ["lai"]="l ai", ["lan"]="l an", ["lang"]="l ang", ["lao"]="l ao",
        ["le"]="l e", ["lei"]="l ei", ["leng"]="l eng", ["li"]="l i", ["lia"]="l ia", ["lian"]="l ian",
        ["liang"]="l iang", ["liao"]="l iao", ["lie"]="l ie", ["lin"]="l in", ["ling"]="l ing", ["liu"]="l iu",
        ["lo"]="l o", ["long"]="l ong", ["lou"]="l ou", ["lu"]="l u", ["luan"]="l uan", ["lun"]="l un",
        ["luo"]="l uo", ["lv"]="l v", ["lve"]="l ve", ["ma"]="m a", ["mai"]="m ai", ["man"]="m an",
        ["mang"]="m ang", ["mao"]="m ao", ["me"]="m e", ["mei"]="m ei", ["men"]="m en", ["meng"]="m eng",
        ["mi"]="m i", ["mian"]="m ian", ["miao"]="m iao", ["mie"]="m ie", ["min"]="m in", ["ming"]="m ing",
        ["miu"]="m iu", ["mo"]="m o", ["mou"]="m ou", ["mu"]="m u", ["na"]="n a", ["nai"]="n ai",
        ["nan"]="n an", ["nang"]="n ang", ["nao"]="n ao", ["ne"]="n e", ["nei"]="n ei", ["nen"]="n en",
        ["neng"]="n eng", ["ni"]="n i", ["nian"]="n ian", ["niang"]="n iang", ["niao"]="n iao", ["nie"]="n ie",
        ["nin"]="n in", ["ning"]="n ing", ["niu"]="n iu", ["nong"]="n ong", ["nou"]="n ou", ["nu"]="n u",
        ["nuan"]="n uan", ["nun"]="n un", ["nuo"]="n uo", ["nv"]="n v", ["nve"]="n ve", ["o"]="OO o",
        ["ou"]="OO ou", ["pa"]="p a", ["pai"]="p ai", ["pan"]="p an", ["pang"]="p ang", ["pao"]="p ao",
        ["pei"]="p ei", ["pen"]="p en", ["peng"]="p eng", ["pi"]="p i", ["pian"]="p ian", ["piao"]="p iao",
        ["pie"]="p ie", ["pin"]="p in", ["ping"]="p ing", ["po"]="p o", ["pou"]="p ou", ["pu"]="p u",
        ["qi"]="q i", ["qia"]="q ia", ["qian"]="q ian", ["qiang"]="q iang", ["qiao"]="q iao", ["qie"]="q ie",
        ["qin"]="q in", ["qing"]="q ing", ["qiong"]="q iong", ["qiu"]="q iu", ["qu"]="q v", ["qv"]="q v",
        ["quan"]="q van", ["qvan"]="q van", ["que"]="q ve", ["qve"]="q ve", ["qun"]="q vn", ["qvn"]="q vn",
        ["ran"]="r an", ["rang"]="r ang", ["rao"]="r ao", ["re"]="r e", ["ren"]="r en", ["reng"]="r eng",
        ["ri"]="r ir", ["rong"]="r ong", ["rou"]="r ou", ["ru"]="r u", ["rua"]="r ua", ["ruan"]="r uan",
        ["rui"]="r ui", ["run"]="r un", ["ruo"]="r uo", ["sa"]="s a", ["sai"]="s ai", ["san"]="s an",
        ["sang"]="s ang", ["sao"]="s ao", ["se"]="s e", ["sen"]="s en", ["seng"]="s eng", ["sha"]="sh a",
        ["shai"]="sh ai", ["shan"]="sh an", ["shang"]="sh ang", ["shao"]="sh ao", ["she"]="sh e", ["shei"]="sh ei",
        ["shen"]="sh en", ["sheng"]="sh eng", ["shi"]="sh ir", ["shou"]="sh ou", ["shu"]="sh u", ["shua"]="sh ua",
        ["shuai"]="sh uai", ["shuan"]="sh uan", ["shuang"]="sh uang", ["shui"]="sh ui", ["shun"]="sh un", ["shuo"]="sh uo",
        ["si"]="s i0", ["song"]="s ong", ["sou"]="s ou", ["su"]="s u", ["suan"]="s uan", ["sui"]="s ui",
        ["sun"]="s un", ["suo"]="s uo", ["ta"]="t a", ["tai"]="t ai", ["tan"]="t an", ["tang"]="t ang",
        ["tao"]="t ao", ["te"]="t e", ["tei"]="t ei", ["teng"]="t eng", ["ti"]="t i", ["tian"]="t ian",
        ["tiao"]="t iao", ["tie"]="t ie", ["ting"]="t ing", ["tong"]="t ong", ["tou"]="t ou", ["tu"]="t u",
        ["tuan"]="t uan", ["tui"]="t ui", ["tun"]="t un", ["tuo"]="t uo", ["wa"]="w a", ["wai"]="w ai",
        ["wan"]="w an", ["wang"]="w ang", ["wei"]="w ei", ["wen"]="w en", ["weng"]="w eng", ["wo"]="w o",
        ["wu"]="w u", ["xi"]="x i", ["xia"]="x ia", ["xian"]="x ian", ["xiang"]="x iang", ["xiao"]="x iao",
        ["xie"]="x ie", ["xin"]="x in", ["xing"]="x ing", ["xiong"]="x iong", ["xiu"]="x iu", ["xu"]="x v",
        ["xv"]="x v", ["xuan"]="x van", ["xvan"]="x van", ["xue"]="x ve", ["xve"]="x ve", ["xun"]="x vn",
        ["xvn"]="x vn", ["ya"]="y a", ["yan"]="y En", ["yang"]="y ang", ["yao"]="y ao", ["ye"]="y E",
        ["yi"]="y i", ["yin"]="y in", ["ying"]="y ing", ["yo"]="y o", ["yong"]="y ong", ["you"]="y ou",
        ["yu"]="y v", ["yv"]="y v", ["yuan"]="y van", ["yvan"]="y van", ["yue"]="y ve", ["yve"]="y ve",
        ["yun"]="y vn", ["yvn"]="y vn", ["za"]="z a", ["zai"]="z ai", ["zan"]="z an", ["zang"]="z ang",
        ["zao"]="z ao", ["ze"]="z e", ["zei"]="z ei", ["zen"]="z en", ["zeng"]="z eng", ["zha"]="zh a",
        ["zhai"]="zh ai", ["zhan"]="zh an", ["zhang"]="zh ang", ["zhao"]="zh ao", ["zhe"]="zh e", ["zhei"]="zh ei",
        ["zhen"]="zh en", ["zheng"]="zh eng", ["zhi"]="zh ir", ["zhong"]="zh ong", ["zhou"]="zh ou", ["zhu"]="zh u",
        ["zhua"]="zh ua", ["zhuai"]="zh uai", ["zhuan"]="zh uan", ["zhuang"]="zh uang", ["zhui"]="zh ui", ["zhun"]="zh un",
        ["zhuo"]="zh uo", ["zi"]="z i0", ["zong"]="z ong", ["zou"]="z ou", ["zu"]="z u", ["zuan"]="z uan",
        ["zui"]="z ui", ["zun"]="z un", ["zuo"]="z uo",
    };

    private static readonly HashSet<char> _punct = ['!', '?', '…', ',', '.', '-'];

    /// <summary>Assembles phonemes + word2ph from aligned per-syllable <paramref name="initials"/> and
    /// <paramref name="finalsTone3"/> (a punctuation token has initial == final). Mirrors the upstream
    /// assertion that every syllable maps to a known pinyin.</summary>
    public static (List<string> Phones, List<int> Word2Ph) Assemble(
        IReadOnlyList<string> initials, IReadOnlyList<string> finalsTone3)
    {
        List<string> phones = [];
        List<int> word2ph = [];
        for (int i = 0; i < initials.Count; i++)
        {
            string c = initials[i], v = finalsTone3[i];
            if (c == v)
            {
                phones.Add(c);
                word2ph.Add(1);
                continue;
            }
            char tone = v[^1];
            string vWithoutTone = v[..^1];
            string pinyin = c + vWithoutTone;

            if (c.Length > 0)
            {
                pinyin = vWithoutTone switch
                {
                    "uei" => c + "ui",
                    "iou" => c + "iu",
                    "uen" => c + "un",
                    _ => pinyin,
                };
            }
            else
            {
                pinyin = pinyin switch
                {
                    "ing" => "ying",
                    "i" => "yi",
                    "in" => "yin",
                    "u" => "wu",
                    _ => SingleRep(pinyin),
                };
            }

            if (!_pinyinToSymbol.TryGetValue(pinyin, out string? sym))
                throw new KeyNotFoundException($"pinyin '{pinyin}' (from '{c}{v}') is not in opencpop-strict.");
            int sp = sym.IndexOf(' ');
            string newC = sym[..sp], newV = sym[(sp + 1)..] + tone;
            phones.Add(newC);
            phones.Add(newV);
            word2ph.Add(2);
        }
        return (phones, word2ph);
    }

    private static string SingleRep(string pinyin) => pinyin[0] switch
    {
        'v' => "yu" + pinyin[1..],
        'e' => pinyin,
        'i' => "y" + pinyin[1..],
        'u' => "w" + pinyin[1..],
        _ => pinyin,
    };

    /// <summary>True if <paramref name="symbol"/> is one of the punctuation tokens (initial == final case).</summary>
    public static bool IsPunctuation(string symbol) => symbol.Length == 1 && _punct.Contains(symbol[0]);
}
