"""
Layihənin mərkəzi konfiqurasiyası: yollar, sabitlər, biokimyəvi cədvəllər.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Yollar
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"

CLINVAR_URL = (
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
)
CLINVAR_LOCAL = DATA_DIR / "variant_summary.txt.gz"

RANDOM_STATE = 42
TEST_SIZE = 0.20

# ---------------------------------------------------------------------------
# Amin turşuları: 3 hərfli -> 1 hərfli kodlar
# ---------------------------------------------------------------------------
AA3_TO_AA1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
}
AMINO_ACIDS = list(AA3_TO_AA1.values())

# ---------------------------------------------------------------------------
# Amin turşularının fiziki-kimyəvi xassələri
#   hydrophobicity : Kyte-Doolittle şkalası
#   volume         : yan zəncirin həcmi (Å^3, Zamyatnin)
#   mw             : molekulyar kütlə (Da)
#   charge         : pH 7-də təxmini yük
#   polar          : polyar (1) / qeyri-polyar (0)
#   aromatic       : aromatik halqa varmı
# ---------------------------------------------------------------------------
AA_PROPERTIES = {
    #      hydro   volume   mw     charge polar aromatic
    "A": (1.8,   88.6,   89.1,   0.0, 0, 0),
    "R": (-4.5, 173.4,  174.2,   1.0, 1, 0),
    "N": (-3.5, 114.1,  132.1,   0.0, 1, 0),
    "D": (-3.5, 111.1,  133.1,  -1.0, 1, 0),
    "C": (2.5,  108.5,  121.2,   0.0, 0, 0),
    "Q": (-3.5, 143.8,  146.2,   0.0, 1, 0),
    "E": (-3.5, 138.4,  147.1,  -1.0, 1, 0),
    "G": (-0.4,  60.1,   75.1,   0.0, 0, 0),
    "H": (-3.2, 153.2,  155.2,   0.1, 1, 1),
    "I": (4.5,  166.7,  131.2,   0.0, 0, 0),
    "L": (3.8,  166.7,  131.2,   0.0, 0, 0),
    "K": (-3.9, 168.6,  146.2,   1.0, 1, 0),
    "M": (1.9,  162.9,  149.2,   0.0, 0, 0),
    "F": (2.8,  189.9,  165.2,   0.0, 0, 1),
    "P": (-1.6, 112.7,  115.1,   0.0, 0, 0),
    "S": (-0.8,  89.0,  105.1,   0.0, 1, 0),
    "T": (-0.7, 116.1,  119.1,   0.0, 1, 0),
    "W": (-0.9, 227.8,  204.2,   0.0, 0, 1),
    "Y": (-1.3, 193.6,  181.2,   0.0, 1, 1),
    "V": (4.2,  140.0,  117.1,   0.0, 0, 0),
}
AA_PROPERTY_NAMES = ["hydrophobicity", "volume", "mw", "charge", "polar", "aromatic"]

# ---------------------------------------------------------------------------
# BLOSUM62 əvəzetmə matrisi (evolyusion olaraq əvəzetmənin "qəbuledilənliyi")
# ---------------------------------------------------------------------------
_BLOSUM_ORDER = "ARNDCQEGHILKMFPSTWYV"
_BLOSUM_ROWS = """
 4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0
-1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3
-2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3
-2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3
 0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
-1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2
-1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2
 0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3
-2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3
-1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3
-1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1
-1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2
-1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1
-2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1
-1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2
 1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2
 0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0
-3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3
-2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1
 0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4
"""


def _build_blosum62() -> dict:
    rows = [list(map(int, r.split())) for r in _BLOSUM_ROWS.strip().splitlines()]
    matrix = {}
    for i, a in enumerate(_BLOSUM_ORDER):
        for j, b in enumerate(_BLOSUM_ORDER):
            matrix[(a, b)] = rows[i][j]
    # Simmetriya yoxlaması (yazı səhvlərini tutmaq üçün)
    for a in _BLOSUM_ORDER:
        for b in _BLOSUM_ORDER:
            assert matrix[(a, b)] == matrix[(b, a)], f"BLOSUM62 asimmetrikdir: {a}{b}"
    return matrix


BLOSUM62 = _build_blosum62()

# ---------------------------------------------------------------------------
# Zülal domenləri (katalitik / funksional vacib bölgələr)
# DİQQƏT: Bu cədvəl təxmini və nümunə məqsədlidir. Real tədqiqatda
# UniProt / InterPro / Pfam annotasiyalarından istifadə edin.
#   gen -> [(başlanğıc, son, domen_adı), ...]
# ---------------------------------------------------------------------------
FUNCTIONAL_DOMAINS = {
    "TP53":  [(94, 292, "DNA-binding domain"), (325, 356, "Tetramerization")],
    "BRCA1": [(24, 64, "RING finger"), (1650, 1857, "BRCT repeats")],
    "BRCA2": [(2402, 3190, "DNA-binding domain")],
    "PTEN":  [(14, 185, "Phosphatase domain"), (190, 350, "C2 domain")],
    "KRAS":  [(10, 17, "P-loop"), (30, 38, "Switch I"), (59, 76, "Switch II")],
    "MLH1":  [(1, 340, "ATPase domain")],
    "CFTR":  [(389, 678, "NBD1"), (1202, 1445, "NBD2")],
}

# Sintetik data üçün zülal uzunluqları
PROTEIN_LENGTHS = {
    "TP53": 393, "BRCA1": 1863, "BRCA2": 3418, "PTEN": 403,
    "KRAS": 189, "MLH1": 756, "CFTR": 1480,
}

# ---------------------------------------------------------------------------
# Evolyusion/proqnostik skor sütunları
#   SIFT     : 0-1, kiçik dəyər = zərərli (<0.05)
#   PolyPhen : 0-1, böyük dəyər = zərərli (>0.85)
#   CADD     : PHRED şkalası, böyük dəyər = zərərli (>20)
# ---------------------------------------------------------------------------
SCORE_COLUMNS = ["sift_score", "polyphen_score", "cadd_phred"]
