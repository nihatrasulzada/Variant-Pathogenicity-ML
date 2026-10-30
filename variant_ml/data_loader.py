"""
ADDIM 1: Verilənlərin toplanması və ilkin emalı.

İş axını:
  1) ClinVar `variant_summary.txt.gz` faylını yüklə (və ya lokal fayldan oxu).
     Canlı bağlantı mümkün deyilsə -> `generate_mock_clinvar()` eyni formatda
     sintetik dataset yaradır, beləliklə qalan pipeline dəyişmədən işləyir.
  2) Yalnız tək nukleotid variantlarını (SNV) saxla.
  3) HGVS protein notasiyasından (p.Arg248Trp) missense mutasiyaları çıxar.
  4) Yalnız qəti 'Pathogenic' (1) və 'Benign' (0) variantları saxla.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

# p.Arg248Trp  ->  ref=Arg, pos=248, alt=Trp
_PROTEIN_CHANGE_RE = re.compile(r"\(p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})\)")

# ClinVar-ın müxtəlif buraxılışlarında klinik əhəmiyyət sütununun adı fərqlənə bilər
_SIGNIFICANCE_COLUMNS = ("ClinicalSignificance", "GermlineClassification")

# Ən aşağı etibarlılıq səviyyələri (ulduzsuz) — istəyə görə çıxarılır
_LOW_CONFIDENCE_REVIEW = {
    "no assertion criteria provided",
    "no assertion provided",
    "no interpretation for the single variant",
    "no classification provided",
    "no classification for the single variant",
}


# ---------------------------------------------------------------------------
# 1.1  ClinVar-ın yüklənməsi
# ---------------------------------------------------------------------------
def download_clinvar(dest: Path = config.CLINVAR_LOCAL, force: bool = False) -> Path:
    """NCBI FTP-dən ClinVar variant_summary faylını yükləyir (~400 MB)."""
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f"[data] Lokal fayl artıq mövcuddur: {dest}")
        return dest

    print(f"[data] ClinVar yüklənir: {config.CLINVAR_URL}")
    with requests.get(config.CLINVAR_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(".part")
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
        tmp.rename(dest)
    print(f"[data] Yükləndi: {dest}")
    return dest


def load_clinvar(path: Path = config.CLINVAR_LOCAL) -> pd.DataFrame:
    """ClinVar faylını yalnız lazım olan sütunlarla oxuyur."""
    header = pd.read_csv(path, sep="\t", nrows=0).columns
    sig_col = next((c for c in _SIGNIFICANCE_COLUMNS if c in header), None)
    if sig_col is None:
        raise ValueError("Faylda klinik əhəmiyyət sütunu tapılmadı.")

    wanted = ["VariationID", "Type", "Name", "GeneSymbol", sig_col,
              "ReviewStatus", "Assembly"]
    df = pd.read_csv(path, sep="\t", usecols=[c for c in wanted if c in header],
                     low_memory=False)
    df = df.rename(columns={sig_col: "ClinicalSignificance"})

    # Hər variant GRCh37 və GRCh38 üçün iki dəfə yazılır -> birini saxla
    if "Assembly" in df.columns:
        df = df[df["Assembly"] == "GRCh38"]
    print(f"[data] ClinVar oxundu: {len(df):,} sətir")
    return df


# ---------------------------------------------------------------------------
# 1.2  Sintetik / mock dataset (ClinVar formatında)
# ---------------------------------------------------------------------------
def _in_functional_domain(gene: str, pos: int) -> int:
    for start, end, _ in config.FUNCTIONAL_DOMAINS.get(gene, []):
        if start <= pos <= end:
            return 1
    return 0


def generate_mock_clinvar(n_samples: int = 5000,
                          missing_rate: float = 0.12,
                          random_state: int = config.RANDOM_STATE) -> pd.DataFrame:
    """
    ClinVar `variant_summary` strukturunu təqlid edən sintetik DataFrame yaradır.

    Bioloji cəhətdən məntiqli siqnal əlavə olunur: patogenlik ehtimalı
      - BLOSUM62 skoru aşağı olduqca (radikal əvəzetmə),
      - hidrofobluq və yük dəyişikliyi böyük olduqca,
      - Gly/Pro/Cys iştirak etdikdə,
      - mutasiya funksional domendə olduqda artır.
    Bundan əlavə, filtrləmə addımını sınamaq üçün qeyri-missense (nonsense,
    sinonim), qeyri-SNV və qeyri-müəyyən (VUS, Likely ...) sətirlər də əlavə edilir.
    SIFT/PolyPhen/CADD skorları etiketlə korrelyasiyalı yaradılır və
    `missing_rate` nisbətində boş (NaN) buraxılır.
    """
    rng = np.random.default_rng(random_state)
    aa1_to_aa3 = {v: k for k, v in config.AA3_TO_AA1.items()}
    genes = list(config.PROTEIN_LENGTHS)

    rows = []
    for i in range(n_samples):
        gene = rng.choice(genes)
        length = config.PROTEIN_LENGTHS[gene]
        # Domenlərdə mutasiyalar bir qədər daha sıx olsun
        if rng.random() < 0.45 and config.FUNCTIONAL_DOMAINS.get(gene):
            s, e, _ = config.FUNCTIONAL_DOMAINS[gene][
                rng.integers(len(config.FUNCTIONAL_DOMAINS[gene]))]
            pos = int(rng.integers(s, e + 1))
        else:
            pos = int(rng.integers(1, length + 1))

        ref = rng.choice(config.AMINO_ACIDS)
        alt = rng.choice([a for a in config.AMINO_ACIDS if a != ref])
        in_domain = _in_functional_domain(gene, pos)

        # --- latent "zərərlilik" ---
        rp, ap = config.AA_PROPERTIES[ref], config.AA_PROPERTIES[alt]
        blosum = config.BLOSUM62[(ref, alt)]
        latent = (
            -0.55 * blosum
            + 0.25 * abs(rp[0] - ap[0])
            + 0.9 * abs(rp[3] - ap[3])
            + 0.8 * (ref in "GPC" or alt in "GPC")
            + 1.8 * in_domain
            - 2.6
            + rng.normal(0, 1.0)
        )
        label = int(rng.random() < 1 / (1 + np.exp(-latent)))

        # --- evolyusion skorlar (etiketlə korrelyasiyalı, səs-küylü) ---
        sift = np.clip(rng.beta(1.2, 6) if label else rng.beta(3, 2), 0, 1)
        polyphen = np.clip(rng.beta(6, 1.5) if label else rng.beta(1.5, 4), 0, 1)
        cadd = np.clip(rng.normal(27, 4.5) if label else rng.normal(16, 5.5), 0, 50)

        # --- ClinVar formatlı sətir ---
        var_type = "single nucleotide variant"
        significance = "Pathogenic" if label else "Benign"
        r = rng.random()
        if r < 0.05:
            var_type = "Deletion"
        elif r < 0.12:
            significance = rng.choice(["Uncertain significance", "Likely pathogenic",
                                       "Likely benign",
                                       "Conflicting classifications of pathogenicity"])
        protein = f"p.{aa1_to_aa3[ref]}{pos}{aa1_to_aa3[alt]}"
        if rng.random() < 0.06:
            protein = f"p.{aa1_to_aa3[ref]}{pos}Ter"        # nonsense
        elif rng.random() < 0.04:
            protein = f"p.{aa1_to_aa3[ref]}{pos}="          # sinonim

        rows.append({
            "VariationID": 100000 + i,
            "Type": var_type,
            "Name": f"NM_MOCK.{gene}:c.{pos * 3}N>N ({protein})",
            "GeneSymbol": gene,
            "ClinicalSignificance": significance,
            "ReviewStatus": rng.choice(["criteria provided, single submitter",
                                        "criteria provided, multiple submitters, no conflicts",
                                        "reviewed by expert panel"]),
            "Assembly": "GRCh38",
            "sift_score": sift,
            "polyphen_score": polyphen,
            "cadd_phred": cadd,
        })

    df = pd.DataFrame(rows)
    # Çatışmayan dəyərlər (real həyatda VEP/dbNSFP hər variant üçün skor vermir)
    for col in config.SCORE_COLUMNS:
        df.loc[rng.random(len(df)) < missing_rate, col] = np.nan
    print(f"[data] Sintetik ClinVar-formatlı dataset yaradıldı: {len(df):,} sətir")
    return df


# ---------------------------------------------------------------------------
# 1.3  Filtrləmə və hədəf dəyişən
# ---------------------------------------------------------------------------
def parse_protein_change(name: str):
    """'NM_000546.6(TP53):c.742C>T (p.Arg248Trp)' -> ('R', 248, 'W') və ya None."""
    if not isinstance(name, str):
        return None
    m = _PROTEIN_CHANGE_RE.search(name)
    if not m:
        return None
    ref3, pos, alt3 = m.groups()
    ref, alt = config.AA3_TO_AA1.get(ref3), config.AA3_TO_AA1.get(alt3)
    # Ter (stop) və ya naməlum kod -> missense deyil; ref == alt -> sinonim
    if ref is None or alt is None or ref == alt:
        return None
    return ref, int(pos), alt


def preprocess(df: pd.DataFrame, min_review_confidence: bool = True) -> pd.DataFrame:
    """
    SNV + missense + qəti Pathogenic/Benign filtrlərini tətbiq edir və
    təmiz, modelə hazır cədvəl qaytarır:
        gene, position, ref_aa, alt_aa, protein_change, [skorlar], label
    """
    n0 = len(df)
    df = df[df["Type"].str.lower() == "single nucleotide variant"].copy()
    print(f"[prep] SNV filtri:            {n0:,} -> {len(df):,}")

    parsed = df["Name"].map(parse_protein_change)
    df = df[parsed.notna()].copy()
    parsed = parsed[parsed.notna()]
    df["ref_aa"] = [p[0] for p in parsed]
    df["position"] = [p[1] for p in parsed]
    df["alt_aa"] = [p[2] for p in parsed]
    print(f"[prep] Missense filtri:       -> {len(df):,}")

    # Yalnız birmənalı etiketlər ("Pathogenic/Likely pathogenic" kimi qarışıqlar çıxır)
    sig = df["ClinicalSignificance"].str.strip()
    df = df[sig.isin(["Pathogenic", "Benign"])].copy()
    df["label"] = (df["ClinicalSignificance"].str.strip() == "Pathogenic").astype(int)
    print(f"[prep] Qəti Pathogenic/Benign: -> {len(df):,}")

    if min_review_confidence and "ReviewStatus" in df.columns:
        df = df[~df["ReviewStatus"].str.lower().isin(_LOW_CONFIDENCE_REVIEW)]
        print(f"[prep] Review status (≥1★):   -> {len(df):,}")

    df = df.rename(columns={"GeneSymbol": "gene"})
    df["protein_change"] = df["ref_aa"] + df["position"].astype(str) + df["alt_aa"]
    df = df.drop_duplicates(subset=["gene", "protein_change"])

    # Real ClinVar-da SIFT/PolyPhen/CADD yoxdur -> boş sütunlar (sonra imputasiya)
    for col in config.SCORE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    keep = ["gene", "position", "ref_aa", "alt_aa", "protein_change",
            *config.SCORE_COLUMNS, "label"]
    df = df[keep].reset_index(drop=True)
    counts = df["label"].value_counts().to_dict()
    print(f"[prep] Yekun: {len(df):,} variant | Pathogenic={counts.get(1, 0):,} "
          f"Benign={counts.get(0, 0):,}")
    return df


def merge_scores(df: pd.DataFrame, scores_path: Path) -> pd.DataFrame:
    """
    Xarici skor faylını (məs. VEP / dbNSFP ixracı) birləşdirir.
    Gözlənilən CSV sütunları: gene, protein_change, sift_score, polyphen_score, cadd_phred
    (protein_change formatı: R248W)
    """
    scores = pd.read_csv(scores_path)
    df = df.drop(columns=config.SCORE_COLUMNS)
    df = df.merge(scores[["gene", "protein_change", *config.SCORE_COLUMNS]],
                  on=["gene", "protein_change"], how="left")
    cover = df[config.SCORE_COLUMNS].notna().mean().round(3).to_dict()
    print(f"[data] Skorlar birləşdirildi. Doluluq nisbəti: {cover}")
    return df


def get_dataset(source: str = "synthetic", clinvar_path: Path | None = None,
                download: bool = False, n_samples: int = 5000,
                scores_path: Path | None = None) -> pd.DataFrame:
    """Vahid giriş nöqtəsi: mənbədən asılı olmayaraq təmiz dataset qaytarır."""
    if source == "clinvar":
        path = Path(clinvar_path) if clinvar_path else config.CLINVAR_LOCAL
        try:
            if download or not path.exists():
                path = download_clinvar(path)
            raw = load_clinvar(path)
        except Exception as exc:  # şəbəkə və ya fayl problemi
            print(f"[data] ClinVar əlçatan deyil ({exc}). Sintetik dataya keçilir.")
            raw = generate_mock_clinvar(n_samples)
    else:
        raw = generate_mock_clinvar(n_samples)

    df = preprocess(raw)
    if scores_path:
        df = merge_scores(df, Path(scores_path))
    return df
