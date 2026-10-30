"""
ADDIM 2: Xüsusiyyət mühəndisliyi (Feature Engineering).

  a) Amin turşusu xüsusiyyətləri  : ref/alt fiziki-kimyəvi xassələr, onların
                                     fərqləri, BLOSUM62, xüsusi sinif dəyişmələri.
  b) Evolyusion skorlar            : SIFT, PolyPhen, CADD (rəqəmsal sütunlar).
                                     İmputasiya model pipeline-ında edilir
                                     (bax: models.py) ki, test datasından
                                     məlumat sızması (data leakage) olmasın.
  c) Zülal domeni                  : mutasiya funksional/katalitik domendədirmi (1/0).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

PROP_IDX = {name: i for i, name in enumerate(config.AA_PROPERTY_NAMES)}


def in_functional_domain(gene: str, position: int) -> int:
    """Mutasiya məlum funksional domendə yerləşirsə 1, əks halda 0."""
    for start, end, _ in config.FUNCTIONAL_DOMAINS.get(str(gene), []):
        if start <= int(position) <= end:
            return 1
    return 0


def domain_name(gene: str, position: int) -> str | None:
    for start, end, name in config.FUNCTIONAL_DOMAINS.get(str(gene), []):
        if start <= int(position) <= end:
            return name
    return None


def _aa_features(ref: str, alt: str) -> dict:
    rp, ap = config.AA_PROPERTIES[ref], config.AA_PROPERTIES[alt]
    f = {}
    # Hər iki amin turşusunun mütləq xassələri
    for name, i in PROP_IDX.items():
        f[f"ref_{name}"] = rp[i]
        f[f"alt_{name}"] = ap[i]
    # Dəyişikliyin miqyası
    f["delta_hydrophobicity"] = ap[PROP_IDX["hydrophobicity"]] - rp[PROP_IDX["hydrophobicity"]]
    f["abs_delta_hydrophobicity"] = abs(f["delta_hydrophobicity"])
    f["delta_volume"] = ap[PROP_IDX["volume"]] - rp[PROP_IDX["volume"]]
    f["abs_delta_volume"] = abs(f["delta_volume"])
    f["delta_mw"] = ap[PROP_IDX["mw"]] - rp[PROP_IDX["mw"]]
    f["delta_charge"] = ap[PROP_IDX["charge"]] - rp[PROP_IDX["charge"]]
    f["charge_change"] = int(abs(f["delta_charge"]) >= 0.5)
    f["charge_reversal"] = int(rp[PROP_IDX["charge"]] * ap[PROP_IDX["charge"]] < -0.5)
    f["polarity_change"] = int(rp[PROP_IDX["polar"]] != ap[PROP_IDX["polar"]])
    f["aromatic_change"] = int(rp[PROP_IDX["aromatic"]] != ap[PROP_IDX["aromatic"]])
    # Evolyusion əvəzetmə skoru
    f["blosum62"] = config.BLOSUM62[(ref, alt)]
    # Struktur baxımından xüsusi amin turşuları
    f["involves_glycine"] = int("G" in (ref, alt))
    f["involves_proline"] = int("P" in (ref, alt))
    f["involves_cysteine"] = int("C" in (ref, alt))
    return f


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Təmiz variant cədvəlindən (gene, position, ref_aa, alt_aa, skorlar)
    modelin istifadə etdiyi rəqəmsal xüsusiyyət matrisini qurur.
    """
    aa = pd.DataFrame([_aa_features(r, a) for r, a in zip(df["ref_aa"], df["alt_aa"])],
                      index=df.index)

    domain = pd.DataFrame({
        "in_functional_domain": [in_functional_domain(g, p)
                                 for g, p in zip(df["gene"], df["position"])],
    }, index=df.index)

    # Evolyusion skorlar: rəqəmsal tipə çevir, NaN-lar saxlanılır (pipeline imputasiya edir)
    scores = df[config.SCORE_COLUMNS].apply(pd.to_numeric, errors="coerce")

    X = pd.concat([aa, scores, domain], axis=1)
    return X.astype(float)


def feature_columns() -> list[str]:
    """Xüsusiyyətlərin sabit sırası (inference zamanı eyni sıra lazımdır)."""
    dummy = pd.DataFrame({"gene": ["X"], "position": [1], "ref_aa": ["A"], "alt_aa": ["V"],
                          **{c: [np.nan] for c in config.SCORE_COLUMNS}})
    return list(build_features(dummy).columns)
