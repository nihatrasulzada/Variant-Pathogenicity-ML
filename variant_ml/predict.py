"""
ADDIM 4b: Inference — yeni, naməlum variantın zərərlilik ehtimalının proqnozu.

İstifadə:
    from variant_ml.predict import load_model, predict_variant_impact
    model = load_model()                       # models/best_model.joblib
    predict_variant_impact("R248W", gene="TP53", model=model,
                           sift=0.0, polyphen=1.0, cadd=32)
"""
from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config
from .features import build_features, domain_name

_ONE_LETTER = re.compile(r"^(?:p\.)?([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])$")
_THREE_LETTER = re.compile(r"^(?:p\.)?([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$")


def parse_mutation(mutation: str) -> tuple[str, int, str]:
    """'R248W', 'p.R248W', 'Arg248Trp', 'p.Arg248Trp' formatlarını qəbul edir."""
    s = mutation.strip()
    if s[:2].lower() == "p.":
        s = s[2:]
    m = _ONE_LETTER.match(s.upper()) if not _THREE_LETTER.match(s) else None
    if m:
        ref, pos, alt = m.groups()
    else:
        m = _THREE_LETTER.match(s)
        if not m:
            raise ValueError(f"Mutasiya formatı tanınmadı: '{mutation}' (gözlənilən: R248W)")
        ref3, pos, alt3 = m.groups()
        ref, alt = config.AA3_TO_AA1.get(ref3), config.AA3_TO_AA1.get(alt3)
        if ref is None or alt is None:
            raise ValueError(f"Naməlum amin turşusu kodu: '{mutation}'")
    if ref == alt:
        raise ValueError("Sinonim dəyişiklikdir (ref == alt), missense deyil.")
    return ref, int(pos), alt


def load_model(path: Path = config.MODEL_DIR / "best_model.joblib"):
    bundle = joblib.load(path)
    return bundle


def _risk_category(p: float) -> str:
    if p >= 0.90:
        return "Yüksək ehtimalla zərərli (Likely Pathogenic)"
    if p >= 0.60:
        return "Mümkün zərərli (Possibly Pathogenic)"
    if p > 0.40:
        return "Qeyri-müəyyən (VUS zonası)"
    if p > 0.10:
        return "Mümkün zərərsiz (Possibly Benign)"
    return "Yüksək ehtimalla zərərsiz (Likely Benign)"


def predict_variant_impact(mutation: str, gene: str = "TP53", model=None,
                           sift: float | None = None, polyphen: float | None = None,
                           cadd: float | None = None, in_domain: int | None = None,
                           threshold: float = 0.5) -> dict:
    """
    Naməlum missense variantın patogenlik ehtimalını qaytarır.

    Parametrlər
    ----------
    mutation  : 'R248W' / 'p.Arg248Trp'
    gene      : gen simvolu (domen cədvəlində axtarış üçün)
    model     : load_model() nəticəsi (dict bundle) və ya sklearn Pipeline
    sift, polyphen, cadd : məlumdursa evolyusion skorlar; None -> imputasiya
    in_domain : domen bayrağını əl ilə təyin etmək (None -> cədvəldən avtomatik)
    """
    if model is None:
        model = load_model()
    if isinstance(model, dict):
        model_name, columns, pipe = model["name"], model["feature_columns"], model["model"]
    else:
        model_name, columns, pipe = type(model).__name__, None, model

    ref, pos, alt = parse_mutation(mutation)
    row = pd.DataFrame([{
        "gene": gene, "position": pos, "ref_aa": ref, "alt_aa": alt,
        "sift_score": np.nan if sift is None else sift,
        "polyphen_score": np.nan if polyphen is None else polyphen,
        "cadd_phred": np.nan if cadd is None else cadd,
    }])
    X = build_features(row)
    if in_domain is not None:
        X["in_functional_domain"] = float(in_domain)
    if columns is not None:
        X = X[columns]

    proba = float(pipe.predict_proba(X)[0, 1])
    return {
        "variant": f"{gene} p.{ref}{pos}{alt}",
        "model": model_name,
        "pathogenic_probability": round(proba, 4),
        "prediction": "Pathogenic" if proba >= threshold else "Benign",
        "risk_category": _risk_category(proba),
        "in_functional_domain": int(X["in_functional_domain"].iloc[0]),
        "domain": domain_name(gene, pos),
        "blosum62": int(X["blosum62"].iloc[0]),
        "scores_provided": {"sift": sift, "polyphen": polyphen, "cadd": cadd},
    }
