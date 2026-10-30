"""
Uçdan-uca pipeline:  Data -> Xüsusiyyətlər -> Təlim -> Qiymətləndirmə -> Proqnoz

Nümunələr:
    python main.py                                   # sintetik data, standart parametrlər
    python main.py --fast                            # sürətli sınaq (az iterasiya)
    python main.py --source clinvar --download       # real ClinVar-ı yüklə və istifadə et
    python main.py --source clinvar --scores-file data/vep_scores.csv
"""
from __future__ import annotations

import argparse
import json

import joblib

from variant_ml import config
from variant_ml.data_loader import get_dataset
from variant_ml.evaluation import evaluate_models
from variant_ml.features import build_features
from variant_ml.models import split_data, train_all
from variant_ml.predict import predict_variant_impact


def parse_args():
    p = argparse.ArgumentParser(description="Missense variantların patogenlik təsnifatı")
    p.add_argument("--source", choices=["synthetic", "clinvar"], default="synthetic")
    p.add_argument("--clinvar-path", default=None, help="Lokal variant_summary.txt.gz yolu")
    p.add_argument("--download", action="store_true", help="ClinVar-ı NCBI-dən yüklə")
    p.add_argument("--scores-file", default=None,
                   help="SIFT/PolyPhen/CADD CSV (gene, protein_change, ...)")
    p.add_argument("--n-samples", type=int, default=5000, help="Sintetik sətir sayı")
    p.add_argument("--n-iter", type=int, default=20, help="RandomizedSearch iterasiyaları")
    p.add_argument("--fast", action="store_true", help="n-iter=5, n-samples=2000")
    return p.parse_args()


def main():
    args = parse_args()
    if args.fast:
        args.n_iter, args.n_samples = 5, 2000

    # ---------- ADDIM 1: Data ----------
    print("\n### ADDIM 1: Verilənlərin toplanması və ilkin emalı")
    df = get_dataset(args.source, args.clinvar_path, args.download,
                     args.n_samples, args.scores_file)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.DATA_DIR / "processed_variants.csv", index=False)

    # ---------- ADDIM 2: Xüsusiyyətlər ----------
    print("\n### ADDIM 2: Xüsusiyyət mühəndisliyi")
    X, y = build_features(df), df["label"]
    print(f"[feat] Xüsusiyyət matrisi: {X.shape[0]:,} x {X.shape[1]}")
    print(f"[feat] Çatışmayan dəyərlər: "
          f"{X[config.SCORE_COLUMNS].isna().mean().round(3).to_dict()}")

    # ---------- ADDIM 3: Təlim ----------
    print("\n### ADDIM 3: Modelləşdirmə və təlim")
    X_train, X_test, y_train, y_test = split_data(X, y)
    models = train_all(X_train, y_train, n_iter=args.n_iter)

    # ---------- ADDIM 4: Qiymətləndirmə ----------
    print("\n### ADDIM 4: Qiymətləndirmə və prediksiya")
    metrics = evaluate_models(models, X_test, y_test, list(X.columns))

    best_name = metrics["ROC-AUC"].idxmax()
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        bundle = {"name": name, "model": model, "feature_columns": list(X.columns)}
        joblib.dump(bundle, config.MODEL_DIR / f"{name.lower().replace(' ', '_')}.joblib")
        if name == best_name:
            joblib.dump(bundle, config.MODEL_DIR / "best_model.joblib")
    print(f"[save] Ən yaxşı model ({best_name}) -> models/best_model.joblib")

    # ---------- Nümunə proqnozlar ----------
    best = {"name": best_name, "model": models[best_name], "feature_columns": list(X.columns)}
    examples = [
        dict(mutation="R248W", gene="TP53", sift=0.0, polyphen=1.0, cadd=32.0),
        dict(mutation="R248W", gene="TP53"),                       # skorsuz -> imputasiya
        dict(mutation="p.Val217Ile", gene="TP53", sift=0.45, polyphen=0.05, cadd=9.5),
    ]
    print("\n[predict] Nümunə proqnozlar:")
    for ex in examples:
        print(json.dumps(predict_variant_impact(model=best, **ex), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
