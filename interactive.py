"""
İnteraktiv rejim: mutasiyanı özünüz yazın, model dərhal cavab versin.

İşlətmək:
    python interactive.py

Əvvəlcə `python main.py --fast` bir dəfə işlədilməlidir ki, model saxlanılsın.
Model tapılmasa, bu skript onu avtomatik öyrədəcək.
"""
from variant_ml import config
from variant_ml.predict import load_model, predict_variant_impact

MODEL_PATH = config.MODEL_DIR / "best_model.joblib"


def _ensure_model():
    if not MODEL_PATH.exists():
        print("Saxlanmış model tapılmadı, indi öyrədilir (təxminən 1 dəqiqə)...\n")
        import sys
        import main
        sys.argv = ["main.py", "--fast"]
        main.main()
    return load_model(MODEL_PATH)


def _ask_float(prompt: str):
    raw = input(prompt).strip().replace(",", ".")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        print("   (rəqəm deyil, nəzərə alınmadı)")
        return None


def _bar(p: float, width: int = 30) -> str:
    filled = int(round(p * width))
    return "█" * filled + "░" * (width - filled)


def main():
    model = _ensure_model()
    genes = ", ".join(config.FUNCTIONAL_DOMAINS)

    print("=" * 62)
    print("  GENETİK VARİANT ZƏRƏRLİLİK PROQNOZU  (model: " + model["name"] + ")")
    print("=" * 62)
    print(f"  Domen məlumatı olan genlər: {genes}")
    print("  Mutasiya nümunələri: R248W, R175H, V217I, p.Arg273His")
    print("  Çıxmaq üçün mutasiya yerinə 'q' yazın.")
    print("=" * 62)

    while True:
        print()
        gene = input("Gen adı [Enter = TP53]: ").strip().upper() or "TP53"
        if gene == "Q":
            break
        mutation = input("Mutasiya (məs. R248W): ").strip()
        if mutation.lower() in {"q", "quit", "exit", "çıx"}:
            break
        if not mutation:
            continue

        print("Skorlar məlumdursa daxil edin, bilmirsinizsə Enter basın:")
        sift = _ask_float("   SIFT (0-1, kiçik = zərərli): ")
        polyphen = _ask_float("   PolyPhen (0-1, böyük = zərərli): ")
        cadd = _ask_float("   CADD (0-50, böyük = zərərli): ")

        try:
            r = predict_variant_impact(mutation, gene=gene, model=model,
                                       sift=sift, polyphen=polyphen, cadd=cadd)
        except ValueError as e:
            print(f"\n   Xəta: {e}")
            continue

        p = r["pathogenic_probability"]
        print("\n" + "-" * 62)
        print(f"  Variant            : {r['variant']}")
        print(f"  Zərərli ehtimalı   : {p * 100:6.2f}%  {_bar(p)}")
        print(f"  Proqnoz            : {r['prediction']}")
        print(f"  Risk kateqoriyası  : {r['risk_category']}")
        print(f"  Zülal domeni       : {r['domain'] or 'məlum funksional domendə deyil'}")
        print(f"  BLOSUM62           : {r['blosum62']}  (mənfi = radikal əvəzetmə)")
        if sift is None and polyphen is None and cadd is None:
            print("  Qeyd               : skorlar verilmədi, orta dəyərlərlə hesablandı")
        print("-" * 62)

    print("\nSağ olun!")


if __name__ == "__main__":
    main()
