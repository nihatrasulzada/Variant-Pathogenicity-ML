# Genetik Variantların ML ilə Təsnifatı və Zərərli Mutasiyaların Proqnozlaşdırılması

Missense variantları ClinVar etiketləri əsasında **Pathogenic (1) / Benign (0)** kimi təsnif edən
Random Forest və XGBoost pipeline-ı.

## Layihə strukturu

```
variant-pathogenicity-ml/
├── main.py                  # Uçdan-uca pipeline (4 addım)
├── requirements.txt
├── variant_ml/
│   ├── config.py            # Yollar, AA xassələri, BLOSUM62, domen cədvəli
│   ├── data_loader.py       # ADDIM 1: ClinVar yükləmə / mock data, SNV+missense+etiket filtri
│   ├── features.py          # ADDIM 2: AA xüsusiyyətləri, skorlar, domen bayrağı
│   ├── models.py            # ADDIM 3: 80/20 split, RF + XGBoost, RandomizedSearchCV
│   ├── evaluation.py        # ADDIM 4a: metrikalar, ROC, confusion matrix, feature importance
│   └── predict.py           # ADDIM 4b: predict_variant_impact()
├── data/                    # ClinVar faylı və emal olunmuş CSV (avtomatik yaranır)
├── models/                  # Saxlanmış modellər (.joblib)
└── outputs/                 # Qrafiklər və metrika CSV-ləri
```

## Quraşdırma

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## İcra

```bash
python main.py --fast                         # ~30 san. sürətli sınaq (sintetik data)
python main.py                                # tam işləmə (sintetik, 5000 sətir, 20 iterasiya)
python main.py --source clinvar --download    # real ClinVar-ı NCBI-dən yüklə (~400 MB)
python main.py --source clinvar --clinvar-path data/variant_summary.txt.gz
python main.py --source clinvar --scores-file data/vep_scores.csv
```

ClinVar əlçatan olmasa, kod avtomatik olaraq eyni formatlı sintetik dataya keçir.

## Proqnoz (inference)

```python
from variant_ml.predict import load_model, predict_variant_impact

model = load_model()   # models/best_model.joblib
predict_variant_impact("R248W", gene="TP53", model=model, sift=0.0, polyphen=1.0, cadd=32)
# {'variant': 'TP53 p.R248W', 'pathogenic_probability': 0.9999, 'prediction': 'Pathogenic', ...}

predict_variant_impact("p.Arg175His", gene="TP53", model=model)   # skorsuz -> median imputasiya
```

Qəbul olunan formatlar: `R248W`, `p.R248W`, `Arg248Trp`, `p.Arg248Trp`.

## Vacib qeydlər

- **Sintetik nəticələr real performansı göstərmir.** Mock datada SIFT/PolyPhen/CADD etiketdən
  törədilir, ona görə AUC ≈ 0.997 süni yüksəkdir. Kod düzgünlüyünü yoxlamaq üçündür.
- **Real ClinVar-da SIFT/PolyPhen/CADD sütunları yoxdur.** Onları Ensembl VEP və ya dbNSFP ilə
  annotasiya edib `--scores-file` ilə verin (sütunlar: `gene, protein_change, sift_score,
  polyphen_score, cadd_phred`; `protein_change` formatı `R248W`). Verilməsə, model yalnız
  amin turşusu və domen xüsusiyyətləri ilə öyrənir.
- **Domen cədvəli** (`config.FUNCTIONAL_DOMAINS`) təxmini və nümunə xarakterlidir; real iş üçün
  UniProt / InterPro / Pfam koordinatlarından istifadə edin.
- **Data sızması:** eyni gendən variantlar həm təlimdə, həm sınaqda ola bilər. Daha sərt
  qiymətləndirmə üçün gen səviyyəsində bölgü (`GroupKFold`) tövsiyə olunur.
- Model klinik diaqnoz vasitəsi deyil; tədqiqat və təhsil məqsədlidir.
