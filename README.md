# Eksperimen_SML_Devin-Novansyah

Repositori eksperimen Machine Learning untuk dataset **Titanic - Machine Learning from Disaster**.

## Struktur Folder

```
Eksperimen_SML_Devin-Novansyah/
├── .github/
│   └── workflows/
│       └── preprocessing.yml       ← GitHub Actions workflow
├── titanic_raw/
│   └── train.csv                   ← Dataset mentah (download dari Kaggle)
└── preprocessing/
    ├── Eksperimen_Devin-Novansyah.ipynb    ← Notebook eksperimen
    ├── automate_Devin-Novansyah.py         ← Script otomatisasi preprocessing
    └── titanic_preprocessing/
        └── train_preprocessed.csv          ← Output hasil preprocessing
```

## Cara Penggunaan

### 1. Download Dataset
Download `train.csv` dari [Kaggle Titanic](https://www.kaggle.com/c/titanic/data) dan letakkan di folder `titanic_raw/`.

### 2. Jalankan Preprocessing Manual (Notebook)
Buka dan jalankan `preprocessing/Eksperimen_Devin-Novansyah.ipynb`.

### 3. Jalankan Script Otomatis
```bash
pip install pandas numpy scikit-learn
python preprocessing/automate_Devin-Novansyah.py
```

### 4. GitHub Actions (Otomatis)
Workflow akan terpantik otomatis ketika ada perubahan pada:
- `titanic_raw/` (dataset baru)
- `preprocessing/automate_Devin-Novansyah.py` (script diupdate)

Atau bisa dijalankan manual lewat tab **Actions** di GitHub.

## Tahapan Preprocessing

| No | Tahap | Keterangan |
|----|-------|------------|
| 1 | Seleksi Fitur | Drop: PassengerId, Name, Ticket, Cabin |
| 2 | Handling Missing | Age → median, Embarked → modus |
| 3 | Feature Engineering | FamilySize, IsAlone, AgeGroup |
| 4 | Encoding | Sex (Label), Embarked & AgeGroup (One-Hot) |
| 5 | Scaling | Age, Fare, FamilySize → StandardScaler |

## Author
**Devin Novansyah** - Machine Learning Submission
