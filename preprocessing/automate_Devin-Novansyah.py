"""
automate_Devin-Novansyah.py
============================
Script otomatisasi preprocessing dataset Titanic.
Mengonversi langkah-langkah eksperimen dari notebook menjadi
pipeline yang dapat dijalankan secara otomatis.

Author  : Devin Novansyah
Dataset : Titanic - Machine Learning from Disaster
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')


# ──────────────────────────────────────────────
# KONFIGURASI PATH
# ──────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH  = os.path.join(BASE_DIR, 'titanic_raw', 'train.csv')
OUTPUT_DIR  = os.path.join(BASE_DIR, 'preprocessing', 'titanic_preprocessing')
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'train_preprocessed.csv')


# ──────────────────────────────────────────────
# FUNGSI-FUNGSI PREPROCESSING
# ──────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Memuat dataset dari filepath yang diberikan."""
    print(f'[1/6] Memuat data dari: {filepath}')
    df = pd.read_csv(filepath)
    print(f'      ✅ Data berhasil dimuat. Shape: {df.shape}')
    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Menghapus kolom yang tidak relevan untuk pemodelan."""
    print('[2/6] Seleksi fitur...')
    cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    df = df.drop(columns=cols_to_drop)
    print(f'      ✅ Kolom dihapus: {cols_to_drop}')
    print(f'      Kolom tersisa: {df.columns.tolist()}')
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Menangani missing values pada kolom Age dan Embarked."""
    print('[3/6] Menangani missing values...')

    # Age → median
    age_median = df['Age'].median()
    df['Age'] = df['Age'].fillna(age_median)
    print(f'      Age: imputasi dengan median = {age_median:.2f}')

    # Embarked → modus
    embarked_mode = df['Embarked'].mode()[0]
    df['Embarked'] = df['Embarked'].fillna(embarked_mode)
    print(f'      Embarked: imputasi dengan modus = {embarked_mode!r}')

    missing_total = df.isnull().sum().sum()
    print(f'      ✅ Total missing values setelah penanganan: {missing_total}')
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Membuat fitur-fitur baru dari fitur yang sudah ada."""
    print('[4/6] Feature engineering...')

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone']    = (df['FamilySize'] == 1).astype(int)
    df['AgeGroup']   = pd.cut(
        df['Age'],
        bins=[0, 12, 18, 35, 60, 100],
        labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior']
    )

    print('      ✅ Fitur baru: FamilySize, IsAlone, AgeGroup')
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encoding fitur kategorikal (Label & One-Hot Encoding)."""
    print('[5/6] Encoding fitur kategorikal...')

    # Label Encoding: Sex
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    print('      Sex → Label Encoding (female=0, male=1)')

    # One-Hot Encoding: Embarked
    embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked', drop_first=True)
    df = pd.concat([df, embarked_dummies], axis=1)
    df = df.drop(columns=['Embarked'])
    print('      Embarked → One-Hot Encoding')

    # One-Hot Encoding: AgeGroup
    agegroup_dummies = pd.get_dummies(df['AgeGroup'], prefix='AgeGroup', drop_first=True)
    df = pd.concat([df, agegroup_dummies], axis=1)
    df = df.drop(columns=['AgeGroup'])
    print('      AgeGroup → One-Hot Encoding')

    # Pastikan semua kolom bool menjadi int
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    print(f'      ✅ Shape setelah encoding: {df.shape}')
    return df


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """Standardisasi fitur numerik menggunakan StandardScaler."""
    print('[6/6] Feature scaling...')

    cols_to_scale = ['Age', 'Fare', 'FamilySize']
    scaler = StandardScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    print(f'      ✅ Fitur yang di-scale: {cols_to_scale}')
    return df


def save_data(df: pd.DataFrame, output_path: str) -> None:
    """Menyimpan DataFrame hasil preprocessing ke file CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f'\n✅ Data preprocessed berhasil disimpan ke: {output_path}')
    print(f'   Shape  : {df.shape}')
    print(f'   Kolom  : {df.columns.tolist()}')


# ──────────────────────────────────────────────
# PIPELINE UTAMA
# ──────────────────────────────────────────────

def run_preprocessing(input_path: str = INPUT_PATH,
                      output_path: str = OUTPUT_PATH) -> pd.DataFrame:
    """
    Menjalankan seluruh pipeline preprocessing secara berurutan.

    Parameters
    ----------
    input_path  : str – path ke file CSV mentah
    output_path : str – path tujuan file CSV hasil preprocessing

    Returns
    -------
    pd.DataFrame – data yang sudah siap dilatih
    """
    print('=' * 55)
    print('  PIPELINE PREPROCESSING - TITANIC DATASET')
    print('  Author: Devin Novansyah')
    print('=' * 55)

    df = load_data(input_path)
    df = select_features(df)
    df = handle_missing_values(df)
    df = feature_engineering(df)
    df = encode_features(df)
    df = scale_features(df)
    save_data(df, output_path)

    print('\n✅ Preprocessing selesai!')
    print('=' * 55)
    return df


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == '__main__':
    run_preprocessing()
