# src/io_utils.py
from pathlib import Path
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Define project root (folder that holds Data/, Notebooks/, Models/, etc.)
ROOT = Path(__file__).resolve().parents[1]

# Paths for convenience
DATA = ROOT / "Data"
MODELS = ROOT / "Models"
REPORTS = ROOT / "Reports"
FIGURES = REPORTS / "figures"

def load_csv(filename, folder="raw"):
    """Load a CSV from Data/raw by default, can specify Data/processed as well."""
    path = DATA / folder / filename
    return pd.read_csv(path)

def save_csv(df, filename, folder="processed"):
    """Save a CSV to Data/processed by default."""
    path = DATA / folder / filename
    df.to_csv(path, index=False)
    print(f"Saved: {path}")

def save_pkl(obj, filename, folder="processed"):
    """Save any Python object (e.g., DataFrame) to Data/processed/."""
    path = DATA / folder / filename
    joblib.dump(obj, path)
    print(f"Pickle saved: {path}")

def load_pkl(filename, folder="processed"):
    """Load a pickle from Data/processed/."""
    path = DATA / folder / filename
    return joblib.load(path)

def save_model(model, filename):
    """Save a model to Models/."""
    path = MODELS / filename
    joblib.dump(model, path)
    print(f"Model saved: {path}")

def load_model(filename):
    """Load a model from Models/."""
    path = MODELS / filename
    return joblib.load(path)

def save_fig(obj, filename, folder=FIGURES, dpi=300):
    """
    Save a matplotlib figure to reports by default.
    - obj : matplotlib.figure.Figure OR matplotlib.pyplot
        - If a Figure object, save directly
        - If pyplot (plt), grab the current figure with plt.gcf()
    - filename: e.g. 'roc_curve.png'
    - folder: defaults to reports
    """
    path = folder / filename                   # build the full save path
    ## Case 1: Figure object
    if isinstance(obj, type(plt.gcf())):
        obj.savefig(path, dpi=dpi, bbox_inches="tight")
    else:
        # Case 2: assume it's pyplot (plt)
        obj.gcf().savefig(path, dpi=dpi, bbox_inches="tight")

    print(f"Figure saved: {path}")