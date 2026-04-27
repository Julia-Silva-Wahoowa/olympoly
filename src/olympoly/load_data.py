from datasets import load_dataset
import os
from dotenv import load_dotenv
import pandas as pd
import requests
from pathlib import Path

load_dotenv()


def load_raw_olympic_data():
    dataset = load_dataset("Haider67795/veriseti_20220203_olimpiyatlar",
                           data_files="veriseti_20220203_olimpiyatlar.csv")
    return dataset['train'].to_pandas()


def get_cleaned_data(season="Summer"):
    """
    Centralized data cleaning. Fixes NaNs, types, and filters by season 
    to prevent Summer/Winter overlap issues in graphs.
    """
    df = load_raw_olympic_data()

    df['Medal'] = df['Medal'].replace('NA', pd.NA)

    for col in ['Year', 'Age', 'Height', 'Weight']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['ID', 'Team', 'Year'])

    df['Team'] = df['Team'].str.strip()
    df['Sport'] = df['Sport'].str.strip()

    if 'Season' in df.columns and season:
        df = df[df['Season'] == season]

    return df


def load_data():
    """Wrapper for legacy support so other files don't break."""
    return get_cleaned_data()

def load_olympic_data():
    """Legacy wrapper for old tests."""
    return get_cleaned_data()


def load_demographic_data():
    API_KEY = os.getenv("API_KEY")
    BASE_URL = "https://api.census.gov/data/timeseries/idb/5year"
    params = {
        "get": "NAME,GENC,POP,TFR,E0,IMR,GR,NMR,CBR,CDR",
        "YR": "2026",
        "for": "genc standard countries and areas:*",
        "key": API_KEY
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    df = pd.DataFrame(data[1:], columns=data[0])
    df = df.drop(columns=["genc standard countries and areas"])
    for col in ["POP", "TFR", "E0", "IMR", "GR", "NMR", "CBR", "CDR"]:
        df[col] = pd.to_numeric(df[col])
    return df
# Load data of a list of host countries 

def load_host_data():
    BASE_DIR = Path(__file__).resolve().parent
    df = pd.read_csv(BASE_DIR / "Host_Countries.csv")
    return df

if __name__ == "__main__":
    data = load_host_data()
    print(data.head(10))
