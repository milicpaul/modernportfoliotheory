import requests
import pandas as pd
from bs4 import BeautifulSoup

# URL de la page contenant des tables HTML
url = "https://example.com/page-avec-tables"  # ← à modifier

# 1. Récupération du HTML
response = requests.get(url)
response.raise_for_status()  # pour lever une exception en cas d’erreur réseau
html = response.text

# 2. Extraction des tables via BeautifulSoup
soup = BeautifulSoup(html, "lxml")
tables = soup.find_all("table")

if not tables:
    print("❌ Aucune table trouvée.")
else:
    print(f"✅ {len(tables)} table(s) trouvée(s).")

    # 3. Conversion des tables en DataFrames pandas
    dataframes = []
    for i, table in enumerate(tables):
        df = pd.read_html(str(table))[0]
        dataframes.append(df)
        print(f"\n🔹 Table {i + 1} :")
        print(df.head())

    # Optionnel : sauvegarder les tables dans des fichiers CSV
    for i, df in enumerate(dataframes):
        df.to_csv(f"table_{i + 1}.csv", index=False)
        print(f"💾 table_{i + 1}.csv enregistrée.")
