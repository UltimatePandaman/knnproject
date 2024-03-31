#!/bin/bash

# Adresa URL, ze které se má stáhnout soubor
URL="https://datashare.ed.ac.uk/download/DS_10283_3443.zip"

# Název souboru, do kterého se má uložit
FILENAME="dataset.zip"

# Adresář, do kterého se má rozbalit obsah ZIP archivu
DESTINATION="Data"

# Stáhnout soubor
echo "Stahuji dataset..."
wget "$URL" -O "$FILENAME"

# Ověřit, zda se soubor úspěšně stáhl
if [ $? -ne 0 ]; then
    echo "Chyba při stahování souboru. Zkontrolujte prosím své připojení k internetu a zkuste to znovu."
    exit 1
fi

# Rozbalit ZIP archiv
echo "Rozbaluji dataset..."
unzip "$FILENAME" -d "$DESTINATION"

# Ověřit, zda se archiv úspěšně rozbalil
if [ $? -ne 0 ]; then
    echo "Chyba při rozbalování archivu. Zkontrolujte prosím, zda je soubor ve formátu ZIP a zkuste to znovu."
    exit 1
fi

echo "Dataset byl úspěšně stažen a rozbalen."
