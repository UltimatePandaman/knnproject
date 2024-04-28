#!/bin/bash

# ID souboru na Google Drive
FILE_ID="1nzTyyl-9A1Hmqya2Q_f2bpZkUoRjbZsY"

# Directory where the contents of the ZIP archive should be extracted
ZIPFILE="models.zip"

# Directory where the contents of the ZIP archive should be extracted
DESTINATION="Models"

# Instalace gdown pokud ještě není nainstalován
if ! command -v gdown &> /dev/null; then
    echo "Installing gdown..."
    pip install gdown
fi

# Download the file
echo "Downloading models..."
gdown --id "$FILE_ID" --output "$ZIPFILE"

# Check if the file was downloaded successfully
if [ $? -ne 0 ]; then
    echo "Error downloading the file. Please check your internet connection and try again."
    exit 1
fi

if [ -d "$DESTINATION" ]; then
    echo "Removing existing $DESTINATION directory..."
    rm -rf "$DESTINATION"
fi

# Extract the ZIP archive
echo "Extracting models..."
unzip -q "$ZIPFILE"

# Check if the archive was extracted successfully
if [ $? -ne 0 ]; then
    echo "Error extracting the archive. Please check if the file is in ZIP format and try again."
    exit 1
fi

# Delete the ZIP file after extraction
echo "Removing the ZIP file..."
rm -fv "$ZIPFILE"

echo "$DESTINATION was downloaded and extracted successfully."