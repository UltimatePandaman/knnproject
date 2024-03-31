#!/bin/bash

# URL from which the file is to be downloaded
URL="https://datashare.ed.ac.uk/download/DS_10283_3443.zip"

# Name of the file to be saved
FILENAME="dataset.zip"

# Directory where the contents of the ZIP archive should be extracted
DESTINATION="Data"

# Download the file
echo "Downloading dataset..."
wget "$URL" -O "$FILENAME"

# Check if the file was downloaded successfully
if [ $? -ne 0 ]; then
    echo "Error downloading the file. Please check your internet connection and try again."
    exit 1
fi

# Extract the ZIP archive
echo "Extracting dataset..."
unzip "$FILENAME" -d "$DESTINATION"

# Check if the archive was extracted successfully
if [ $? -ne 0 ]; then
    echo "Error extracting the archive. Please check if the file is in ZIP format and try again."
    exit 1
fi

# Delete the ZIP file after extraction
echo "Removing the ZIP file..."
rm -fv "$FILENAME"

echo "Dataset was downloaded and extracted successfully."