#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -euo pipefail

# Check if we are receiving input via a pipe or a file argument
if [ -t 0 ] && [ "$#" -eq 0 ]; then
    echo "Error: No input provided. Please pipe a list of files or pass a file containing paths."
    echo "Usage: ./your_existing_script.sh | ./convert_to_pdf.sh"
    echo "   or: ./convert_to_pdf.sh list_of_files.txt"
    exit 1
fi

# Read input line by line (handles spaces and special characters in filenames safely)
while IFS= read -r file_path || [[ -n "$file_path" ]]; do
    # Skip empty lines
    [[ -z "$file_path" ]] && continue

    # Check if the file actually exists
    if [[ ! -f "$file_path" ]]; then
        echo "Warning: File not found, skipping: $file_path" >&2
        continue
    fi

    # Extract the directory where the file lives
    output_dir=$(dirname "$file_path")

    echo "Converting: '$file_path' ..."

    # Run headless LibreOffice conversion
    # --outdir specifies the exact folder to save the output PDF
    libreoffice --headless \
                --convert-to pdf \
                --outdir "$output_dir" \
                "$file_path"

done < "${1:-/dev/stdin}"

echo "Processing complete!"
