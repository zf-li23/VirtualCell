#!/bin/bash
# Copy _template.md to all unfilled note README.md files
# Filled notes that should NOT be overwritten
FILLED=(
  "fm/geneformer"
  "fm/scgpt"
  "fm/scfoundation"
  "fm/scprint"
  "fm/nicheformer"
  "fm/novae"
  "fm/scbert"
  "fm/saturn"
  "fm/cell-atlas-fm"
  "fm/genecompass"
  "fm/epiagent"
  "fm/scpoli"
  "fm/visual-omics-fm"
)

TEMPLATE="notes/_template.md"
NOTES_DIR="notes"

# Get all subdirectories that contain a README.md
find "$NOTES_DIR" -name "README.md" | while read -r readme; do
  dir=$(dirname "$readme")
  rel_dir="${dir#$NOTES_DIR/}"
  
  # Skip if it's a filled note
  skip=false
  for filled in "${FILLED[@]}"; do
    if [ "$rel_dir" = "$filled" ]; then
      skip=true
      break
    fi
  done
  
  if [ "$skip" = false ]; then
    echo "Copying template to: $readme"
    cp "$TEMPLATE" "$readme"
  else
    echo "Skipping filled note: $readme"
  fi
done

echo "Done!"
