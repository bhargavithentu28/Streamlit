# Dataset Directory Structure

This directory stores raw and processed electrical equipment images.

## Directory Layout

```
data/
├── raw/
│   ├── transformer/
│   │   ├── img_001.jpg
│   │   └── ...
│   ├── circuit_breaker/
│   ├── insulator/
│   ├── disconnect_switch/
│   ├── current_transformer/
│   └── voltage_transformer/
└── processed/
    ├── train/
    ├── val/
    ├── test/
    ├── class_mapping.json
    └── dataset_summary.json
```

## Supported Equipment Classes

- **Transformer**: Power transformers, distribution transformers, cooling radiators, high-voltage bushings.
- **Circuit Breaker**: High/medium-voltage SF6, air-blast, or vacuum circuit breakers.
- **Insulator**: Ceramic, glass, and composite suspension or post insulators.
- **Disconnect Switch**: Air-break disconnectors and isolators.
- **Current Transformer (CT)**: Instrument transformers for current measurement and protection.
- **Voltage Transformer (VT / PT)**: Instrument transformers for potential/voltage sensing.

## Supported Formats

- `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`

## Preparation Workflow

Run the automated data preparation script:

```powershell
python -m src.data_preparation --input_dir data/raw --output_dir data/processed
```

The script will:
1. Validate all image files (detecting corrupt, truncated, or unreadable files).
2. Check for duplicate or near-duplicate images to prevent leakage across splits.
3. Perform stratified splitting into 70% train, 15% validation, and 15% test sets.
4. Save the canonical class mapping index (`class_mapping.json`).
5. Output a comprehensive summary report (`dataset_summary.json` and Markdown report).
