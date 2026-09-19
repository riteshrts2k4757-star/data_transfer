# Paired audio cleaning pipeline

This project keeps the original paired dataset in `data/` untouched and writes
standardized fixed-length examples to `cleaned_data/`.

## Install and run

```powershell
myenv\Scripts\python.exe -m pip install -r requirements.txt
myenv\Scripts\python.exe scripts\clean_dataset.py
```

Use a dry run to analyze and log rejected pairs without creating output WAVs:

```powershell
myenv\Scripts\python.exe scripts\clean_dataset.py --dry-run
```

Optional overrides:

```powershell
myenv\Scripts\python.exe scripts\clean_dataset.py --input data --output cleaned_data --sample-rate 16000 --chunk-duration 4 --silence-threshold -40
```

The cleaner matches noisy and clean files by stem, rejects the complete pair
when either side is unreadable, clipped, too short, or badly duration-misaligned,
and applies one shared normalization gain to both signals. Silence boundaries
are found from the clean reference and the same time interval is applied to both
signals. Internal pauses are preserved. The last partial chunk is discarded by
default. Change the configuration block in `scripts/clean_dataset.py` to use
padding instead.

Reports and rejection logs are written to `logs/`. The final metadata is
`cleaned_data/metadata.csv`. Optional waveform plots can be generated with:

```powershell
myenv\Scripts\python.exe scripts\analyze_dataset.py
```
