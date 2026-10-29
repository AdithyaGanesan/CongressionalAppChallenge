# Colab cells

## Extract ASL STEM Wiki landmarks (`extract_stemwiki.py`)

Paste this into a new Colab notebook, set `SLICE`, and run it. It downloads the latest version of
the extraction code from the `next` branch and runs it.

```python
SLICE = 0        # 0 in the first notebook, 1 in the second, 2 in the third
import urllib.request
exec(urllib.request.urlopen("https://raw.githubusercontent.com/AdithyaGanesan/CongressionalAppChallenge/next/colab/extract_stemwiki.py").read().decode())
```

Optional settings (put them above the `import` line): `NUM_SLICES = 3` (how many notebooks run at
once; must be the same in all of them), `TARGET_HOURS = 40`, `WORKERS = 2`.

Safe to re-run after a disconnect: finished videos are skipped. Output goes to
`stat_stemwiki` in your Google Drive. Never put it on GitHub.

## Export existing landmarks (`export_landmarks.py`)

Packages the ASL Citizen landmarks in Drive into one zip to share with a Claude Code session.
