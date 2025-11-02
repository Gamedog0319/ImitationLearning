# ImitationLearning (MetaMalmo)

Record human gameplay from Minecraft (via Project Malmo) into datasets and play them back for inspection.

---

## Environment

**Minimal env**
```bat
conda env create -f environment.yml
conda activate marlo
```

**Alternative (full spec)**
```bat
conda env create -f environment.full.yml
conda activate marlo
pip install -r requirements.txt
```

---

## Run

### Terminal #1 — Start Minecraft (Malmo client)
```bat
conda activate marlo
set MALMO_MINECRAFT_ROOT=C:\Users\gamed\Malmo-0.37.0-Windows-64bit_withBoost_Python3.6
set MALMO_XSD_PATH=%MALMO_MINECRAFT_ROOT%\Schemas
cd "%MALMO_MINECRAFT_ROOT%\Minecraft"
launchClient.bat -port 10001
```
1. In Minecraft: **Singleplayer → Create New World** (or load any world).
2. **Stay inside the world** (not in menus). Keep this terminal running.

### Terminal #2 — Record or View (project scripts)
```bat
conda activate marlo
cd "C:\Users\gamed\OneDrive - University of Utah\Desktop\Daniel Brown\ImitationLearning\ProjectFiles"

:: Record one (runs ~60s per mission XML)
python record_building.py
python record_navigation.py
python record_farming.py
python record_combat.py

:: View a recorded dataset
python view_building_dataset.py
python view_navigation_dataset.py
python view_farming_dataset.py
python view_combat_dataset.py
```

---

## Data

- Outputs are saved to:
  ```
  ProjectFiles\datasets\*_dataset_hybrid.pkl
  ```
- `datasets/` is **git-ignored** to keep the repo small. A placeholder `datasets/.gitkeep` is tracked so the folder exists.
- ⚠️ `.pkl` files can be very large. Don’t commit them.

---

## Repository Structure (key files)

```
ProjectFiles/
  record_building.py
  record_navigation.py
  record_farming.py
  record_combat.py
  view_building_dataset.py
  view_navigation_dataset.py
  view_farming_dataset.py
  view_combat_dataset.py
  datasets/              # ignored by Git (large outputs)
  .gitignore
  environment.yml
  environment.full.yml
  requirements.txt
  README.md
```

---

## Troubleshooting

- **Client not found:** ensure `MALMO_MINECRAFT_ROOT` points to your Malmo install and `launchClient.bat` exists under `...\Minecraft`.
- **Port mismatch:** scripts expect port `10001`. If you change the client port, update the scripts accordingly.
- **Large files in repo:** if a `.pkl` shows up in Git, add/confirm `datasets/` in `.gitignore` and run:
  ```bat
  git rm -r --cached datasets
  git add .gitignore datasets\.gitkeep
  git commit -m "Stop tracking datasets"
  ```
