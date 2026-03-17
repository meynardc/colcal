# colcal — Colour Calibration Tool

A desktop application for **colour-matrix calibration** from a physical colour chart.
Shoot a colour chart alongside your subject, measure the patches, compute a 3×3 (or 3×3 + offset) correction matrix, and apply it to any number of images.

Built with **Python + PySide6 + NumPy**. No subscription, no cloud, no telemetry.

---

## Screenshot

![screenshot](screenshot.jpg)

---

## Features

### Colour chart localisation
- **Manual placement** — click four corners to define the chart quad on the zoomed view.
- **🎯 Auto-detect** — fully automatic detection over the entire image using a hybrid colour-matching + homography pipeline (multi-scale coarse search → Nelder-Mead refinement → iterative fine localisation by connected component → geometric validation and interpolation of missing cells). **The chart must be approximately horizontal and the right way up** — use the rotation buttons beforehand if needed (the algorithm tolerates a few degrees of tilt but not arbitrary angles or upside-down orientations).
- **🔍 Detect in zoom** — same algorithm, but restricted to the currently visible zoom window; useful when the chart occupies a small portion of a large image. Same orientation requirement applies.
- After detection, a joint Nelder-Mead optimisation refines both the quad corners and the inter-cell gap simultaneously.

### Rotation
- **±90° buttons** for quick coarse rotation.
- **Spinbox (0–359°, step 1°)** showing the current angle at all times.
  - Scroll wheel: ±1°
  - Shift + scroll *or* right-click + scroll: ±5°
- The zoom window centre and any existing quad points are automatically converted to the new display space when the rotation changes, so nothing is lost.

### Colour matrix computation
- Least-squares regression: measured patch colours → reference colours.
- **9-parameter** mode: pure 3×3 matrix `M` such that `corrected = M @ rgb`.
- **12-parameter** mode (`+offset` checkbox): affine transform `corrected = M @ rgb + offset`, which additionally corrects a global brightness or white-balance shift.
- Residuals (mean ΔE, per-channel RMS) displayed after computation.

### Preview & export
- **👁 Preview** — live toggle between original and corrected image without recomputing.
- **💾 Save corrected** — export the corrected image as PNG, TIFF, BMP, or JPEG.
- **✨ Correct batch…** — apply the matrix to a folder of images in parallel (multi-threaded), with a progress dialog and a per-file result summary.
- **💾 Export / 📂 Import** — save and reload a correction matrix as a JSON file (`matrix_<imagename>.json`).

### Raw Bayer support
When a raw Bayer mosaic is detected, a debayering bar appears with three algorithms:
- **NN 2×2** — nearest-neighbour (fast, half resolution).
- **Bilinear 3×3** — standard bilinear interpolation (uses OpenCV if available, pure-NumPy fallback).
- **VNG (anti-moiré)** — Variable Number of Gradients, best quality.

Patterns supported: RGGB, BGGR, GRBG, GBRG.

### Zoom viewer
- Mouse drag to pan, scroll wheel to zoom.
- Shift + scroll wheel to adjust the inter-cell gap directly from the viewer.
- Highlighted cells (high std-dev) shown in red.
- Colour tooltip on hover over the measured or reference palette strips.

---

## Requirements

| Package | Version | Notes |
|---------|---------|-------|
| Python | ≥ 3.10 | f-strings, `match`, type unions |
| PySide6 | ≥ 6.4 | Qt 6 bindings |
| NumPy | ≥ 1.22 | core maths |
| OpenCV (`cv2`) | any | *optional* — faster debayering and auto-detect |
| Pillow | any | *optional* — batch processing fallback when cv2 absent |

OpenCV and Pillow are optional but recommended. Without them the application still works fully, using pure-NumPy paths that are somewhat slower.

---

## Installation

```bash
# 1. Clone
git clone https://github.com/your-username/colcal.git
cd colcal

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install PySide6 numpy

# Optional but recommended:
pip install opencv-python-headless pillow
```

---

## Usage

```bash
# Make executable once (Linux / macOS)
chmod +x colcal.py

# Then launch directly
./colcal.py

# Or always via the interpreter
python colcal.py
```

---

## User preferences

colcal stores its preferences (window geometry, last open directories, grid settings, Bayer options) using Qt's `QSettings` in INI format. The file is written automatically on exit.

| Platform | Location |
|----------|----------|
| **Linux** | `~/.config/JoeSoft/colcal.ini` |
| **macOS** | `~/Library/Preferences/JoeSoft/colcal.ini` |
| **Windows** | `%APPDATA%\JoeSoft\colcal\colcal.ini` |

### Resetting preferences

To start fresh (useful after a crash or corrupted state), simply delete the file:

```bash
# Linux
rm ~/.config/JoeSoft/colcal.ini

# macOS
rm ~/Library/Preferences/JoeSoft/colcal.ini

# Windows (PowerShell)
Remove-Item "$env:APPDATA\JoeSoft\colcal\colcal.ini"
```

### Uninstalling

colcal writes nothing outside its own directory except the preferences file above. To fully remove it:

```bash
# 1. Delete the preferences file (see paths above)
rm ~/.config/JoeSoft/colcal.ini   # Linux example

# 2. Delete the application directory
rm -rf /path/to/colcal/

# 3. Optionally remove the virtual environment if you created one
rm -rf /path/to/.venv/
```

### Typical workflow

1. **Open image** — click `📂 Open image` and select the photo that contains the colour chart.
2. **Load palette** — click `📂 Open…` next to the palette strip and select `colorchart.json` (or your own palette file).
3. **Locate the chart** — either:
   - Click `🎯 Auto-detect` to let the algorithm find it automatically, or
   - Zoom in on the chart and click four corners manually in the zoom viewer (top-left → top-right → bottom-right → bottom-left).
4. **Adjust** — fine-tune the grid gap, rows, and columns in the settings panel if needed.
5. **Compute** — click `⚙ Compute` to calculate the correction matrix.
6. **Preview** — click `👁 Preview` to verify the result visually.
7. **Save** — click `💾 Save corrected` for a single image, or `✨ Correct batch…` for a whole folder.

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| `←` / `→` | Rotate image ±90° |
| `↑` / `↓` | Zoom in / out |
| `Delete` / `Backspace` | Clear quad points |

---

## Palette JSON format

Palette files are plain JSON. The `colorchart.json` file included in this repository contains the standard 24-patch **X-Rite ColorChecker Classic** reference values (sRGB, D50).

```json
{
  "name": "My Colour Chart",
  "rows": 4,
  "cols": 6,
  "palette": [
    [ [R, G, B], [R, G, B], ... ],   ← row 0 (top)
    [ [R, G, B], [R, G, B], ... ],   ← row 1
    ...
  ]
}
```

- `palette` is a list of rows, each row a list of `[R, G, B]` values in **0–255 sRGB**.
- Row and column order must match the physical chart as seen when the image is correctly oriented (rotation applied).
- The `name` field is optional and is displayed in the UI.

### Creating your own palette

Any chart with known reference values works. Measure the reference colours from the manufacturer datasheet (sRGB, D50/D65 as appropriate) and build the JSON manually or with a small script.

---

## Matrix JSON format

Exported matrices are plain JSON and can be reloaded or applied programmatically:

```json
{
  "name": "Correction for Canon R5 — studio strobe",
  "matrix": [
    [ 1.0234, -0.0123,  0.0056 ],
    [-0.0089,  1.0312, -0.0201 ],
    [ 0.0034, -0.0145,  1.0087 ]
  ],
  "offset": [2.1, -0.8, 1.3]
}
```

`offset` is optional (present only when the `+offset` mode was used).

### Applying a matrix in Python without the GUI

```python
import json, numpy as np
from PIL import Image

with open("matrix_myphoto.json") as f:
    data = json.load(f)

M      = np.array(data["matrix"], dtype=np.float32)
offset = np.array(data.get("offset", [0, 0, 0]), dtype=np.float32)

img = np.array(Image.open("photo.jpg")).astype(np.float32)
h, w, _ = img.shape
corrected = (img.reshape(-1, 3) @ M.T + offset).clip(0, 255).astype(np.uint8)
Image.fromarray(corrected.reshape(h, w, 3)).save("photo_corrected.png")
```

---

## Repository layout

```
colcal/
├── colcal.py           # main application (single file)
├── colorchart.json     # X-Rite ColorChecker Classic 24-patch reference
└── README.md
```

---

## Licence

MIT — see `LICENSE`.

---

## Acknowledgements

- Colour science fundamentals: *Digital Color Management* (Giorgianni & Madden).
- Reference sRGB values for the ColorChecker Classic: X-Rite / Calibrite.
- UI toolkit: [Qt / PySide6](https://doc.qt.io/qtforpython/).
