# Background Driven Portrait Relight

A Python portrait relighting pipeline that transfers the lighting impression of a
background image onto a rendered portrait pass set.

The current route is a single `look-safe` pipeline. It keeps face/body exposure
stable while using the background to drive light direction, ambient color, rim
light, contrast and final display finish.

## Structure

```text
main.py                 # CLI entry
interface/cli.py        # argument parsing and path setup
config/                 # constants, paths, background assets
lighting/               # background analysis, light scene, presets
rendering/              # relight pipeline and rendering stages
tools/                  # image, color, filter and geometry helpers
```

## Installation

Python 3.10+ is recommended.

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Minimal runtime dependencies:

```text
numpy
Pillow
tqdm
```

No PyTorch, OpenCV or GPU runtime is required for the current core route.

Optional CUDA acceleration uses CuPy for selected large blur operations:

```bash
pip install -r requirements-gpu.txt
python main.py --device cuda --input-dir E:/your/input_passes --back cyber
```

Use auto mode to fall back to CPU when CuPy/CUDA is unavailable:

```bash
python main.py --device auto --input-dir E:/your/input_passes --back cyber
```

The GPU route is optional and conservative. Image loading/saving still uses
Pillow on CPU, and only selected large array filters are sent to CUDA.

If you use Conda:

```bash
conda create -n render python=3.10 -y
conda activate render
pip install -r requirements.txt
```

## Input Layout

The input directory must contain these pass folders:

```text
Source/
Alpha/
Normal/
Depth/
BaseColor/ or EightColor/ or Color/
```

Optional folders:

```text
Specular/
Roughness/
Camera.json
```

Supported image formats are PNG, JPG, JPEG and WEBP. EXR inputs should be
converted before running the current core route.

## Run

Use the bundled sample input:

```bash
sh run_test.sh --back cyber
```

Use your own input directory:

```bash
sh run_test.sh --input-dir E:/your/input_passes --back cyber
sh run_test.sh --device auto --input-dir E:/your/input_passes --back cyber
```

Choose an exact output directory:

```bash
sh run_test.sh --input-dir E:/your/input_passes --back cyber --output-dir E:/your/results/cyber
```

Direct Python entry:

```bash
python main.py --input-dir E:/your/input_passes --back cyber
```

## Backgrounds

Built-in backgrounds live in:

```text
config/background/
```

Select a background by name or file path:

```bash
python main.py --back cyber
python main.py --back E:/backgrounds/my_bg.png
```

## Outputs

Each run creates:

```text
Render/          # final composite
Relit/           # relit foreground
Cutout/          # transparent foreground cutout
HDRI/            # lighting preview
LightingInfo/    # extracted light data
QualityReport/   # optional quality report
```

The CLI prints the final render folder and the first generated image path after
processing finishes.

## Notes

- `main.py` is the only user-facing entry.
- The default route is `look-safe`; there is no legacy route switch.
- Style expression comes from background analysis, not filename-specific rules.
- Rendering logic is separated into `lighting/`, `rendering/` and `tools/`.
