"""Wanalizer - Intelligent Wallpaper Organization.

Wanalizer automatically sorts wallpaper collections into named category
folders using four pluggable analysis pipelines:

  Low-Level CV  Classical computer vision algorithms for image analysis.
                No ML models required. Fast, deterministic, works on any
                hardware. Uses edge detection, silhouette analysis,
                texture analysis, HOG, FFT, contour moments, colour
                statistics, composition, symmetry, and pattern detection.

  CLIP          OpenAI CLIP zero-shot vision-language model. Matches
                images against category descriptions using semantic
                understanding. Requires PyTorch and the CLIP weights
                download (~150-350 MB).

  Fusion        CLIP + LowLevel CV combined. Runs both pipelines in
                parallel and feeds every signal into a multi-signal
                scorer. Recommended when CLIP is installed. Gracefully
                degrades to LowLevel when CLIP cannot be loaded.

  Ollama        Local vision language models via Ollama (LLaVA,
                MiniCPM-V, Llama 3.2 Vision, Moondream, etc.). Can
                describe, classify, and detect NSFW content with natural
                language.

Usage:
  ./run.sh                Launch the Qt6 graphical interface
  ./run.sh --cli --help   Show CLI options
  wanalizer --help        Show CLI options after pip install
  wanalizer-gui           Launch the GUI after pip install
"""

__version__ = "3.0.0"
__project__ = "Wanalizer"