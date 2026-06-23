from setuptools import setup

setup(
    name="wanalyzer",
    version="3.0.0",
    description="Wanalizer - Intelligent wallpaper organizer with 4 analysis modes: Low-Level CV, CLIP, CLIP+LowLevel fusion, and Ollama vision LLMs.",
    long_description="Wanalizer automatically sorts wallpaper collections into named "
                    "category folders using classical computer vision, CLIP zero-shot "
                    "models, CLIP+CV fusion, or local Ollama vision LLMs.",
    long_description_content_type="text/plain",
    license="MIT",
    python_requires=">=3.10",
    install_requires=[
        "Pillow>=9.0.0",
        "numpy>=1.24.0",
        "requests>=2.28.0",
    ],
    extras_require={
        "gui": ["PySide6>=6.6.0"],
        "cv": ["opencv-python-headless>=4.8.0", "imagehash>=4.3.0",
               "scikit-learn>=1.3.0"],
        "clip": ["torch>=2.0.0", "torchvision>=0.15.0",
                 "clip @ git+https://github.com/openai/CLIP.git"],
        "all": ["PySide6>=6.6.0", "opencv-python-headless>=4.8.0",
                "imagehash>=4.3.0", "scikit-learn>=1.3.0"],
    },
    entry_points={
        "console_scripts": [
            "wanalyzer=wallpaper_analyzer.cli:main",
            "wanalyzer-gui=wallpaper_analyzer.gui:main",
        ],
    },
)