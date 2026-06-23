"""Entry point for: python -m wallpaper_analyzer.gui

Can also be invoked via: wanalizer-gui
"""
import sys

try:
    from . import main
except ImportError:
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from wallpaper_analyzer.gui import main

sys.exit(main())
