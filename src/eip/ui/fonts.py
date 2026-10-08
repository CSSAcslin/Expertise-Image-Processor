import os
from pathlib import Path

from PySide6.QtGui import QFont, QFontDatabase


def configure_fonts(app):
    families = set(QFontDatabase.families())
    preferred = ["Microsoft YaHei UI", "Microsoft YaHei", "Noto Sans CJK SC", "PingFang SC", "WenQuanYi Micro Hei"]
    if os.name == "nt" and not any(name in families for name in preferred):
        path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/msyh.ttc"
        if path.is_file():
            identifier = QFontDatabase.addApplicationFont(str(path))
            families.update(QFontDatabase.applicationFontFamilies(identifier))
    family = next((name for name in preferred if name in families), "sans-serif")
    font = QFont(family, 9)
    app.setFont(font)
