import multiprocessing
import sys


def desktop_main() -> int:
    multiprocessing.freeze_support()
    from PySide6.QtWidgets import QApplication
    from eip.ui.fonts import configure_fonts
    from eip.ui.window import MainWindow
    app = QApplication(sys.argv)
    configure_fonts(app)
    app.setApplicationName("Expertise Image Processor")
    app.setOrganizationName("EIP")
    window = MainWindow()
    window.show()
    return app.exec()


def main() -> int:
    from eip.web.main import main as web_main
    return web_main()


if __name__ == "__main__":
    raise SystemExit(main())
