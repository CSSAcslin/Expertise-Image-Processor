from pathlib import Path
import argparse
import multiprocessing
import threading
import webbrowser


def main():
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description="本机图像与视频工作台")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    from eip.web.app import create_app
    import uvicorn
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, args=(f"http://127.0.0.1:{args.port}",)).start()
    uvicorn.run(create_app(Path.cwd()), host="127.0.0.1", port=args.port, log_level="warning")
    return 0
