from dataclasses import asdict
import traceback

from eip.domain import source_from_dict
from eip.readers import open_source


def preview_entry(commands, results):
    reader = None
    key = None
    last_ref = None
    while True:
        command = commands.get()
        if command["action"] == "stop":
            break
        try:
            source = source_from_dict(command["source"])
            if source.key != key:
                if reader:
                    reader.close()
                reader = None
                key = None
                reader = open_source(source)
                key = source.key
                last_ref = None
            action = command["action"]
            if action == "open":
                frame = reader.first() if reader.meta.kind == "video" else reader.read_frame(0)
            elif action == "time":
                frame = reader.read_at_time(command["time"])
            elif action == "next":
                if last_ref is None or last_ref.time != command.get("time"):
                    reader.read_at_time(command.get("time", 0))
                frame = reader.read_next()
            elif action == "previous":
                frame = reader.read_previous(command["time"])
            else:
                frame = reader.read_frame(command["index"])
            last_ref = frame.ref
            results.put({"id": command["id"], "type": "frame", "action": action, "source": command["source"], "meta": asdict(reader.meta), "pixels": frame.pixels, "frame": asdict(frame.ref)})
        except StopIteration:
            results.put({"id": command["id"], "type": "eof"})
        except Exception as error:
            results.put({"id": command["id"], "type": "failed", "message": str(error), "details": traceback.format_exc()})
    if reader:
        reader.close()
