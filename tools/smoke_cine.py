"""Verify a real Packed10 CINE through an isolated browser and server.

No original data is copied or changed. Reference decoding uses unpackbits,
independent of the application's four-sample decoder.
"""
from pathlib import Path
import argparse
import csv
import io
import json
import os
import struct
import subprocess
import sys
import time

import httpx
import numpy as np
import tifffile
from playwright.sync_api import sync_playwright, expect


def reference(path):
    with path.open('rb') as file:
        header = struct.unpack('<HHHHiIiIIIIQ', file.read(44))
        bitmap = struct.unpack('<IiiHHIIiiII', file.read(40))
        width, height, count = bitmap[1], abs(bitmap[2]), header[7]
        assert header[3] == 1 and bitmap[4:6] == (16, 256), 'This smoke check requires v1 Packed10'
        file.seek(header[9] + 142)
        setup_length = struct.unpack('<H', file.read(2))[0]
        cursor, ticks = header[9] + setup_length, None
        while cursor < header[10]:
            file.seek(cursor)
            size, kind, _ = struct.unpack('<IHH', file.read(8))
            if kind == 1002:
                ticks = np.frombuffer(file.read(size - 8), dtype='<u8').copy()
                ticks &= np.uint64(0xFFFFFFFFFFFFFFFC)
            cursor += size
        assert ticks is not None and len(ticks) == count
        file.seek(header[10])
        offsets = np.frombuffer(file.read(count * 8), dtype='<u8').copy()
    times = (ticks - ticks[0]).astype(np.float64) / 2**32

    def decode(index):
        with path.open('rb') as file:
            file.seek(int(offsets[index]))
            annotation = struct.unpack('<I', file.read(4))[0]
            file.seek(int(offsets[index]) + annotation - 4)
            size = struct.unpack('<I', file.read(4))[0]
            payload = np.frombuffer(file.read(size), dtype=np.uint8)
        bits = np.unpackbits(payload, bitorder='big').reshape(-1, 10)
        return (bits @ (2**np.arange(9, -1, -1, dtype=np.uint16))).reshape(height, width)

    return header, (width, height), ticks, times, decode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8883)
    args = parser.parse_args()
    path = args.source.resolve()
    before = (path.stat().st_size, path.stat().st_mtime_ns)
    header, shape, ticks, times, decode = reference(path)
    root = Path(__file__).resolve().parents[1]
    output = root / '.test-output' / f'cine-{int(time.time())}'
    output.mkdir(parents=True)
    url = f'http://127.0.0.1:{args.port}'
    environment = dict(os.environ, PYTHONIOENCODING='utf-8')
    with (output / 'server.log').open('w', encoding='utf-8') as log:
        server = subprocess.Popen([sys.executable, str(root / 'run_web.py'), '--port', str(args.port), '--no-browser', '--workspace', str(output)], cwd=root, stdout=log, stderr=log, env=environment, creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
        try:
            for _ in range(60):
                try:
                    if httpx.get(url + '/api/state', timeout=.5).status_code == 200:
                        break
                except httpx.HTTPError:
                    time.sleep(.1)
            with httpx.Client(base_url=url, timeout=30) as api, sync_playwright() as playwright:
                api.headers['X-EIP-Token'] = api.get('/api/state').json()['session_token']
                browser = playwright.chromium.launch(channel='msedge' if sys.platform == 'win32' else 'chromium', headless=True)
                page = browser.new_page(viewport={'width': 1440, 'height': 960})
                errors = []
                frames = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.on('response', lambda response: frames.append(response.json()) if response.url.endswith('/frame') and response.status == 200 else None)
                page.goto(url)
                expect(page.locator('#connection')).to_have_text('本机已连接')
                page.locator('#import').click()
                page.locator('#browse-path').fill(str(path.parent)); page.locator('#browse-go').click()
                page.locator('#file-list .file-row').filter(has_text=path.name).click()
                page.locator('#dialog-submit').click()
                expect(page.locator('#preview-title')).to_have_text(path.name, timeout=20000)
                expect(page.locator('#dtype-badge')).to_have_text('uint16')
                expect(page.locator('#shape-badge')).to_have_text(f'{shape[0]} × {shape[1]}')
                expect(page.locator('#seek-mode')).to_have_value('frame')
                expect(page.locator('#frame-info')).to_contain_text(f'第 1 帧 / {len(times)}')
                expect(page.locator('#timeline-note')).to_contain_text(f'采集帧 {header[6]}')
                assert '帧号仅在可确定时显示' not in page.locator('#timeline-note').inner_text()
                source = api.get('/api/state').json()['sources'][0]

                def locate(index):
                    page.locator('#seek-mode').select_option('frame')
                    page.locator('#seek-value').fill(str(index + 1)); page.locator('#seek').click()
                    expect(page.locator('#frame-info')).to_contain_text(f'第 {index + 1} 帧 /')
                    expect(page.locator('#timeline-note')).to_contain_text(f'采集帧 {header[6] + index}')
                    assert frames[-1]['ref']['index'] == index
                    assert frames[-1]['ref']['time'] == times[index]
                    assert frames[-1]['ref']['pts'] == int(ticks[index]) - int(ticks[0])
                    raw = decode(index)
                    frame = frames[-1]
                    for x, y in [(0, 0), (17, 31), (shape[0] - 1, shape[1] - 1)]:
                        pixel = api.get(f'/api/frames/{frame["token"]}/pixel', params={'x': x, 'y': y}).json()
                        assert pixel['value'] == int(raw[y, x])
                    return frame, raw

                locate(0)
                trigger_index = -header[6]
                locate(trigger_index)
                page.locator('#next').click()
                expect(page.locator('#frame-info')).to_contain_text(f'第 {trigger_index + 2} 帧 /')
                page.locator('#previous').click()
                expect(page.locator('#frame-info')).to_contain_text(f'第 {trigger_index + 1} 帧 /')
                page.locator('#seek-mode').select_option('time')
                page.locator('#seek-value').fill('1'); page.locator('#seek').click()
                expected_index = min(len(times) - 1, int(np.searchsorted(times, 1 - 1e-12)))
                expect(page.locator('#frame-info')).to_contain_text(f'第 {expected_index + 1} 帧 /')
                # Reusing the exact displayed time must keep the same actual frame.
                page.locator('#seek').click()
                expect(page.locator('#frame-info')).to_contain_text(f'第 {expected_index + 1} 帧 /')
                locate(len(times) - 2)
                page.locator('#play').click()
                expect(page.locator('#frame-info')).to_contain_text(f'第 {len(times)} 帧 /')
                expect(page.locator('#play')).to_have_attribute('aria-label', '播放')
                expect(page.locator('#next')).to_be_disabled()
                frame, raw = locate(trigger_index)
                page.locator('#roi-full').click()
                for key, value in [('width', 40), ('height', 30), ('x', 20), ('y', 10)]:
                    page.locator('#roi-' + key).fill(str(value)); page.locator('#roi-' + key).press('Tab')
                    expect(page.locator('#roi-' + key)).to_have_value(str(value))
                page.locator('#algorithm').select_option('intensity'); page.locator('#add-step').click()
                for key, value in [('gain', '2'), ('offset', '7')]:
                    page.locator(f'[data-key="{key}"]').fill(value); page.locator(f'[data-key="{key}"]').press('Tab')
                page.locator('#trial').click()
                expect(page.locator('#result-label')).to_have_text('当前帧精确试算', timeout=15000)
                expected = raw[10:40, 20:60] * 2 + 7
                trial = next(j for j in api.get('/api/jobs').json() if j['mode'] == 'trial')
                trial_result = api.get(f'/api/jobs/{trial["id"]}').json()['result']
                assert api.get(f'/api/frames/{trial_result["token"]}/pixel?x=20&y=10').json()['value'] == int(expected[0, 0])
                page.screenshot(path=str(output / 'cine-roi-1440.png'))
                page.locator('#run').click()
                expect(page.locator('#result-count')).to_have_text('1', timeout=15000)
                job = next(j for j in api.get('/api/jobs').json() if j['mode'] == 'run')
                image = api.get(f'/api/jobs/{job["id"]}/download?file=image').content
                np.testing.assert_array_equal(tifffile.imread(io.BytesIO(image)), expected)
                (output / 'verified-export.tif').write_bytes(image)
                table = api.get(f'/api/jobs/{job["id"]}/download?file=csv').content.decode('utf-8-sig')
                row = next(csv.DictReader(io.StringIO(table)))
                assert int(row['source_frame']) == trigger_index and int(row['source_frame_number']) == 0
                assert float(row['time_seconds']) == times[trigger_index]
                assert float(row['trigger_time_seconds']) == frame['ref']['trigger_time']
                page.locator('#result-list .source-card').click()
                expect(page.locator('#result-label')).to_contain_text('正式结果')
                page.locator('#seek-mode').select_option('time')
                page.reload()
                expect(page.locator('#seek-mode')).to_have_value('time')
                expect(page.locator('#frame-info')).to_contain_text(f'第 {trigger_index + 1} 帧 /')
                expect(page.locator('#roi-width')).to_have_value('40')
                for width, height in [(1440, 960), (1024, 768)]:
                    page.set_viewport_size({'width': width, 'height': height})
                    page.screenshot(path=str(output / f'cine-{width}.png'))
                    bounds = [page.locator(selector).bounding_box() for selector in ['#previous', '#timeline-note', '.seek-control', '#seek']]
                    assert all(b and b['x'] >= 0 and b['x'] + b['width'] <= width and b['y'] + b['height'] <= height for b in bounds), bounds
                    assert page.locator('#run').bounding_box()['y'] < height
                page.set_viewport_size({'width':1440,'height':960})
                page.locator('#analysis-mode').select_option('split_correlation')
                expect(page.locator('#split-empty')).to_be_hidden()
                page.locator('#split-flow-list [data-algorithm="correlation"] [data-edit]').click()
                page.locator('#split-window').fill('4');page.locator('#split-window').dispatch_event('change')
                expect(page.locator('#split-trial')).to_be_disabled()
                page.locator('#split-view').select_option('raw')
                page.locator('#split-open-context').click()
                index=min(len(times)-1,trigger_index+10)
                page.locator('#split-center').fill(str(index+1));page.locator('#split-center').dispatch_event('change')
                page.locator('#split-seek').click()
                expect(page.locator('#split-view-caption')).to_contain_text(f'原始帧 {index+1}')
                actual=frames[-1]
                assert actual['ref']['index']==index and actual['ref']['time']==times[index]
                expected_raw=decode(index)
                assert api.get(f'/api/frames/{actual["token"]}/pixel?x=17&y=31').json()['value']==int(expected_raw[31,17])
                page.wait_for_timeout(500)
                page.reload()
                expect(page.locator('#analysis-mode')).to_have_value('split_correlation')
                expect(page.locator('#split-view-caption')).to_contain_text(f'原始帧 {index+1}')
                expect(page.locator('#split-empty')).to_be_hidden()
                page.screenshot(path=str(output/'cine-split-invalid-window.png'))
                assert not errors, errors
                assert (path.stat().st_size, path.stat().st_mtime_ns) == before
                report = {'status': 'passed', 'source': str(path), 'file_bytes': before[0], 'frames': len(times), 'shape': list(shape), 'screenshots': str(output), 'browser_errors': errors, 'checks': ['independent Packed10 values', 'exact frame/trigger/time', 'next/previous and EOF', 'time round trip', 'ROI trial and TIFF export', 'CSV provenance', 'formal matching', 'refresh position', '1440/1024 layout', 'original unchanged']}
                (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
                print(json.dumps(report, ensure_ascii=False))
                browser.close()
        finally:
            if sys.platform == 'win32':
                subprocess.run(['taskkill', '/PID', str(server.pid), '/T', '/F'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                server.terminate()
            server.wait(timeout=5)


if __name__ == '__main__':
    main()
