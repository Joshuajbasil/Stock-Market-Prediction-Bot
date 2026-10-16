"""Local dashboard and bounded background evaluation runner (no extra packages)."""
import csv
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import threading
import webbrowser
from datetime import date, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / 'outputs'
STATIC = Path(__file__).with_name('web')
TOKEN = secrets.token_urlsafe(32)
LOCK = threading.Lock()
JOB = {'status': 'idle', 'log': '', 'run': None}


def render_page():
    """Bundle UI assets and saved evidence so direct file previews also work."""
    page = (STATIC / 'index.html').read_text(encoding='utf-8')
    saved = {}
    for item in runs():
        try:
            saved[item['id']] = detail(item['id'])
        except (ValueError, KeyError, OSError):
            continue
    payload = json.dumps({'runs': [r for r in runs() if r['id'] in saved], 'details': saved}).replace('<', '\\u003c')
    page = page.replace('<link rel="stylesheet" href="./style.css">',
                        '<style>' + (STATIC / 'style.css').read_text(encoding='utf-8') + '</style>')
    page = page.replace('<script src="./app.js"></script>',
                        '<script id="saved-data" type="application/json">' + payload + '</script><script>' +
                        (STATIC / 'app.js').read_text(encoding='utf-8') + '</script>')
    return page


def run_path(name):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        raise ValueError('Invalid run name')
    path = (OUTPUTS / name).resolve()
    if path.parent != OUTPUTS.resolve():
        raise ValueError('Invalid run path')
    return path


def runs():
    result = []
    for file in OUTPUTS.glob('*/report.json'):
        try:
            report = json.loads(file.read_text())
            result.append({'id': file.parent.name, 'synthetic': report['synthetic'],
                           'backend': report['config']['backend'], 'modified': file.stat().st_mtime})
        except (ValueError, KeyError, OSError):
            continue
    return sorted(result, key=lambda item: item['modified'], reverse=True)


def detail(name):
    directory = run_path(name)
    report = json.loads((directory / 'report.json').read_text())
    tickers = {}
    for ticker in report['tickers']:
        if not re.fullmatch(r'[A-Za-z0-9^][A-Za-z0-9.^_-]*', ticker):
            raise ValueError('Invalid ticker in report')
        folder = directory / ticker
        with (folder / 'ohlcv.csv').open() as f:
            prices = [{'date': row['Date'][:10], 'close': float(row['Close'])} for row in csv.DictReader(f)]
        with (folder / 'backtest.csv').open() as f:
            predictions = list(csv.DictReader(f))
        equity = 1.0
        curve = []
        for row in predictions:
            equity *= 1 + float(row['strategy_return'])
            curve.append({'date': row['target_date'][:10], 'value': equity})
        tickers[ticker] = {'forecast': json.loads((folder / 'forecast.json').read_text()),
                           'prices': prices, 'equity': curve, 'recent': predictions[-12:][::-1]}
    return {'report': report, 'tickers': tickers}


def command(config, name):
    source, backend = config.get('source'), config.get('backend')
    if source not in ('demo', 'yahoo') or backend not in ('ridge', 'keras'):
        raise ValueError('Select a valid source and model')
    tickers = list(dict.fromkeys(re.split(r'[\s,]+', str(config.get('tickers', '')).strip().upper())))
    if not 1 <= len(tickers) <= 8 or any(not re.fullmatch(r'[A-Z0-9^][A-Z0-9.^_-]{0,14}', t) for t in tickers):
        raise ValueError('Enter 1–8 valid ticker symbols')
    start, end = date.fromisoformat(config['start']), date.fromisoformat(config['end'])
    if start >= end:
        raise ValueError('End date must follow start date')
    epochs = int(config.get('epochs', 30))
    if not 1 <= epochs <= 100:
        raise ValueError('Epochs must be between 1 and 100')
    return [sys.executable, '-u', '-m', 'stockbot', '--source', source, '--backend', backend,
            '--tickers', *tickers, '--start', str(start), '--end', str(end), '--epochs', str(epochs),
            '--output', str(run_path(name)), '--dashboard-output']


def execute(args):
    try:
        process = subprocess.Popen(args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding='utf-8', errors='replace',
                                   env={**os.environ, 'PYTHONIOENCODING': 'utf-8'},
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        for line in process.stdout:
            with LOCK:
                JOB['log'] = (JOB['log'] + line)[-20000:]
        process.stdout.close()
        code = process.wait()
        with LOCK:
            JOB['status'] = 'complete' if code == 0 else 'failed'
    except Exception as exc:
        with LOCK:
            JOB.update(status='failed', log=str(exc))


class Handler(BaseHTTPRequestHandler):
    def respond(self, body, content_type='application/json', status=200):
        if not isinstance(body, bytes):
            body = json.dumps(body).encode()
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path in ('/', '/index.html'):
                return self.respond(render_page().encode('utf-8'), 'text/html; charset=utf-8')
            if parsed.path == '/api/runs':
                return self.respond({'runs': runs(), 'token': TOKEN})
            if parsed.path == '/api/run':
                return self.respond(detail(parse_qs(parsed.query).get('id', [''])[0]))
            if parsed.path == '/api/job':
                with LOCK:
                    return self.respond(JOB.copy())
            files = {'/': ('index.html', 'text/html; charset=utf-8'),
                     '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                     '/style.css': ('style.css', 'text/css; charset=utf-8')}
            if parsed.path in files:
                filename, mime = files[parsed.path]
                return self.respond((STATIC / filename).read_bytes(), mime)
            self.respond({'error': 'Not found'}, status=404)
        except (ValueError, KeyError, OSError) as exc:
            self.respond({'error': str(exc)}, status=400)

    def do_POST(self):
        if self.path != '/api/run' or self.headers.get('X-Dashboard-Token') != TOKEN:
            return self.respond({'error': 'Unauthorized request'}, status=403)
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 8192:
                raise ValueError('Invalid request size')
            config = json.loads(self.rfile.read(length))
            name = datetime.now().strftime('dashboard-%Y%m%d-%H%M%S-') + secrets.token_hex(3)
            args = command(config, name)
            with LOCK:
                if JOB['status'] == 'running':
                    return self.respond({'error': 'An evaluation is already running'}, status=409)
                JOB.update(status='running', log='Starting evaluation…\n', run=name)
            threading.Thread(target=execute, args=(args,), daemon=True).start()
            self.respond({'run': name}, status=202)
        except (ValueError, TypeError, KeyError) as exc:
            self.respond({'error': str(exc)}, status=400)


def main():
    (ROOT / 'Dashboard.html').write_text(render_page(), encoding='utf-8')
    if '--export' in sys.argv:
        print('Saved Dashboard.html: open directly to explore saved results.')
        return
    server = ThreadingHTTPServer(('127.0.0.1', 8765), Handler)
    url = 'http://127.0.0.1:8765'
    print(f'Stockbot dashboard: {url}', flush=True)
    if '--no-browser' not in sys.argv:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
