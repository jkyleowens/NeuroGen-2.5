"""Subprocess lifecycle and thread-safe event queues; no Tk or torch imports."""
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading


def child_environment():
    environment = dict(os.environ, PYTHONUNBUFFERED='1')
    # Also works from a source checkout when the selected workspace is elsewhere.
    source = str(Path(__file__).resolve().parents[2])
    environment['PYTHONPATH'] = source + os.pathsep + environment.get('PYTHONPATH', '')
    return environment


class ModelProcess:
    def __init__(self, workspace):
        self.events = queue.Queue(maxsize=4096)
        self.sequence = 0
        self.process = subprocess.Popen([sys.executable, '-m', 'neurogen.workbench.worker'],
            cwd=workspace, env=child_environment(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding='utf-8', bufsize=1)
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._errors, daemon=True).start()

    def _read(self):
        for line in self.process.stdout:
            try:
                self.events.put(json.loads(line))
            except ValueError:
                self.events.put({'event': 'log', 'text': line.rstrip()})
        code = self.process.wait()
        self.events.put({'event': 'exited', 'returncode': code})

    def _errors(self):
        for line in self.process.stderr:
            self.events.put({'event': 'log', 'text': line.rstrip()})

    def send(self, action, **values):
        if self.process.poll() is not None:
            raise ValueError('Model worker is not running. Use Restart worker in Runs & diagnostics.')
        self.sequence += 1
        self.process.stdin.write(json.dumps({'id': self.sequence, 'action': action, **values}, allow_nan=False) + '\n')
        self.process.stdin.flush()
        return self.sequence

    def stop(self):
        if self.process.poll() is None:
            try:
                self.send('shutdown')
            except (OSError, ValueError):
                pass

    def force_stop(self):
        if self.process.poll() is None:
            self.process.terminate()


class JobProcess:
    def __init__(self):
        self.process = None
        self.events = queue.Queue(maxsize=4096)
        self.output = None
        self.kind = None
        self.log_path = None

    @property
    def running(self):
        return self.process is not None and self.process.poll() is None

    def start(self, kind, arguments, workspace, log_path, output=None):
        if self.running:
            raise ValueError('Another experiment is still running')
        self.kind, self.output, self.log_path = kind, output, Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.process = subprocess.Popen([sys.executable, '-m', 'neurogen', *arguments],
            cwd=workspace, env=child_environment(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace', bufsize=1,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0)
        process = self.process
        threading.Thread(target=self._read, args=(process, self.log_path, kind, output), daemon=True).start()

    def _read(self, process, log_path, kind, output):
        with log_path.open('w', encoding='utf-8') as log:
            for line in process.stdout:
                log.write(line)
                log.flush()
                self.events.put({'event': 'job_log', 'kind': kind, 'text': line.rstrip()})
                try:
                    row = json.loads(line)
                    if isinstance(row, dict) and 'step' in row:
                        self.events.put({'event': 'metrics', 'row': row})
                    elif isinstance(row, dict) and row.get('event') == 'corpus_progress':
                        self.events.put({'event': 'corpus_progress', 'row': row})
                except ValueError:
                    pass
        self.events.put({'event': 'job_done', 'kind': kind, 'returncode': process.wait(), 'output': output})

    def stop(self, force=False):
        if not self.running:
            return
        if force:
            self.process.terminate()
        elif os.name == 'nt':
            self.process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            self.process.send_signal(signal.SIGINT)
