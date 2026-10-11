"""Native Tk desktop application for model experiments. No tensor work on the UI thread."""
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import queue
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import webbrowser

from .configuration import estimate, inference_settings, parse_config, save_json
from .processes import JobProcess, ModelProcess
from neurogen.language.config import LanguageConfig
from neurogen.language.corpus import DEFAULT_DATASET

BG, PANEL, FIELD, TEXT, MUTED, ACCENT = '#101721', '#192432', '#0d131c', '#e3edf7', '#9eb0c4', '#54d6bf'


def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


class Workbench:
    def __init__(self, root, workspace, checkpoint=None):
        self.root, self.workspace = root, Path(workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.home = self.workspace / 'runs/workbench'
        self.home.mkdir(parents=True, exist_ok=True)
        self.worker = ModelProcess(self.workspace)
        self.job = JobProcess()
        self.model_info, self.busy_id, self.pending_job = None, None, None
        self.job_stage = None
        self.conversation, self.current_reply = [], None
        self.closing, self.worker_ready, self.image = False, False, None
        self.initial_checkpoint = checkpoint
        self.recording, self.recorded, self.recorded_frames = None, [], 0
        self.record_overflow = False
        self.last_image_stamp, self.last_run_refresh = None, 0.
        self.vars, self.train_vars = {}, {}
        self._theme()
        root.title('NeuroGen · Model Workbench')
        root.geometry('1280x880')
        root.minsize(980, 720)
        root.protocol('WM_DELETE_WINDOW', self.close)
        self._build()
        self._restore_preferences()
        self.refresh_checkpoints()
        self.root.after(40, self.poll)

    def _theme(self):
        self.root.configure(background=BG)
        self.root.option_add('*Font', ('DejaVu Sans', 10))
        style = ttk.Style(self.root)
        style.theme_use('clam')
        style.configure('.', background=BG, foreground=TEXT, font=('DejaVu Sans', 10))
        style.configure('TFrame', background=BG)
        style.configure('Card.TFrame', background=PANEL)
        style.configure('TLabel', background=BG, foreground=TEXT)
        style.configure('Muted.TLabel', foreground=MUTED)
        style.configure('Title.TLabel', font=('DejaVu Sans', 22, 'bold'))
        style.configure('Sub.TLabel', font=('DejaVu Sans', 12, 'bold'))
        style.configure('TButton', background=PANEL, foreground=TEXT, padding=(12, 7), borderwidth=0)
        style.map('TButton', background=[('active', '#2d4358'), ('disabled', '#17202c')], foreground=[('disabled', '#607187')])
        style.configure('Accent.TButton', background=ACCENT, foreground=BG)
        style.map('Accent.TButton', background=[('active', '#85eedb')])
        style.configure('TEntry', fieldbackground=FIELD, foreground=TEXT, padding=5, insertcolor=TEXT)
        style.configure('TCombobox', fieldbackground=FIELD, background=PANEL, foreground=TEXT, arrowcolor=ACCENT, padding=5)
        style.map('TCombobox', fieldbackground=[('readonly', FIELD)], foreground=[('readonly', TEXT)])
        style.configure('TCheckbutton', background=BG, foreground=TEXT)
        style.map('TCheckbutton', background=[('active', BG)])
        style.configure('TNotebook', background=BG, borderwidth=0)
        style.configure('TNotebook.Tab', background=PANEL, foreground=MUTED, padding=(13, 11))
        style.map('TNotebook.Tab', background=[('selected', '#25374a')], foreground=[('selected', ACCENT)])
        style.configure('TLabelframe', background=BG, bordercolor='#2c3e52')
        style.configure('TLabelframe.Label', background=BG, foreground=ACCENT)
        style.configure('Horizontal.TProgressbar', background=ACCENT, troughcolor=FIELD)
        self.root.option_add('*TCombobox*Listbox.background', FIELD)
        self.root.option_add('*TCombobox*Listbox.foreground', TEXT)

    def variable(self, name, default='', boolean=False):
        value = tk.BooleanVar(value=default) if boolean else tk.StringVar(value=str(default))
        self.vars[name] = value
        return value

    def label(self, parent, text, style='TLabel', **kwargs):
        widget = ttk.Label(parent, text=text, style=style, **kwargs)
        return widget

    def button(self, parent, text, command, accent=False):
        return ttk.Button(parent, text=text, command=lambda: self.guard(command), style='Accent.TButton' if accent else 'TButton')

    def text_box(self, parent, height=8, readonly=False):
        frame = ttk.Frame(parent)
        text = tk.Text(frame, height=height, background=FIELD, foreground=TEXT, insertbackground=TEXT,
                       selectbackground='#31546a', relief='flat', padx=12, pady=10, wrap='word',
                       font=('DejaVu Sans Mono', 10), undo=not readonly)
        scroll = ttk.Scrollbar(frame, command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        text.pack(side='left', fill='both', expand=True)
        scroll.pack(side='right', fill='y')
        if readonly:
            text.configure(state='disabled')
        return frame, text

    @staticmethod
    def replace_text(widget, value, readonly=False):
        widget.configure(state='normal')
        widget.delete('1.0', 'end')
        widget.insert('1.0', value)
        if readonly:
            widget.configure(state='disabled')

    def browse(self, variable, save=False, directory=False, types=None):
        options = {'parent': self.root, 'initialdir': str(self.workspace)}
        if types:
            options['filetypes'] = types
        path = (filedialog.askdirectory(**options) if directory else
                filedialog.asksaveasfilename(**options) if save else filedialog.askopenfilename(**options))
        if path:
            variable.set(path)
        return path

    def path_row(self, parent, row, title, variable, save=False, directory=False, types=None):
        self.label(parent, title).grid(row=row, column=0, sticky='w', padx=(0, 12), pady=5)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky='ew', pady=5)
        self.button(parent, 'Browse', lambda: self.browse(variable, save, directory, types)).grid(row=row, column=2, padx=(8, 0))
        parent.columnconfigure(1, weight=1)

    def _build(self):
        header = ttk.Frame(self.root, padding=(22, 18, 22, 12))
        header.pack(fill='x')
        self.label(header, 'NEUROGEN', 'Title.TLabel').pack(side='left')
        self.label(header, '  /  Model workbench', 'Muted.TLabel').pack(side='left', padx=10)
        self.status = self.variable('status', 'Starting model worker…')
        ttk.Label(header, textvariable=self.status, style='Muted.TLabel').pack(side='right')
        toolbar = ttk.Frame(self.root, padding=(22, 0, 22, 10))
        toolbar.pack(fill='x')
        self.checkpoint = self.variable('checkpoint')
        self.checkpoint_picker = ttk.Combobox(toolbar, textvariable=self.checkpoint, width=58)
        self.checkpoint_picker.pack(side='left', fill='x', expand=True, padx=(0, 8))
        self.button(toolbar, 'Browse checkpoint', lambda: self.browse(self.checkpoint, types=[('Language checkpoints', '*.pt *.pth'), ('All files', '*')])).pack(side='left', padx=4)
        self.device = self.variable('device', 'auto')
        device_picker = ttk.Combobox(toolbar, textvariable=self.device, values=('auto', 'cpu', 'cuda', 'rocm'), state='readonly', width=7)
        device_picker.pack(side='left', padx=8)
        device_picker.bind('<<ComboboxSelected>>', lambda event: self.status.set(
            f'Selected {self.device.get()}; reload the model to apply. New jobs use this device.'))
        self.button(toolbar, 'Load model', self.load_model, accent=True).pack(side='left', padx=4)
        self.button(toolbar, 'Unload', lambda: self.request('unload')).pack(side='left', padx=(4, 0))
        self.summary = self.variable('summary', 'Load a language checkpoint, or create a model in Model & configuration.')
        ttk.Label(self.root, textvariable=self.summary, style='Muted.TLabel', padding=(22, 0, 22, 13)).pack(fill='x')
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill='both', expand=True, padx=18)
        self.pages = {}
        for title in ('Chat', 'Model & configuration', 'Train', 'Speech', 'Biological simulation', 'Runs & diagnostics'):
            page = ttk.Frame(self.tabs, padding=16)
            self.tabs.add(page, text=title)
            self.pages[title] = page
        self._chat()
        self._configuration()
        self._training()
        self._speech()
        self._biology()
        self._runs()
        self.footer = self.variable('footer', 'Local execution · Language activity is measured per internal tick, not biological Hz.')
        ttk.Label(self.root, textvariable=self.footer, style='Muted.TLabel', padding=(22, 9)).pack(fill='x')

    def _chat(self):
        page = self.pages['Chat']
        page.columnconfigure(0, weight=1)
        page.rowconfigure(0, weight=1)
        left = ttk.Frame(page)
        left.grid(row=0, column=0, sticky='nsew', padx=(0, 18))
        frame, self.transcript = self.text_box(left, height=14, readonly=True)
        frame.pack(fill='both', expand=True)
        self.transcript.tag_configure('user', foreground=ACCENT, font=('DejaVu Sans', 10, 'bold'))
        self.transcript.tag_configure('model', foreground='#aecaff', font=('DejaVu Sans', 10, 'bold'))
        self.transcript.tag_configure('note', foreground=MUTED)
        self.chat_note('Your conversation stays local. Use Text continuation to test a model trained on plain text.')
        self.label(left, 'Prompt   ·   Ctrl+Enter sends', 'Muted.TLabel').pack(anchor='w', pady=(12, 5))
        frame, self.prompt = self.text_box(left, height=4)
        frame.pack(fill='x')
        self.prompt.bind('<Control-Return>', lambda event: self._send_shortcut())
        row = ttk.Frame(left)
        row.pack(fill='x', pady=(8, 0))
        self.mode = self.variable('mode', 'Dialogue')
        ttk.Combobox(row, textvariable=self.mode, values=('Dialogue', 'Text continuation'), state='readonly', width=19).pack(side='left')
        self.button(row, 'Send', self.send_prompt, True).pack(side='left', padx=8)
        self.button(row, 'Stop reply', self.cancel).pack(side='left')
        self.button(row, 'Score text', self.score).pack(side='left', padx=8)
        self.button(row, 'Export chat', self.export_chat).pack(side='right')
        right = ttk.Frame(page, width=260)
        right.grid(row=0, column=1, sticky='ns')
        self.label(right, 'Generation', 'Sub.TLabel').pack(anchor='w', pady=(0, 10))
        for name, title, value in [('temperature', 'Temperature · 0 = greedy', .7), ('top_k', 'Top-k', 40),
                                   ('max_new_tokens', 'Maximum output bytes', 256), ('idle_hz', 'Idle updates / second', 10)]:
            self.label(right, title, 'Muted.TLabel').pack(anchor='w', pady=(6, 3))
            ttk.Entry(right, textvariable=self.variable(name, value), width=24).pack(fill='x')
        ttk.Checkbutton(right, text='Continuous idle activity', variable=self.variable('continuous', False, True)).pack(anchor='w', pady=12)
        self.button(right, 'Apply settings', self.apply_settings).pack(fill='x')
        self.button(right, 'Advance 10 idle updates', lambda: self.request('idle', steps=10)).pack(fill='x', pady=6)
        self.button(right, 'New session', lambda: self.request('reset')).pack(fill='x', pady=(12, 6))
        self.button(right, 'Save session', self.save_session).pack(fill='x', pady=3)
        self.button(right, 'Restore session', self.restore_session).pack(fill='x', pady=3)
        self.chat_metrics = self.variable('chat_metrics', 'No reply measured yet.')
        ttk.Label(right, textvariable=self.chat_metrics, wraplength=245, style='Muted.TLabel').pack(anchor='w', pady=(18, 0))
        self.activity_label = self.variable('activity', 'E/I activity appears after a reply or idle update.')
        ttk.Label(right, textvariable=self.activity_label, wraplength=245, style='Muted.TLabel').pack(anchor='w', pady=12)

    def _configuration(self):
        page = self.pages['Model & configuration']
        self.label(page, 'Architecture for a new model', 'Sub.TLabel').pack(anchor='w')
        self.label(page, 'Loaded weights keep their original size. Editing this configuration affects new models and fresh training runs.',
                   'Muted.TLabel', wraplength=1000).pack(anchor='w', pady=(5, 12))
        row = ttk.Frame(page)
        row.pack(fill='x')
        self.preset = self.variable('preset', 'bc250')
        ttk.Combobox(row, textvariable=self.preset, values=('tiny', 'laptop', 'bc250', 'large'), state='readonly', width=12).pack(side='left')
        self.button(row, 'Use preset', self.use_preset).pack(side='left', padx=8)
        self.button(row, 'Copy loaded architecture', self.copy_architecture).pack(side='left')
        self.button(row, 'Open JSON', lambda: self.open_config(False)).pack(side='left', padx=8)
        self.button(row, 'Save JSON', lambda: self.save_config(False)).pack(side='left')
        frame, self.config_editor = self.text_box(page, height=12)
        frame.pack(fill='both', expand=True, pady=12)
        self.replace_text(self.config_editor, json.dumps(asdict(LanguageConfig()), indent=2))
        row = ttk.Frame(page)
        row.pack(fill='x')
        self.button(row, 'Validate & estimate size', self.validate_config).pack(side='left')
        self.label(row, 'Initialization seed', 'Muted.TLabel').pack(side='left', padx=(18, 8))
        ttk.Entry(row, textvariable=self.variable('model_seed', 1), width=10).pack(side='left')
        self.button(row, 'Create fresh model', self.create_model, True).pack(side='right')
        self.button(row, 'Save model weights…', self.save_model).pack(side='right', padx=8)
        self.estimate_label = self.variable('estimate')
        ttk.Label(page, textvariable=self.estimate_label, style='Muted.TLabel', wraplength=1000).pack(anchor='w', pady=12)
        self.validate_config()

    def _training(self):
        page = self.pages['Train']
        page.columnconfigure(0, weight=1)
        source_row = ttk.Frame(page)
        source_row.pack(fill='x', pady=(0, 8))
        self.label(source_row, 'Text data').pack(side='left')
        self.data_source = self.variable('data_source', 'SlimPajama (automatic)')
        picker = ttk.Combobox(source_row, textvariable=self.data_source,
                             values=('SlimPajama (automatic)', 'Local files'), state='readonly', width=26)
        picker.pack(side='left', padx=12)
        picker.bind('<<ComboboxSelected>>', lambda event: self.update_data_source())
        self.label(source_row, 'Fresh runs create a new subset; optimizer resume reuses saved data.', 'Muted.TLabel').pack(side='left')
        self.dataset_controls = ttk.Frame(page)
        self.dataset_controls.pack(fill='x')
        repo_row = ttk.Frame(self.dataset_controls)
        repo_row.pack(fill='x', pady=3)
        self.label(repo_row, 'Dataset repository', 'Muted.TLabel').pack(side='left')
        ttk.Entry(repo_row, textvariable=self.variable('dataset_id', DEFAULT_DATASET), width=52).pack(side='left', padx=8, fill='x', expand=True)
        self.label(repo_row, 'Revision', 'Muted.TLabel').pack(side='left')
        ttk.Entry(repo_row, textvariable=self.variable('dataset_revision', 'main'), width=12).pack(side='left', padx=8)
        limits = ttk.Frame(self.dataset_controls)
        limits.pack(fill='x', pady=3)
        for name, title, default in (('train_documents', 'Train documents', 2000),
                ('validation_documents', 'Validation documents', 128), ('max_document_bytes', 'Max bytes/document', 8192),
                ('dataset_seed', 'Data seed (blank = new)', '')):
            cell = ttk.Frame(limits)
            cell.pack(side='left', fill='x', expand=True, padx=(0, 12))
            self.label(cell, title, 'Muted.TLabel').pack(anchor='w')
            ttk.Entry(cell, textvariable=self.variable(name, default), width=15).pack(fill='x')
        paths = ttk.Frame(page)
        self.training_paths, self.text_file_widgets = paths, []
        paths.pack(fill='x')
        for index, (name, title) in enumerate([('train', 'Training text / JSONL'), ('validation', 'Validation text / JSONL'),
                ('speech_train', 'Speech training manifest (optional)'), ('speech_validation', 'Speech validation manifest (optional)'),
                ('source_checkpoint', 'Source checkpoint (initialize / resume)'), ('train_output', 'Output directory (blank = automatic)')]):
            self.path_row(paths, index, title, self.variable(name), directory=name == 'train_output')
            if name in ('train', 'validation'):
                self.text_file_widgets.extend(paths.grid_slaves(row=index))
        mode_row = ttk.Frame(page)
        mode_row.pack(fill='x', pady=8)
        self.label(mode_row, 'Training mode').pack(side='left')
        self.train_mode = self.variable('train_mode', 'Fresh model')
        ttk.Combobox(mode_row, textvariable=self.train_mode, values=('Fresh model', 'Initialize from weights', 'Resume optimizer'),
                     state='readonly', width=25).pack(side='left', padx=12)
        self.button(mode_row, 'Use loaded checkpoint settings', self.use_training_settings).pack(side='left')
        fields = ttk.Frame(page)
        fields.pack(fill='x', pady=4)
        definitions = [('steps', 'Total steps', 1000), ('batch_size', 'Batch', 2), ('sequence_length', 'Sequence bytes', 96),
                       ('accumulate', 'Accumulate', 4), ('learning_rate', 'Learning rate', .001), ('seed', 'Seed', 1),
                       ('eval_every', 'Evaluate every', 25), ('eval_batches', 'Eval batches', 4), ('threads', 'CPU threads', 1),
                       ('memory_budget_mib', 'GPU budget MiB', 6144), ('max_audio_seconds', 'Max audio seconds', 6),
                       ('ctc_weight', 'CTC weight', .1), ('mel_weight', 'Mel weight', 1.), ('stop_weight', 'Stop weight', .1)]
        for index, (name, title, default) in enumerate(definitions):
            row, col = divmod(index, 4)
            cell = ttk.Frame(fields)
            cell.grid(row=row, column=col, sticky='ew', padx=(0, 18), pady=3)
            fields.columnconfigure(col, weight=1)
            self.label(cell, title, 'Muted.TLabel').pack(anchor='w')
            variable = tk.StringVar(value=str(default))
            self.train_vars[name] = variable
            ttk.Entry(cell, textvariable=variable, width=14).pack(fill='x')
        actions = ttk.Frame(page)
        actions.pack(fill='x', pady=(12, 8))
        self.button(actions, 'Start training', self.start_training, True).pack(side='left')
        self.button(actions, 'Stop & checkpoint', lambda: self.stop_job()).pack(side='left', padx=8)
        self.button(actions, 'Load resulting checkpoint', self.load_result).pack(side='left')
        self.button(actions, 'View logs & charts', lambda: self.tabs.select(self.pages['Runs & diagnostics'])).pack(side='right')
        self.train_progress = ttk.Progressbar(page, mode='determinate')
        self.train_progress.pack(fill='x')
        self.train_status = self.variable('train_status', 'Fresh training uses the architecture editor. Starting a job unloads the chat model to free memory.')
        ttk.Label(page, textvariable=self.train_status, style='Muted.TLabel', wraplength=1050).pack(anchor='w', pady=8)
        self.update_data_source()

    def update_data_source(self):
        automatic = self.data_source.get() == 'SlimPajama (automatic)'
        for widget in self.text_file_widgets:
            widget.grid_remove() if automatic else widget.grid()
        if automatic:
            self.dataset_controls.pack(fill='x', before=self.training_paths)
        else:
            self.dataset_controls.pack_forget()

    def _speech(self):
        page = self.pages['Speech']
        self.label(page, 'Listening and speaking', 'Sub.TLabel').pack(anchor='w')
        self.label(page, 'Requires a checkpoint trained with paired speech. Input: mono PCM16 WAV, 16 kHz. Audio quality depends on training.',
                   'Muted.TLabel', wraplength=1000).pack(anchor='w', pady=(6, 16))
        paths = ttk.Frame(page)
        paths.pack(fill='x')
        self.wav_path = self.variable('wav_path')
        self.path_row(paths, 0, 'Input recording', self.wav_path, types=[('WAV audio', '*.wav')])
        row = ttk.Frame(page)
        row.pack(fill='x', pady=8)
        self.button(row, 'Transcribe WAV', self.listen).pack(side='left')
        self.record_button = self.button(row, 'Record microphone', self.toggle_recording)
        self.record_button.pack(side='left', padx=8)
        self.button(row, 'Send transcript to chat', self.transcript_to_chat).pack(side='left')
        self.button(row, 'Cancel audio operation', self.cancel).pack(side='right')
        frame, self.speech_transcript = self.text_box(page, height=5, readonly=True)
        frame.pack(fill='both', expand=True, pady=(3, 12))
        self.label(page, 'Text to synthesize', 'Muted.TLabel').pack(anchor='w')
        frame, self.speech_text = self.text_box(page, height=4)
        frame.pack(fill='both', expand=True, pady=8)
        row = ttk.Frame(page)
        row.pack(fill='x')
        self.label(row, 'Maximum acoustic frames', 'Muted.TLabel').pack(side='left')
        ttk.Entry(row, textvariable=self.variable('voice_frames', 300), width=9).pack(side='left', padx=8)
        self.button(row, 'Use last model reply', self.reply_to_speech).pack(side='left')
        self.button(row, 'Synthesize WAV…', self.speak, True).pack(side='right')
        self.button(row, 'Play last WAV', self.play_audio).pack(side='right', padx=8)
        self.audio_status = self.variable('audio_status', 'Microphone/playback require the optional voice dependencies. Recordings stop after 30 seconds.')
        ttk.Label(page, textvariable=self.audio_status, style='Muted.TLabel', wraplength=1000).pack(anchor='w', pady=12)
        self.last_audio = None

    def _biology(self):
        from neurogen.biology import BiologyConfig
        page = self.pages['Biological simulation']
        self.label(page, 'Conductance simulation · physical time', 'Sub.TLabel').pack(anchor='w')
        self.label(page, 'Biological .npz checkpoints and local plasticity are separate from language .pt checkpoints.', 'Muted.TLabel').pack(anchor='w', pady=6)
        paths = ttk.Frame(page)
        paths.pack(fill='x')
        self.bio_resume = self.variable('bio_resume')
        self.path_row(paths, 0, 'Resume .npz (optional)', self.bio_resume, types=[('Biological checkpoints', '*.npz')])
        frame, self.bio_editor = self.text_box(page, height=10)
        frame.pack(fill='both', expand=True, pady=10)
        self.replace_text(self.bio_editor, json.dumps(asdict(BiologyConfig()), indent=2))
        row = ttk.Frame(page)
        row.pack(fill='x')
        self.button(row, 'Open JSON', lambda: self.open_config(True)).pack(side='left')
        self.button(row, 'Save JSON', lambda: self.save_config(True)).pack(side='left', padx=6)
        self.button(row, 'Validate JSON', lambda: self.validate_config(True)).pack(side='left')
        settings = ttk.Frame(page)
        settings.pack(fill='x', pady=12)
        for name, title, default in [('duration_ms', 'Duration ms', 15000), ('report_ms', 'Report ms', 1000), ('plot_ms', 'Plot ms', 5000)]:
            self.label(settings, title, 'Muted.TLabel').pack(side='left', padx=(0, 7))
            ttk.Entry(settings, textvariable=self.variable(name, default), width=9).pack(side='left', padx=(0, 15))
        self.stimulus = self.variable('stimulus', 'cue')
        ttk.Combobox(settings, textvariable=self.stimulus, values=('cue', 'alternating', 'none'), state='readonly', width=12).pack(side='left')
        row = ttk.Frame(page)
        row.pack(fill='x')
        for name, title in [('frozen', 'Freeze plasticity'), ('realtime', 'Pace to wall time'), ('forever', 'Run until stopped')]:
            ttk.Checkbutton(row, text=title, variable=self.variable(name, False, True)).pack(side='left', padx=(0, 12))
        self.button(row, 'Start simulation', self.start_biology, True).pack(side='right')
        self.button(row, 'Stop & save', lambda: self.stop_job()).pack(side='right', padx=8)

    def _runs(self):
        page = self.pages['Runs & diagnostics']
        tools = ttk.Frame(page)
        tools.pack(fill='x')
        self.button(tools, 'Hardware diagnostic', self.doctor).pack(side='left')
        self.button(tools, 'Refresh checkpoints', self.refresh_checkpoints).pack(side='left', padx=6)
        self.button(tools, 'Restart model worker', self.restart_worker).pack(side='left')
        self.button(tools, 'Stop job', lambda: self.stop_job()).pack(side='right')
        self.button(tools, 'Force stop (no save)', lambda: self.stop_job(True)).pack(side='right', padx=6)
        path = ttk.Frame(page)
        path.pack(fill='x', pady=8)
        self.run_path = self.variable('run_path')
        self.path_row(path, 0, 'Run directory', self.run_path, directory=True)
        row = ttk.Frame(page)
        row.pack(fill='x')
        self.plot_choice = self.variable('plot_choice', 'learning.png')
        self.plot_picker = ttk.Combobox(row, textvariable=self.plot_choice, values=('learning.png',), state='readonly', width=28)
        self.plot_picker.pack(side='left')
        self.plot_picker.bind('<<ComboboxSelected>>', lambda event: self.guard(lambda: self.refresh_run(force=True)))
        self.button(row, 'Refresh run', lambda: self.refresh_run(force=True)).pack(side='left', padx=8)
        self.button(row, 'Open HTML report', self.open_report).pack(side='left')
        self.run_metrics = self.variable('run_metrics', 'Choose a run to inspect its plots and latest metrics.')
        ttk.Label(page, textvariable=self.run_metrics, style='Muted.TLabel', wraplength=1080).pack(anchor='w', pady=8)
        split = ttk.Panedwindow(page, orient='vertical')
        split.pack(fill='both', expand=True)
        self.plot_frame = ttk.Frame(split)
        self.plot_label = ttk.Label(self.plot_frame, text='Run charts will appear here.', anchor='center')
        self.plot_label.pack(fill='both', expand=True)
        split.add(self.plot_frame, weight=3)
        frame, self.logs = self.text_box(split, height=8, readonly=True)
        split.add(frame, weight=2)
        self.plot_frame.bind('<Configure>', lambda event: setattr(self, 'last_image_stamp', None))

    def guard(self, function):
        try:
            return function()
        except Exception as error:
            self.status.set('Action needs attention')
            self.footer.set(str(error))
            self.log(f'ERROR: {error}')
            messagebox.showerror('NeuroGen workbench', str(error), parent=self.root)

    def request(self, action, **values):
        if self.closing:
            raise ValueError('The workbench is closing')
        if self.job.running or self.pending_job:
            raise ValueError('Stop the current experiment before using the model')
        if self.busy_id is not None:
            raise ValueError('The model is busy. Stop its current operation or wait for it to finish.')
        if not self.worker_ready:
            raise ValueError('Wait for the model worker to finish starting')
        self.busy_id = self.worker.send(action, **values)
        self.status.set(f'{action.replace("_", " ").capitalize()}…')
        return self.busy_id

    def load_model(self):
        path = self.resolve(self.checkpoint.get())
        if not path.is_file():
            raise ValueError('Choose an existing language checkpoint')
        if path.suffix == '.npz':
            raise ValueError('Load biological .npz files in Biological simulation')
        self.request('load', path=str(path), device=self.device.get(), seed=int(self.vars['model_seed'].get()))

    def resolve(self, value):
        if not str(value).strip():
            raise ValueError('A file or directory path is required')
        path = Path(value).expanduser()
        return (path if path.is_absolute() else self.workspace / path).resolve()

    def get_settings(self):
        return inference_settings({name: self.vars[name].get() for name in
                                  ('temperature', 'top_k', 'max_new_tokens', 'idle_hz', 'continuous')})

    def apply_settings(self):
        self.request('settings', settings=self.get_settings())

    def _send_shortcut(self):
        self.guard(self.send_prompt)
        return 'break'

    def send_prompt(self):
        text = self.prompt.get('1.0', 'end-1c')
        if not text.strip():
            raise ValueError('Enter a prompt first')
        self.request('chat', text=text, mode='dialogue' if self.mode.get() == 'Dialogue' else 'continuation', settings=self.get_settings())
        self.conversation.append({'role': 'user', 'text': text, 'mode': self.mode.get()})
        self.conversation.append({'role': 'model', 'text': ''})
        self.conversation = self.conversation[-500:]
        self.current_reply = self.conversation[-1]
        self.chat_insert('\nYou  ·  ' + self.mode.get() + '\n', 'user')
        self.chat_insert(text + '\n\n')
        self.chat_insert('Model\n', 'model')
        self.prompt.delete('1.0', 'end')

    def chat_insert(self, text, tag=None):
        self.transcript.configure(state='normal')
        self.transcript.insert('end', text, tag or ())
        if int(self.transcript.index('end-1c').split('.')[0]) > 6000:
            self.transcript.delete('1.0', '1000.0')
        self.transcript.see('end')
        self.transcript.configure(state='disabled')

    def chat_note(self, text):
        self.chat_insert('\n' + text + '\n', 'note')

    def score(self):
        self.request('score', text=self.prompt.get('1.0', 'end-1c'))

    def cancel(self):
        if self.worker.process.poll() is None:
            self.worker.send('cancel')
            self.status.set('Stopping after the current model operation…')

    def export_chat(self):
        path = filedialog.asksaveasfilename(parent=self.root, initialdir=str(self.workspace),
                                            defaultextension='.json', filetypes=[('Chat JSON', '*.json')])
        if path:
            save_json(path, {'checkpoint': self.model_info.get('path') if self.model_info else None,
                             'conversation': self.conversation})
            self.footer.set(f'Chat exported to {path}')

    def save_session(self):
        if self.busy_id is not None:
            raise ValueError('Finish or stop the reply before saving a session')
        path = filedialog.asksaveasfilename(parent=self.root, initialdir=str(self.home),
                                            defaultextension='.pt', filetypes=[('Session', '*.pt')])
        if path:
            self.request('save_session', path=path, conversation=self.conversation)

    def restore_session(self):
        path = filedialog.askopenfilename(parent=self.root, initialdir=str(self.home), filetypes=[('Session', '*.pt')])
        if path:
            self.request('load_session', path=path)

    def use_preset(self):
        self.replace_text(self.config_editor, json.dumps(asdict(LanguageConfig.preset(self.preset.get())), indent=2))
        self.validate_config()

    def copy_architecture(self):
        if not self.model_info:
            raise ValueError('Load a model first')
        self.replace_text(self.config_editor, json.dumps(self.model_info['config'], indent=2))
        self.validate_config()

    def validate_config(self, biological=False):
        editor = self.bio_editor if biological else self.config_editor
        config = parse_config(editor.get('1.0', 'end-1c'), biological)
        self.replace_text(editor, json.dumps(config, indent=2))
        if biological:
            self.footer.set('Biological configuration is valid.')
        else:
            size = estimate(config)
            self.estimate_label.set(f"{size['parameters']:,} parameters · {size['neurons']:,} spiking units · "
                f"FP32 weights {size['weights_mib']:.1f} MiB · Adam parameter state {size['adam_mib']:.1f} MiB. "
                'Activations and runtime memory are additional.')
        return config

    def open_config(self, biological):
        path = filedialog.askopenfilename(parent=self.root, initialdir=str(self.workspace / 'configs'), filetypes=[('JSON', '*.json')])
        if path:
            text = Path(path).read_text(encoding='utf-8')
            config = parse_config(text, biological)
            self.replace_text(self.bio_editor if biological else self.config_editor, json.dumps(config, indent=2))
            self.validate_config(biological)

    def save_config(self, biological):
        config = self.validate_config(biological)
        path = filedialog.asksaveasfilename(parent=self.root, initialdir=str(self.workspace / 'configs'),
                                            defaultextension='.json', filetypes=[('JSON', '*.json')])
        if path:
            save_json(path, config)
            self.footer.set(f'Configuration saved to {path}')

    def create_model(self):
        self.request('create', config=self.validate_config(), device=self.device.get(), seed=int(self.vars['model_seed'].get()))

    def save_model(self):
        path = filedialog.asksaveasfilename(parent=self.root, initialdir=str(self.home),
                                            defaultextension='.pt', filetypes=[('Model checkpoint', '*.pt')])
        if path:
            self.request('save_model', path=path)

    def use_training_settings(self):
        if not self.model_info or not self.model_info.get('path'):
            raise ValueError('Load or save a model checkpoint first')
        self.vars['source_checkpoint'].set(self.model_info['path'])
        for key, value in self.model_info.get('settings', {}).items():
            if key in self.train_vars:
                self.train_vars[key].set(str(value))
        self.copy_architecture()
        self.train_mode.set('Resume optimizer' if self.model_info.get('resume_available') else 'Initialize from weights')
        source = self.model_info.get('data_source', {})
        if self.model_info.get('resume_available'):
            self.data_source.set('SlimPajama (automatic)' if source.get('kind') == 'slimpajama' else 'Local files')
            if source.get('kind') == 'files':
                for key in ('train', 'validation'):
                    self.vars[key].set(source[key])
            self.update_data_source()
        self.train_vars['steps'].set(str(max(int(self.train_vars['steps'].get()), self.model_info.get('step', 0) + 100)))
        self.footer.set('Checkpoint and training settings copied. SlimPajama resume reuses the saved corpus folder.')

    def _input_path(self, name, optional=False):
        value = self.vars[name].get().strip()
        if not value and optional:
            return None
        path = self.resolve(value)
        if not path.is_file():
            raise ValueError(f'{name}: choose an existing file')
        return str(path)

    def start_training(self):
        args = ['language', 'train', '--device', self.device.get()]
        if self.data_source.get() == 'SlimPajama (automatic)':
            args += ['--data-source', 'slimpajama']
            for key in ('dataset_id', 'dataset_revision'):
                value = self.vars[key].get().strip()
                if not value:
                    raise ValueError(f'{key} cannot be empty')
                args += ['--' + key.replace('_', '-'), value]
            for key in ('train_documents', 'validation_documents', 'max_document_bytes'):
                value = int(self.vars[key].get())
                if value < 1:
                    raise ValueError(f'{key} must be positive')
                args += ['--' + key.replace('_', '-'), str(value)]
            seed = self.vars['dataset_seed'].get().strip()
            if seed:
                if not 0 <= int(seed) < 2**32:
                    raise ValueError('Dataset seed must be between 0 and 4294967295')
                args += ['--dataset-seed', seed]
            file_keys = ('speech_train', 'speech_validation')
        else:
            args += ['--data-source', 'files']
            file_keys = ('train', 'validation', 'speech_train', 'speech_validation')
        for key in file_keys:
            value = self._input_path(key, optional=key.startswith('speech_'))
            if value:
                args += ['--' + key.replace('_', '-'), value]
        if bool(self.vars['speech_train'].get().strip()) != bool(self.vars['speech_validation'].get().strip()):
            raise ValueError('Select both speech manifests or neither')
        floats = {'learning_rate', 'memory_budget_mib', 'ctc_weight', 'mel_weight', 'stop_weight'}
        for key, variable in self.train_vars.items():
            value = float(variable.get()) if key in floats else int(variable.get())
            if not math.isfinite(value) or value < 0 or (key not in {'seed', 'ctc_weight', 'mel_weight', 'stop_weight'} and value == 0):
                raise ValueError(f'{key} must be finite and positive (seed and loss weights may be zero)')
            args += ['--' + key.replace('_', '-'), str(value)]
        identifier = stamp()
        mode = self.train_mode.get()
        if mode == 'Fresh model':
            config = self.validate_config()
            config_path = self.home / 'configs' / (identifier + '-language.json')
            save_json(config_path, config)
            args += ['--config', str(config_path)]
        else:
            source = self._input_path('source_checkpoint')
            args += ['--resume' if mode == 'Resume optimizer' else '--initialize-from', source]
        value = self.vars['train_output'].get().strip()
        output = self.resolve(value) if value else self.home / ('train-' + identifier)
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError('Training output must be a new or empty directory')
        args += ['--output', str(output)]
        self.train_progress.configure(maximum=int(self.train_vars['steps'].get()), value=0)
        self.queue_job('train', args, output)

    def start_biology(self):
        args = ['simulate', '--stimulus', self.stimulus.get()]
        for key in ('duration_ms', 'report_ms', 'plot_ms'):
            value = float(self.vars[key].get())
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be positive and finite')
            args += ['--' + key.replace('_', '-'), str(value)]
        identifier = stamp()
        resume = self._input_path('bio_resume', optional=True)
        if resume:
            args += ['--resume', resume]
        else:
            config = self.validate_config(True)
            path = self.home / 'configs' / (identifier + '-biology.json')
            save_json(path, config)
            args += ['--config', str(path)]
        for key in ('forever', 'frozen', 'realtime'):
            if self.vars[key].get():
                args.append('--' + key)
        # --forever and --duration-ms are mutually exclusive in the CLI.
        if self.vars['forever'].get():
            index = args.index('--duration-ms')
            del args[index:index + 2]
        output = self.home / ('biology-' + identifier)
        args += ['--output', str(output)]
        self.queue_job('biology', args, output)

    def doctor(self):
        self.queue_job('doctor', ['language', 'doctor', '--device', self.device.get(), '--preset', self.preset.get()], None)

    def queue_job(self, kind, args, output):
        if self.job.running or self.pending_job:
            raise ValueError('An experiment is already running')
        if self.busy_id is not None:
            raise ValueError('Finish or cancel the current model operation before starting an experiment')
        if self.recording:
            raise ValueError('Finish microphone recording before starting an experiment')
        self.pending_job = (kind, args, str(output) if output else None)
        self.tabs.select(self.pages['Runs & diagnostics'])
        if self.model_info:
            if self.model_info.get('path'):
                self.job_stage = 'saving'
                path = self.home / 'sessions' / (stamp() + '-before-job.pt')
                self.log('Saving chat session before experiment: ' + str(path))
                self.busy_id = self.worker.send('save_session', path=str(path), conversation=self.conversation)
            else:
                self.job_stage = 'unloading'
                self.busy_id = self.worker.send('unload')
        else:
            self.launch_pending_job()

    def launch_pending_job(self):
        kind, args, output = self.pending_job
        self.pending_job = None
        self.job_stage = None
        identifier = stamp()
        log = self.home / 'logs' / (identifier + '-' + kind + '.log')
        save_json(log.with_suffix('.json'), {'kind': kind, 'arguments': args, 'workspace': str(self.workspace), 'output': output})
        self.job.start(kind, args, self.workspace, log, output)
        self.log('Started ' + kind + '. Log: ' + str(log))
        self.status.set(kind.capitalize() + ' running')
        self.train_status.set('Experiment running. Stop requests finish the current training step or simulation tick before saving.')
        if output:
            self.run_path.set(output)
            self.last_image_stamp = None

    def stop_job(self, force=False):
        if self.pending_job:
            self.pending_job = None
        self.job.stop(force)
        if force and self.closing:
            self.worker.force_stop()
        self.status.set('Force stop requested' if force else 'Stop requested · waiting for checkpoint')

    def load_result(self):
        if self.job.running:
            raise ValueError('Wait for training to finish or checkpoint and stop it first')
        path = self.resolve(self.run_path.get()) / 'checkpoint.pt'
        if not path.is_file():
            raise ValueError('This run does not have a language checkpoint yet')
        self.checkpoint.set(str(path))
        self.load_model()
        self.tabs.select(self.pages['Chat'])

    def refresh_checkpoints(self):
        paths = []
        runs = self.workspace / 'runs'
        if runs.exists():
            paths = sorted(runs.rglob('checkpoint.pt'), key=lambda path: path.stat().st_mtime, reverse=True)[:60]
        values = [str(path) for path in paths]
        if self.checkpoint.get() and self.checkpoint.get() not in values:
            values.insert(0, self.checkpoint.get())
        self.checkpoint_picker.configure(values=values)
        if not self.checkpoint.get() and values:
            self.checkpoint.set(values[0])

    def listen(self):
        self.request('listen', path=self._input_path('wav_path'))
        self.replace_text(self.speech_transcript, '', readonly=True)
        self.audio_status.set('Transcribing…')

    def transcript_to_chat(self):
        text = self.speech_transcript.get('1.0', 'end-1c')
        if not text.strip():
            raise ValueError('Transcribe a recording first')
        self.replace_text(self.prompt, text)
        self.tabs.select(self.pages['Chat'])

    def reply_to_speech(self):
        reply = next((row['text'] for row in reversed(self.conversation) if row['role'] == 'model' and row['text']), '')
        if not reply:
            raise ValueError('There is no model reply to synthesize')
        self.replace_text(self.speech_text, reply)

    def speak(self):
        text = self.speech_text.get('1.0', 'end-1c')
        if not text.strip():
            raise ValueError('Enter text to synthesize')
        path = filedialog.asksaveasfilename(parent=self.root, initialdir=str(self.home),
                                            defaultextension='.wav', filetypes=[('WAV audio', '*.wav')])
        if path:
            self.request('speak', text=text, path=path, max_frames=int(self.vars['voice_frames'].get()))
            self.audio_status.set('Synthesizing…')

    def play_audio(self):
        if not self.last_audio:
            raise ValueError('Synthesize or record a WAV first')
        try:
            import sounddevice as sd
        except ImportError as error:
            raise ValueError("Install optional voice dependencies with: python -m pip install -e '.[voice]'") from error
        import wave
        import numpy as np
        with wave.open(str(self.last_audio), 'rb') as source:
            samples = np.frombuffer(source.readframes(source.getnframes()), dtype='<i2').copy()
            sd.play(samples, samplerate=source.getframerate(), blocking=False)
        self.audio_status.set('Playing ' + str(self.last_audio))

    def toggle_recording(self):
        if self.recording is not None:
            return self.finish_recording()
        if self.job.running:
            raise ValueError('Stop the experiment before recording audio')
        try:
            import sounddevice as sd
        except ImportError as error:
            raise ValueError("Install optional voice dependencies with: python -m pip install -e '.[voice]'") from error
        self.recorded, self.recorded_frames, self.record_overflow = [], 0, False

        def capture(data, frames, timing, status):
            if status:
                self.record_overflow = True
            if self.recorded_frames < 16000 * 30:
                self.recorded.append(bytes(data))
                self.recorded_frames += frames

        stream = sd.RawInputStream(samplerate=16000, blocksize=1600, channels=1, dtype='int16', callback=capture)
        stream.start()
        self.recording = stream
        self.record_button.configure(text='Finish recording')
        self.audio_status.set('Recording microphone · press Finish recording to save and transcribe.')

    def finish_recording(self):
        import wave
        if self.recording is None:
            return
        self.recording.stop()
        self.recording.close()
        self.recording = None
        self.record_button.configure(text='Record microphone')
        if self.record_overflow:
            raise ValueError('Microphone overflow detected. Recording discarded; try a less busy audio device.')
        path = self.home / 'audio' / (stamp() + '-input.wav')
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), 'wb') as target:
            target.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            target.writeframes(b''.join(self.recorded))
        self.last_audio = path
        self.wav_path.set(str(path))
        self.audio_status.set(f'Recording saved: {path}. Click Transcribe WAV when ready.')

    def log(self, text):
        if not hasattr(self, 'logs'):
            return
        self.logs.configure(state='normal')
        self.logs.insert('end', str(text) + '\n')
        if int(self.logs.index('end-1c').split('.')[0]) > 2000:
            self.logs.delete('1.0', '300.0')
        self.logs.see('end')
        self.logs.configure(state='disabled')

    def update_model_info(self, info):
        self.model_info = info
        if info is None:
            self.summary.set('No model loaded. Select a checkpoint or create a fresh model.')
            return
        config, environment = info['config'], info['environment']
        label = Path(info['path']).name if info.get('path') else 'Fresh untrained model'
        self.summary.set(f"{label}  ·  {info['parameters']:,} parameters  ·  {info['spiking_neurons']:,} units  ·  "
            f"{config['regions']} regions × {config['columns']} columns  ·  {environment.get('gpu', environment['device'])}  ·  "
            f"step {info['step']}  ·  speech {'trained' if info['speech_trained'] else 'untrained'}")
        if info.get('path'):
            self.checkpoint.set(info['path'])
        self.vars['source_checkpoint'].set(info.get('path') or '')

    def show_activity(self, populations):
        if not populations:
            self.activity_label.set('Activity unavailable for this restored session until reset.')
            return
        excitatory = [value for key, value in populations.items() if key.endswith('_excitatory')]
        inhibitory = [value for key, value in populations.items() if key.endswith('_inhibitory')]
        if excitatory and inhibitory:
            self.activity_label.set(f"Session mean spike fraction\nE {sum(excitatory) / len(excitatory):.4f} · "
                                    f"I {sum(inhibitory) / len(inhibitory):.4f}\nSpikes / unit / internal tick")

    def on_worker_event(self, event):
        kind = event['event']
        if kind == 'ready':
            self.worker_ready = True
            self.status.set('Ready')
            if self.initial_checkpoint:
                self.checkpoint.set(str(self.initial_checkpoint))
                self.initial_checkpoint = None
                self.guard(self.load_model)
        elif kind == 'loaded':
            self.update_model_info(event['info'])
            self.conversation, self.current_reply = [], None
            self.replace_text(self.transcript, '', readonly=True)
            self.chat_note('New session ready. ' + ('This model has random initial weights.' if not event['info'].get('path') else 'Checkpoint loaded.'))
            self.copy_architecture()
        elif kind == 'unloaded':
            self.update_model_info(None)
        elif kind == 'model_saved':
            self.update_model_info(event['info'])
            self.footer.set('Model weights saved: ' + event['path'] + '. Use Initialize from weights to train this export.')
            self.refresh_checkpoints()
        elif kind == 'delta':
            if self.current_reply is not None:
                self.current_reply['text'] += event['text']
            self.chat_insert(event['text'])
        elif kind == 'turn_end':
            self.chat_insert('\n')
            self.chat_metrics.set(f"{event['tokens']} output tokens · {event['tokens_s']:.1f} tokens/s\n"
                                  f"{event['seconds']:.2f} seconds · stopped: {event['reason']}")
            self.show_activity(event['populations'])
            self.current_reply = None
        elif kind == 'activity':
            self.show_activity(event['populations'])
        elif kind == 'score':
            result = (f"Fresh-state text score · CE {event['loss']:.4f} · perplexity {event['perplexity']:.2f} · "
                      f"accuracy {event['accuracy']:.1%} over {event['targets']} next-byte/end targets")
            self.chat_note(result)
            self.chat_metrics.set(result)
        elif kind == 'reset':
            self.conversation, self.current_reply = [], None
            self.replace_text(self.transcript, '', readonly=True)
            self.chat_note('Neural state and conversation reset.')
            self.activity_label.set('Session reset.')
        elif kind == 'settings':
            self.footer.set('Inference settings applied. Changes also accompany the next prompt.')
        elif kind == 'session_saved':
            self.footer.set('Session and conversation saved: ' + event['path'])
        elif kind == 'session_loaded':
            self.conversation = event['conversation'][-500:]
            self.replace_text(self.transcript, '', readonly=True)
            for item in self.conversation:
                self.chat_insert('\n' + item.get('role', 'message').capitalize() + '\n', 'user' if item.get('role') == 'user' else 'model')
                self.chat_insert(item['text'] + '\n')
            self.chat_note('Session restored.')
        elif kind == 'transcript_delta':
            self.speech_transcript.configure(state='normal')
            self.speech_transcript.insert('end', event['text'])
            self.speech_transcript.configure(state='disabled')
        elif kind == 'transcript':
            self.replace_text(self.speech_transcript, event['text'], readonly=True)
            self.audio_status.set(f"Transcription {'cancelled' if event['cancelled'] else 'complete'} · real-time factor {event['real_time_factor']:.3f}")
        elif kind == 'audio':
            self.last_audio = Path(event['path'])
            self.audio_status.set(f"Saved {event['seconds']:.2f}s audio · real-time factor {event['real_time_factor']:.3f} · {event['path']}")
        elif kind in ('error', 'fatal'):
            self.footer.set(event['message'])
            self.log(event['message'])
            self.chat_note('Operation failed: ' + event['message'])
            self.current_reply = None
            if self.pending_job and event.get('id') == self.busy_id:
                self.pending_job, self.job_stage = None, None
            if kind == 'fatal':
                self.worker_ready = False
        elif kind == 'cancelled':
            self.footer.set('Operation cancelled.')
        elif kind == 'log':
            self.log(event['text'])
        elif kind == 'done':
            if self.busy_id == event.get('id'):
                self.busy_id = None
                if self.pending_job and not self.closing:
                    if self.job_stage == 'saving':
                        self.job_stage = 'unloading'
                        self.busy_id = self.worker.send('unload')
                    elif self.job_stage == 'unloading':
                        self.guard(self.launch_pending_job)
            if not self.job.running and not self.closing:
                self.status.set('Ready' if self.worker_ready else 'Worker stopped')
        elif kind == 'exited':
            self.worker_ready, self.busy_id = False, None
            if not self.closing:
                self.update_model_info(None)
                self.status.set('Model worker stopped · restart in diagnostics')
                self.log(f"Model worker exited with code {event['returncode']}")

    def on_job_event(self, event):
        kind = event['event']
        if kind == 'job_log':
            self.log(event['text'])
        elif kind == 'corpus_progress':
            row = event['row']
            ready = row['phase'] in ('ready', 'cancelled')
            self.train_progress.stop()
            self.train_progress.configure(mode='determinate' if ready else 'indeterminate')
            if not ready:
                self.train_progress.start(40)
            message = row.get('message') or f"Preparing SlimPajama {row.get('split', '')}: {row.get('documents', 0)}/{row.get('target', 0)} documents"
            self.train_status.set(message)
        elif kind == 'metrics':
            row = event['row']
            self.train_progress.configure(value=row['step'])
            pieces = [f"Step {row['step']}"]
            for key, title in [('val_text_loss', 'Validation CE'), ('val_accuracy', 'Accuracy'), ('activity', 'Spike fraction'), ('val_cer', 'CER')]:
                if key in row:
                    pieces.append(f'{title} {row[key]:.4f}')
            self.train_status.set(' · '.join(pieces))
        elif kind == 'job_done':
            self.train_progress.stop()
            self.train_progress.configure(mode='determinate')
            success = event['returncode'] == 0
            self.status.set('Experiment finished' if success else f"Experiment exited ({event['returncode']})")
            self.log(f"{event['kind']} finished, exit code {event['returncode']}. Full log: {self.job.log_path}")
            self.refresh_checkpoints()
            self.refresh_run(force=True)
            if not success:
                self.footer.set('Experiment failed. Inspect the log above; earlier completed checkpoints are retained.')
            else:
                self.footer.set('Experiment complete. Inspect charts or use Load resulting checkpoint in Train.')

    def refresh_run(self, force=False):
        value = self.run_path.get().strip()
        if not value:
            return
        directory = self.resolve(value)
        if not directory.is_dir():
            return
        images = sorted(str(path.relative_to(directory)) for path in directory.glob('*.png'))
        images += sorted(str(path.relative_to(directory)) for path in (directory / 'plots').glob('*.png') if not path.stem.rsplit('_', 1)[-1].isdigit())
        self.plot_picker.configure(values=images)
        if self.plot_choice.get() not in images and images:
            self.plot_choice.set(images[0])
        metrics = directory / 'metrics.jsonl'
        if metrics.is_file():
            with metrics.open('rb') as stream:
                stream.seek(max(0, metrics.stat().st_size - 65536))
                lines = stream.read().decode('utf-8', errors='replace').splitlines()
            for line in reversed(lines):
                try:
                    row = json.loads(line)
                    if 'step' in row:
                        self.run_metrics.set(f"Step {row['step']} · loss {row.get('val_text_loss', 0):.4f} · accuracy {row.get('val_accuracy', 0):.1%} · "
                                             f"spike fraction {row.get('activity', 0):.4f} · peak GPU allocation {row.get('peak_gpu_allocated_mib', 0):.1f} MiB")
                    elif 'mean_hz' in row:
                        self.run_metrics.set(f"Biological time {row.get('time_ms', 0) / 1000:.2f}s · mean firing {row['mean_hz']:.2f} Hz/neuron · "
                                             f"E {row.get('excitatory_hz', 0):.2f} · I {row.get('inhibitory_hz', 0):.2f}")
                    break
                except (ValueError, TypeError):
                    continue
        if not images:
            self.plot_label.configure(image='', text='Waiting for the first evaluation chart…')
            return
        path = directory / self.plot_choice.get()
        size = (max(100, self.plot_frame.winfo_width() - 8), max(100, self.plot_frame.winfo_height() - 8))
        key = (str(path), path.stat().st_mtime_ns, size)
        if force or key != self.last_image_stamp:
            try:
                from PIL import Image, ImageTk
                with Image.open(path) as source:
                    preview = source.copy()
                preview.thumbnail(size)
                self.image = ImageTk.PhotoImage(preview, master=self.root)
                self.plot_label.configure(image=self.image, text='')
                self.last_image_stamp = key
            except OSError:
                pass  # A running experiment may currently be replacing a PNG.

    def open_report(self):
        path = self.resolve(self.run_path.get()) / 'report.html'
        if not path.is_file():
            raise ValueError('No HTML report in this run yet')
        webbrowser.open(path.as_uri())

    def restart_worker(self):
        if self.job.running or self.pending_job:
            raise ValueError('Stop the experiment before restarting the model worker')
        self.worker.force_stop()
        self.worker_ready, self.busy_id = False, None
        self.update_model_info(None)
        self.status.set('Restarting model worker…')

        def restart_when_exited():
            if self.worker.process.poll() is None:
                self.root.after(50, restart_when_exited)
            elif not self.closing:
                self.worker = ModelProcess(self.workspace)
        self.root.after(50, restart_when_exited)

    def _restore_preferences(self):
        path = self.home / 'preferences.json'
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            for key in ('checkpoint', 'device', 'mode', 'preset', 'temperature', 'top_k', 'max_new_tokens', 'idle_hz',
                        'train', 'validation', 'speech_train', 'speech_validation', 'source_checkpoint', 'train_mode', 'run_path',
                        'data_source', 'dataset_id', 'dataset_revision', 'dataset_seed', 'train_documents', 'validation_documents', 'max_document_bytes'):
                if key in data.get('values', {}):
                    self.vars[key].set(data['values'][key])
            for key, value in data.get('training', {}).items():
                if key in self.train_vars:
                    self.train_vars[key].set(value)
            for name, editor in [('language_config', self.config_editor), ('biology_config', self.bio_editor)]:
                if name in data:
                    self.replace_text(editor, data[name])
            self.validate_config()
            self.update_data_source()
        except (ValueError, TypeError, OSError, tk.TclError) as error:
            self.log(f'Could not restore all preferences: {error}')

    def _save_preferences(self):
        keys = ('checkpoint', 'device', 'mode', 'preset', 'temperature', 'top_k', 'max_new_tokens', 'idle_hz',
                'train', 'validation', 'speech_train', 'speech_validation', 'source_checkpoint', 'train_mode', 'run_path',
                'data_source', 'dataset_id', 'dataset_revision', 'dataset_seed', 'train_documents', 'validation_documents', 'max_document_bytes')
        save_json(self.home / 'preferences.json', {'values': {key: self.vars[key].get() for key in keys},
                  'training': {key: value.get() for key, value in self.train_vars.items()},
                  'language_config': self.config_editor.get('1.0', 'end-1c'), 'biology_config': self.bio_editor.get('1.0', 'end-1c')})

    def poll(self):
        for source, handler in ((self.worker.events, self.on_worker_event), (self.job.events, self.on_job_event)):
            for _ in range(250):
                try:
                    event = source.get_nowait()
                except queue.Empty:
                    break
                try:
                    handler(event)
                except Exception as error:
                    self.log(f'Workbench update: {error}')
                    self.footer.set(str(error))
        if self.recording and (self.recorded_frames >= 30 * 16000 or self.record_overflow):
            self.guard(self.finish_recording)
        if time.monotonic() - self.last_run_refresh >= 2:
            self.last_run_refresh = time.monotonic()
            if self.tabs.select() == str(self.pages['Runs & diagnostics']):
                try:
                    self.refresh_run()
                except (ValueError, OSError, tk.TclError) as error:
                    self.log(f'Run refresh: {error}')
        if self.closing and not self.job.running and self.worker.process.poll() is not None:
            self.root.destroy()
            return
        self.root.after(40, self.poll)

    def close(self):
        if self.closing:
            return
        if self.recording:
            self.guard(self.finish_recording)
        try:
            self._save_preferences()
        except (OSError, ValueError) as error:
            self.log(f'Preferences: {error}')
        self.closing, self.pending_job = True, None
        self.worker.stop()
        self.job.stop()
        self.status.set('Closing · waiting for processes to stop and save')
        self.footer.set('Training stops at a completed step. Force stop (no save) is available in Runs & diagnostics if needed.')


def run_gui(args):
    try:
        root = tk.Tk()
    except tk.TclError as error:
        raise ValueError('Cannot open a desktop display. Run the GUI in a graphical desktop session with Tk installed.') from error
    app = Workbench(root, args.workspace, args.checkpoint)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.close()
        root.mainloop()
