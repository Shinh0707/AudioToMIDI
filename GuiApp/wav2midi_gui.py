import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
import subprocess
import threading
from pathlib import Path
import sys
import os
import shutil
import numpy as np
from scipy.io import wavfile
import yaml

def mix_audio(file1, file2, output_file):
    rate1, data1 = wavfile.read(file1)
    rate2, data2 = wavfile.read(file2)
    
    if rate1 != rate2:
        raise ValueError("Sample rates do not match!")
        
    # Ensure same length
    min_len = min(len(data1), len(data2))
    data1 = data1[:min_len]
    data2 = data2[:min_len]
    
    mixed = data1 + data2
    
    # Clip to prevent overflow if necessary, though wavfile.write handles some, better to be safe for int16
    if mixed.dtype == np.int16:
        mixed = np.clip(mixed, -32768, 32767)
    
    wavfile.write(output_file, rate1, mixed)

def scan_bandit_models():
    """
    Scans GuiApp/bandit for subdirectories containing configuration files (*.yaml) and checkpoints (*.ckpt, *.chpt).
    Returns a dict {label: path_to_directory}.
    """
    models = {}
    base_dir = Path("GuiApp") / "bandit"
    
    if base_dir.exists():
        for item in base_dir.iterdir():
            if item.is_dir():
                # Check for yaml and ckpt
                yamls = list(item.glob("*.yaml"))
                ckpts = list(item.glob("*.ckpt")) + list(item.glob("*.chpt"))
                
                if yamls and ckpts:
                    models[item.name] = item.resolve()
    
    return models

class Wav2MidiApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Wav2Midi Converter")
        self.root.geometry("600x550")

        # Variables
        self.file_path = tk.StringVar()
        self.force_separate = tk.BooleanVar()
        self.force_midi = tk.BooleanVar()
        self.use_6_stems = tk.BooleanVar()
        self.use_bandit = tk.BooleanVar()
        self.bandit_model_name = tk.StringVar()
        self.is_running = False
        
        # Scan models
        self.bandit_models = scan_bandit_models()
        self.bandit_model_names = list(self.bandit_models.keys())
        
        if self.bandit_model_names:
            self.bandit_model_name.set(self.bandit_model_names[0])
        
        # Build UI
        self.create_widgets()

    def create_widgets(self):
        # File Selection Frame
        frame_top = tk.Frame(self.root, padx=10, pady=10)
        frame_top.pack(fill=tk.X)

        tk.Label(frame_top, text="Selected Audio File:").pack(anchor=tk.W)
        
        entry_frame = tk.Frame(frame_top)
        entry_frame.pack(fill=tk.X, pady=5)
        
        tk.Entry(entry_frame, textvariable=self.file_path).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(entry_frame, text="Browse", command=self.browse_file).pack(side=tk.LEFT, padx=5)

        # Options
        frame_options = tk.LabelFrame(self.root, text="Options", padx=5, pady=5)
        frame_options.pack(fill=tk.X, pady=5)
        
        tk.Checkbutton(frame_options, text="Enable 6-stem separation (Guitar/Piano)", variable=self.use_6_stems).pack(anchor=tk.W)
        tk.Checkbutton(frame_options, text="Force Separate (Re-run all separations)", variable=self.force_separate).pack(anchor=tk.W)
        tk.Checkbutton(frame_options, text="Force MIDI Conversion", variable=self.force_midi).pack(anchor=tk.W)
        
        # BandIt Options
        frame_bandit = tk.LabelFrame(frame_options, text="BandIt (Cinematic Separation)", padx=5, pady=5)
        frame_bandit.pack(fill=tk.X, pady=5)
        
        self.chk_bandit = tk.Checkbutton(frame_bandit, text="Use BandIt (Separate Speech/Music/Effects first)", variable=self.use_bandit)
        self.chk_bandit.pack(anchor=tk.W)
        
        frame_model = tk.Frame(frame_bandit)
        frame_model.pack(fill=tk.X, pady=2)
        tk.Label(frame_model, text="Model:").pack(side=tk.LEFT)
        
        self.combo_model = ttk.Combobox(frame_model, textvariable=self.bandit_model_name, values=self.bandit_model_names, state="readonly", width=40)
        self.combo_model.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.combo_model.bind("<<ComboboxSelected>>", self.on_model_selected)

        # Dynamic Stem Mapping Frame
        self.frame_stems = tk.Frame(frame_bandit)
        self.frame_stems.pack(fill=tk.X, pady=5)
        
        # Variables to store mapping state
        self.demucs_input_stem = tk.StringVar(value="music") 
        self.stem_merge_targets = {} 

        if not self.bandit_model_names:
            self.combo_model.set("No models found in GuiApp/bandit")
            self.use_bandit.set(False)
            self.chk_bandit.config(state=tk.DISABLED)
        else:
             self.setup_bandit_ui()

        # Action Frame
        frame_action = tk.Frame(self.root, padx=10, pady=10)
        frame_action.pack(fill=tk.X)
        
        self.btn_convert = tk.Button(frame_action, text="Start Conversion", command=self.start_conversion, bg="#dddddd", height=2)
        self.btn_convert.pack(fill=tk.X)

        # Logs Frame
        frame_logs = tk.Frame(self.root, padx=10, pady=10)
        frame_logs.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(frame_logs, text="Logs:").pack(anchor=tk.W)
        self.log_area = scrolledtext.ScrolledText(frame_logs, height=15)
        self.log_area.pack(fill=tk.BOTH, expand=True)

    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.ogg")] # Added other common types supported by demucs usually, but sticking to wav primarily
        )
        if filename:
            self.file_path.set(filename)

    def browse_ckpt(self):
        # ckpt can be a file or directory in inference.py, but file is safer to select
        # Updated to prefer directory since our setup creates a directory structure
        filename = filedialog.askdirectory(title="Select BandIt Model Directory (containing checkpoints/ and hparams.yaml)") 
        if not filename:
             filename = filedialog.askopenfilename(title="Select BandIt Checkpoint File", filetypes=[("Checkpoint", "*.ckpt")])
        
        if filename:
             self.bandit_ckpt_path.set(filename)

    def on_model_selected(self, event):
        self.setup_bandit_ui()
        
    def setup_bandit_ui(self):
        # Clear existing widgets in frame_stems
        for widget in self.frame_stems.winfo_children():
            widget.destroy()
            
        model_label = self.bandit_model_name.get()
        if not model_label or model_label not in self.bandit_models:
            return

        ckpt_path = self.bandit_models[model_label]
        yaml_files = list(ckpt_path.glob("*.yaml"))
        if not yaml_files:
            return
            
        try:
            with open(yaml_files[0], 'r') as f:
                config = yaml.safe_load(f)
            
            # Extract stems
            stems = []
            if "training" in config and "instruments" in config["training"]:
                stems = config["training"]["instruments"]
            elif "model" in config and "stems" in config["model"]:
                stems = config["model"]["stems"]
            
            if not stems:
                tk.Label(self.frame_stems, text="No stems found in config.").pack()
                return

            # Headers
            headers = tk.Frame(self.frame_stems)
            headers.pack(fill=tk.X)
            tk.Label(headers, text="Stem Name", width=15, anchor=tk.W).pack(side=tk.LEFT)
            tk.Label(headers, text="Demucs Input", width=15, anchor=tk.CENTER).pack(side=tk.LEFT)
            tk.Label(headers, text="Merge To Checkpoint", anchor=tk.W).pack(side=tk.LEFT)
            
            self.stem_merge_targets = {} # Reset
            
            default_demucs_input = "music" if "music" in stems else stems[0]
            self.demucs_input_stem.set(default_demucs_input)

            merge_options = ["None", "Vocals", "Drums", "Bass", "Other", "Guitar", "Piano"]

            for stem in stems:
                row = tk.Frame(self.frame_stems)
                row.pack(fill=tk.X, pady=2)
                
                tk.Label(row, text=stem, width=15, anchor=tk.W).pack(side=tk.LEFT)
                
                tk.Radiobutton(row, variable=self.demucs_input_stem, value=stem, width=15).pack(side=tk.LEFT)
                
                # Default map guess
                default_merge = "None"
                lower = stem.lower()
                if "speech" in lower or "vocal" in lower: default_merge = "Vocals"
                elif "drum" in lower: default_merge = "Drums"
                elif "bass" in lower: default_merge = "Bass"
                elif "guitar" in lower: default_merge = "Guitar"
                elif "piano" in lower: default_merge = "Piano"
                
                target_var = tk.StringVar(value=default_merge)
                self.stem_merge_targets[stem] = target_var
                
                ttk.Combobox(row, textvariable=target_var, values=merge_options, state="readonly", width=15).pack(side=tk.LEFT, padx=5)

        except Exception as e:
            tk.Label(self.frame_stems, text=f"Error loading config: {e}").pack()

    def log(self, message):
        self.root.after(0, self._log_thread_safe, message)

    def _log_thread_safe(self, message):
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.see(tk.END)

    def toggle_inputs(self, enable):
        state = tk.NORMAL if enable else tk.DISABLED
        self.btn_convert.config(state=state)

    def start_conversion(self):
        if self.is_running:
            return

        input_path = self.file_path.get()
        if not input_path:
            messagebox.showerror("Error", "Please select an input file first.")
            return
        
        if not os.path.exists(input_path):
             messagebox.showerror("Error", "Selected file does not exist.")
             return

        self.is_running = True
        self.toggle_inputs(False)
        self.log("--- Starting Process ---")


        # Capture settings safe for thread
        bandit_settings = {}
        if self.use_bandit.get():
             bandit_settings["enabled"] = True
             bandit_settings["demucs_input_stem"] = self.demucs_input_stem.get()
             bandit_settings["merge_targets"] = {k: v.get() for k, v in self.stem_merge_targets.items()}
        else:
             bandit_settings["enabled"] = False
        
        # Start processing in a separate thread
        thread = threading.Thread(target=self.run_conversion, args=(Path(input_path), bandit_settings))
        thread.start()

    def run_command_capture(self, cmd, description):
        self.log(f"Running: {description}")
        self.log(f"Command: {' '.join(cmd)}")
        try:
            # Using Popen to capture output in real-time could be better, but simpler run() is easier to manage for now.
            # To show output in real-time we need Popen.
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            
            for line in process.stdout:
                self.log(line.strip())
            
            process.wait()
            
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)
                
            self.log(f"--- Finished {description} ---")
            return True
        except subprocess.CalledProcessError as e:
            self.log(f"Error during {description}: {e}")
            return False
        except FileNotFoundError:
            self.log(f"Command not found: {cmd[0]}")
            return False

    def run_conversion(self, input_path, bandit_settings):
        try:
            song_name = input_path.stem
            
            # Define output directories
            # Assuming the script is run from project root, or we use relative paths.
            # The original script used Path("outputs") which is relative to CWD.
            # We should probably respect that or use input file location?
            # User request: "出力先指定, defaultは このディレクトリ/outputs/[曲名]/"
            # 'このディレクトリ' implies the directory where the app is running / project root.
            # Since we are placing this in GuiApp/, we need to be careful.
            # If we run from GuiApp/, "outputs" will be inside GuiApp.
            # Better to go one level up if running from GuiApp, or simple assume CWD is project root.
            # Let's assume CWD is project root (/Users/shinh0707/Documents/Helpers/WAV2MIDI)
            
            base_output_dir = Path("outputs") / song_name
            audio_output_dir = base_output_dir
            midi_output_dir = base_output_dir / "midi"

            # Create directories
            audio_output_dir.mkdir(parents=True, exist_ok=True)
            midi_output_dir.mkdir(parents=True, exist_ok=True)
            
            self.log(f"Output Directory: {base_output_dir}")

            current_input_audio = input_path
            
            # --- BandIt Separation (Optional) ---
            bandit_speech = None
            bandit_effects = None
            
            if self.use_bandit.get():
                model_label = self.bandit_model_name.get()
                if not model_label or model_label not in self.bandit_models:
                     raise Exception("BandIt enabled but no valid model selected.")
                
                ckpt_path = self.bandit_models[model_label]
                
                # Check config
                yaml_files = list(ckpt_path. glob("*.yaml"))
                if not yaml_files:
                    raise Exception(f"No config file found in {ckpt_path}")
                config_path = yaml_files[0] # Assume the first yaml is the config
                
                with open(config_path, 'r') as f:
                    model_config = yaml.safe_load(f)
                
                # Try to find model_type
                # ZFTurbo configs usually have:
                # training:
                #    model_type: mdx23c
                # or model: { ... } where inference.py infers it? 
                # inference.py says: args.model_type.
                # If we don't pass it, we must know it.
                # Let's try to find it in the yaml.
                zft_model_type = "bandit" # Default fallback
                if "training" in model_config and "model_type" in model_config["training"]:
                    zft_model_type = model_config["training"]["model_type"]
                elif "model_type" in model_config:
                     zft_model_type = model_config["model_type"]
                
                # Also find checkpoint
                ckpt_files = list(ckpt_path.glob("*.ckpt")) + list(ckpt_path.glob("*.chpt"))
                if not ckpt_files:
                     raise Exception(f"No checkpoint found in {ckpt_path}")
                model_ckpt_path = ckpt_files[0]

                # Get expected stems
                # In ZFTurbo config: training: instruments: [speech, music, sfx]
                expected_stems = []
                if "training" in model_config and "instruments" in model_config["training"]:
                    expected_stems = model_config["training"]["instruments"]
                elif "model" in model_config and "stems" in model_config["model"]:
                    expected_stems = model_config["model"]["stems"] # Old BandIt style fallback
                     
                self.log(f"Selected Model: {model_label} (Type: {zft_model_type})")
                self.log(f"Expected Stems: {expected_stems}")
                
                bandit_output_dir = audio_output_dir / "bandit"
                bandit_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Check if ZFTurbo separation is actually needed
                should_run_zft = True
                if bandit_output_dir.exists() and any(bandit_output_dir.rglob("*.wav")):
                     if self.force_separate.get():
                         self.log("Force separation enabled: Re-running ZFTurbo.")
                     else:
                         self.log(f"ZFTurbo output already exists at {bandit_output_dir}. Skipping separation.")
                         should_run_zft = False

                if should_run_zft:
                    self.log("Running ZFTurbo Separation...")
                    
                    # ZFTurbo inference.py usage:
                    # --model_type ... --config_path ... --start_check_point ... --input_folder ... --store_dir ...
                    # It processes a folder. So we need to point to the input file's folder?
                    # No, inference.py processes all files in input_folder.
                    # We want to process ONLY our file.
                    # Simplest way: Temporary folder with just the input file? 
                    # Or modify inference.py? 
                    # run_folder(...) iterates glob(input_folder/*.*).
                    # Let's create a temporary input folder inside outputs/tmp_input?
                    
                    tmp_input_dir = base_output_dir / "tmp_input"
                    tmp_input_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Symlink or Copy file to tmp_input
                    tmp_input_file = tmp_input_dir / input_path.name
                    if not tmp_input_file.exists():
                        shutil.copy2(input_path, tmp_input_file)
                    
                    zft_cmd = [
                        sys.executable, "Music-Source-Separation-Training/inference.py",
                        "--model_type", zft_model_type,
                        "--config_path", str(config_path),
                        "--start_check_point", str(model_ckpt_path),
                        "--input_folder", str(tmp_input_dir),
                        "--store_dir", str(bandit_output_dir),
                        "--disable_detailed_pbar" # Clean logs
                    ]
                    
                    # Optional: Use GPU if available (default in inference.py is to use valid device)
                    
                    if not self.run_command_capture(zft_cmd, "Separation Inference"):
                        raise Exception("Separation failed")
                    
                    # Clean up tmp
                    # shutil.rmtree(tmp_input_dir) # Keep for debug or delete? Delete to be clean.
                    shutil.rmtree(tmp_input_dir, ignore_errors=True)
                else:
                    self.log("Skipped ZFTurbo execution.")
                                   # --- 16-bit Conversion & Logic Update ---
                
                # Identify stems based on output
                found_files = list(bandit_output_dir.rglob("*.wav"))
                self.log(f"Generated files: {[f.name for f in found_files]}")
                
                # Convert all BandIt outputs to 16-bit Int
                self.log("Converting BandIt outputs to 16-bit WAV...")
                bandit_stems = {} # Map stem_name -> file_path
                
                for f in found_files:
                    try:
                        rate, data = wavfile.read(f)
                        # Check if float
                        if data.dtype == np.float32 or data.dtype == np.float64:
                            data_int16 = np.int16(data * 32767)
                            wavfile.write(f, rate, data_int16)
                            self.log(f"Converted {f.name} to 16-bit.")
                        else:
                            self.log(f"{f.name} is already {data.dtype}.")
                    except Exception as e:
                        self.log(f"Error converting {f.name}: {e}")
                        
                    bandit_stems[f.stem.lower()] = f

                # Select Demucs Input
                if "demucs_input_stem" in bandit_settings:
                    target_stem_name = bandit_settings["demucs_input_stem"].lower()
                else:
                    target_stem_name = "music" # Fallback

                music_stem = None
                if target_stem_name in bandit_stems:
                    music_stem = bandit_stems[target_stem_name]
                else:
                    # Fuzzy match fallback
                    for name in bandit_stems:
                        if target_stem_name in name:
                            music_stem = bandit_stems[name]
                            break
                
                if not music_stem:
                    raise Exception(f"Selected Demucs input stem '{target_stem_name}' not found in BandIt outputs.")
                
                current_input_audio = music_stem
                self.log(f"Using BandIt stem '{music_stem.name}' as input for Demucs.")

            # Step 1: Split audio with Demucs
            # demucs output dir depends on model name
            model_name = "htdemucs_6s" if self.use_6_stems.get() else "htdemucs"
            demucs_output_path = audio_output_dir / model_name / song_name
            
            should_run_demucs = True
            if demucs_output_path.exists() and any(demucs_output_path.rglob("*.wav")):
                if self.force_separate.get():
                    self.log("Force separation enabled.")
                else:
                    self.log(f"Demucs output already exists at {demucs_output_path}. Skipping separation.")
                    should_run_demucs = False
            
            if should_run_demucs:
                model_name = "htdemucs_6s" if self.use_6_stems.get() else "htdemucs"
                demucs_cmd = [
                    "demucs",
                    "-n", model_name,
                    str(current_input_audio),
                    "-o", str(audio_output_dir)
                ]
                if not self.run_command_capture(demucs_cmd, "Demucs (Audio Separation)"):
                    raise Exception("Demucs failed")
            
            # Find the generated wav files.
            wav_files = list(audio_output_dir.rglob("*.wav"))
            
            if not wav_files:
                self.log(f"Warning: No wav files found in {audio_output_dir}.")
                raise Exception("No wav files found")
            
            # --- Post-Process: Merge BandIt Stems ---
            if self.use_bandit.get() and bandit_settings.get("merge_targets"):
                self.log("Processing Merge Targets...")
                demucs_map = {w.stem.lower(): w for w in wav_files} 
                merge_map = bandit_settings["merge_targets"]

                for b_stem_name, target in merge_map.items():
                    if target == "None":
                        continue
                    
                    target_key = target.lower() # vocals, drums...
                    
                    # Find BandIt file
                    b_file = bandit_stems.get(b_stem_name.lower())
                    if not b_file:
                        self.log(f"Warning: Could not find BandIt output for stem '{b_stem_name}' to merge.")
                        continue
                        
                    # Find Demucs target
                    d_file = demucs_map.get(target_key)
                    
                    if d_file:
                        self.log(f"Merging BandIt '{b_stem_name}' into Demucs '{target}'.")
                        try:
                            mix_audio(str(b_file), str(d_file), str(d_file)) # Overwrite Demucs file
                        except Exception as e:
                            self.log(f"Failed to merge: {e}")
                    else:
                        self.log(f"Warning: Demucs target '{target}' not found. Cannot merge '{b_stem_name}'.")

            if not wav_files:
                self.log(f"Warning: No wav files found in {audio_output_dir}. Please check if Demucs ran correctly.")
                raise Exception("No wav files found")

            self.log(f"Found {len(wav_files)} split audio files.")

            # Step 2: Convert to MIDI
            for wav_file in wav_files:
                stem_name = wav_file.stem  # e.g., "drums", "vocals", "bass", "other"
                
                if stem_name == "drums":
                    midi_out = midi_output_dir / f"{stem_name}_adtof.mid"
                    
                    if midi_out.exists() and not self.force_midi.get():
                        self.log(f"MIDI file {midi_out.name} already exists. Skipping.")
                        continue

                    # ADTOF
                    adtof_cmd = [
                        "adtof",
                        "--audio", str(wav_file),
                        "--out", str(midi_out),
                        "--device", "cpu"
                    ]
                    self.run_command_capture(adtof_cmd, f"ADTOF (Drums) for {wav_file.name}")
                elif stem_name != "effects":
                    # Basic Pitch
                    # basic-pitch <output_dir> <input_audio>
                    expected_midi_name = f"{wav_file.stem}_basic_pitch.mid"
                    midi_out = midi_output_dir / expected_midi_name
                    
                    if midi_out.exists() and not self.force_midi.get():
                         self.log(f"MIDI file {midi_out.name} already exists. Skipping.")
                         continue

                    basic_pitch_cmd = [
                        "basic-pitch",
                        str(midi_output_dir),
                        str(wav_file)
                    ]
                    
                    # User request: Finer granularity (1/32, 1/64) for bass and other.
                    # Default minimum note length is ~58ms. 30ms is approx 1/64 at 120bpm.
                    # Also applying to guitar and piano for 6-stem mode.
                    if stem_name in ["bass", "other", "guitar", "piano"]:
                        basic_pitch_cmd.extend(["--minimum-note-length", "30"])

                    self.run_command_capture(basic_pitch_cmd, f"Basic Pitch for {wav_file.name}")

            self.log("\n--- All tasks completed successfully ---")
            self.root.after(0, lambda: messagebox.showinfo("Success", "Conversion Completed!"))
            
        except Exception as e:
            err_msg = f"An error occurred: {e}"
            self.log(f"\nError: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", err_msg))
        finally:
            self.is_running = False
            self.root.after(0, lambda: self.toggle_inputs(True))

if __name__ == "__main__":
    root = tk.Tk()
    app = Wav2MidiApp(root)
    root.mainloop()
