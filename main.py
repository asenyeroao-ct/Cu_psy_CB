import customtkinter as ctk
import threading
import queue
import time
import math
import numpy as np
import cv2
import tkinter as tk
import os
import json
import glob
import logging
import pathlib

from cyndilib.finder import Finder
from cyndilib.receiver import Receiver
from cyndilib.wrapper.ndi_recv import RecvColorFormat, RecvBandwidth
from cyndilib.video_frame import VideoFrameSync
from cyndilib.audio_frame import AudioFrameSync

from config import config
from mouse import Mouse, is_button_pressed
from detection import load_model, perform_detection

# Create data directory if it doesn't exist
pathlib.Path("data").mkdir(exist_ok=True)

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s',
    handlers=[
        logging.FileHandler('data/bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("COLORBOT")


BUTTONS = {
    0: 'Left Mouse Button',
    1: 'Right Mouse Button',
    2: 'Middle Mouse Button',
    3: 'Side Mouse 4 Button',
    4: 'Side Mouse 5 Button'
}

class LanguageManager:
    def __init__(self):
        self.languages = {}
        self.current_language = "english"
        self.load_languages()
        
    def load_languages(self):
        language_files = glob.glob("languages/*.json")
        for file in language_files:
            try:
                with open(file, "r", encoding="utf-8") as f:
                    lang_data = json.load(f)
                    lang_name = os.path.basename(file).split(".")[0]
                    self.languages[lang_name] = lang_data
            except Exception as e:
                print(f"Error loading language file {file}: {e}")
        
        # Ensure English is available as fallback
        if "english" not in self.languages:
            self.languages["english"] = {
                "language_name": "English",
                "general": {"status": "Status", "connected": "Connected", "disconnected": "Disconnected"},
                "aimbot": {"x_speed": "X Speed", "y_speed": "Y Speed"},
                "triggerbot": {"tb_fov_size": "TB FOV Size", "tb_delay": "TB Delay"},
                "config": {"choose_config": "Choose a config", "save": "Save"}
            }
    
    def get_language_names(self):
        return [data.get("language_name", key) for key, data in self.languages.items()]
    
    def get_language_by_name(self, display_name):
        for key, data in self.languages.items():
            if data.get("language_name") == display_name:
                return key
        return "english"  # Default fallback
    
    def set_language(self, lang_key):
        if lang_key in self.languages:
            self.current_language = lang_key
    
    def get_text(self, section, key, default=None):
        try:
            return self.languages[self.current_language][section][key]
        except (KeyError, TypeError):
            # Try English as fallback
            try:
                return self.languages["english"][section][key]
            except (KeyError, TypeError):
                return default if default is not None else key

def threaded_silent_move(controller, dx, dy):
    """Petit move-restore pour le mode Silent."""
    controller.move(dx, dy)
    time.sleep(0.001)
    controller.click()
    time.sleep(0.001)
    controller.move(-dx, -dy)


class AimTracker:
    def __init__(self, app, target_fps=80):
        self.app = app
        # --- Params (avec valeurs fallback) ---
        # Common parameters
        self.normalsmooth = float(getattr(config, "normalsmooth", 10))
        self.normalsmoothfov = float(getattr(config, "normalsmoothfov", 10))
        self.mouse_dpi = float(getattr(config, "mouse_dpi", 800))
        self.fovsize = float(getattr(config, "fovsize", 300))
        self.in_game_sens = float(getattr(config, "in_game_sens", 7))
        
        # Main aimbot button parameters
        self.main_x_speed = float(getattr(config, "main_x_speed", 0.5))
        self.main_y_speed = float(getattr(config, "main_y_speed", 0.5))
        
        # Secondary aimbot button parameters
        self.sec_x_speed = float(getattr(config, "sec_x_speed", 0.5))
        self.sec_y_speed = float(getattr(config, "sec_y_speed", 0.5))
        
        # For backward compatibility
        self.normal_x_speed = self.main_x_speed
        self.normal_y_speed = self.main_y_speed
        
        # Triggerbot parameters
        self.tbfovsize = float(getattr(config, "tbfovsize", 70))
        self.tbdelay = float(getattr(config, "tbdelay", 0.08))
        self.tbcooldown = float(getattr(config, "tbcooldown", 0.5))
        self.last_tb_click_time = 0.0
        
        # FOV colors
        self.fov_color = getattr(config, "fov_color", "white")
        self.fov_smooth_color = getattr(config, "fov_smooth_color", "cyan")
        self.tb_fov_color = getattr(config, "tb_fov_color", "white")

        # Other settings
        self.color = getattr(config, "color", "yellow")
        self.mode = getattr(config, "mode", "Normal")
        self.selected_mouse_button = getattr(config, "selected_mouse_button", 3)
        self.selected_sec_mouse_button = getattr(config, "selected_sec_mouse_button", 4)
        self.selected_tb_btn = getattr(config, "selected_tb_btn", 3)
        self.max_speed = float(getattr(config, "max_speed", 1000.0))
        
        logger.info(f"AimTracker initialized with main speeds: {self.main_x_speed}/{self.main_y_speed}, sec speeds: {self.sec_x_speed}/{self.sec_y_speed}")

        self.controller = Mouse()
        self.move_queue = queue.Queue(maxsize=50)
        self._move_thread = threading.Thread(target=self._process_move_queue, daemon=True)
        self._move_thread.start()

        self.model, self.class_names = load_model()
        print("Classes:", self.class_names)
        self._stop_event = threading.Event()
        self._target_fps = target_fps
        self._track_thread = threading.Thread(target=self._track_loop, daemon=True)
        self._track_thread.start()

    def stop(self):
        self._stop_event.set()
        try:
            self._track_thread.join(timeout=1.0)
        except Exception:
            pass

    def _process_move_queue(self):
        while True:
            try:
                dx, dy, delay = self.move_queue.get(timeout=0.1)
                try:
                    self.controller.move(dx, dy)
                except Exception as e:
                    print("[Mouse.move error]", e)
                if delay and delay > 0:
                    time.sleep(delay)
            except queue.Empty:
                time.sleep(0.001)
                continue
            except Exception as e:
                print(f"[Move Queue Error] {e}")
                time.sleep(0.01)

    def _clip_movement(self, dx, dy):
        clipped_dx = np.clip(dx, -abs(self.max_speed), abs(self.max_speed))
        clipped_dy = np.clip(dy, -abs(self.max_speed), abs(self.max_speed))
        return float(clipped_dx), float(clipped_dy)

    def _track_loop(self):
        period = 1.0 / float(self._target_fps)
        while not self._stop_event.is_set():
            start = time.time()
            try:
                self.track_once()
            except Exception as e:
                print("[Track error]", e)
            elapsed = time.time() - start
            to_sleep = period - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)

    def _draw_fovs(self, img, frame):
        center_x = int(frame.xres / 2)
        center_y = int(frame.yres / 2)
        
        # Get colors from config or use defaults
        fov_color = getattr(config, "fov_color", self.fov_color)
        fov_smooth_color = getattr(config, "fov_smooth_color", self.fov_smooth_color)
        tb_fov_color = getattr(config, "tb_fov_color", self.tb_fov_color)
        
        # Convert color names to BGR
        color_map = {
            "white": (255, 255, 255),
            "red": (0, 0, 255),
            "green": (0, 255, 0),
            "blue": (255, 0, 0),
            "yellow": (0, 255, 255),
            "cyan": (255, 255, 0),
            "magenta": (255, 0, 255),
            "orange": (0, 165, 255),
            "purple": (128, 0, 128)
        }
        
        fov_color_bgr = color_map.get(fov_color, (255, 255, 255))
        fov_smooth_color_bgr = color_map.get(fov_smooth_color, (51, 255, 255))
        tb_fov_color_bgr = color_map.get(tb_fov_color, (255, 255, 255))
        
        if getattr(config, "enableaim", False):
            cv2.circle(img, (center_x, center_y), int(getattr(config, "fovsize", self.fovsize)), fov_color_bgr, 2)
            # Smoothing circle
            cv2.circle(img, (center_x, center_y), int(getattr(config, "normalsmoothfov", self.normalsmoothfov)), fov_smooth_color_bgr, 2)
        if getattr(config, "enabletb", False):
            cv2.circle(img, (center_x, center_y), int(getattr(config, "tbfovsize", self.tbfovsize)), tb_fov_color_bgr, 2)

    def track_once(self):
        if not getattr(self.app, "connected", False):
            return

        try:
            self.app.receiver.frame_sync.capture_video()
        except Exception:
            return

        frame = self.app.video_frame
        if frame is None or getattr(frame, "xres", 0) == 0 or getattr(frame, "yres", 0) == 0:
            return

        try:
            img = np.array(frame, dtype=np.uint8).reshape((frame.yres, frame.xres, 4))
        except Exception:
            return

        bgr_img = img[:, :, [2, 1, 0]].copy()

        try:
            detection_results = perform_detection(self.model, bgr_img)
        except Exception as e:
            print("[perform_detection error]", e)
            detection_results = []

        targets = []
        if detection_results:
            for det in detection_results:
                try:
                    x, y, w, h = det['bbox']
                    conf = det.get('confidence', 1.0)
                    x1, y1 = int(x), int(y)
                    x2, y2 = int(x + w), int(y + h)
                    y1 *= 1.03
                    # Dessin corps
                    self._draw_body(bgr_img, x1, y1, x2, y2, conf)
                    # Estimation têtes dans la bbox
                    head_positions = self._estimate_head_positions(x1, y1, x2, y2, bgr_img)
                    for head_cx, head_cy, bbox in head_positions:
                        self._draw_head_bbox(bgr_img, head_cx, head_cy)
                        d = math.hypot(head_cx - frame.xres / 2.0, head_cy - frame.yres / 2.0)
                        targets.append((head_cx, head_cy, d))
                except Exception:
                    continue

        # FOVs une fois par frame
        try:
            self._draw_fovs(bgr_img, frame)
        except Exception:
            pass

        if targets:
            try:
                self._aim_and_move(targets, frame, bgr_img)
            except Exception as e:
                print("[Aim error]", e)

        try:
            cv2.imshow("Detection", bgr_img)
            cv2.waitKey(1)
        except Exception:
            pass

    def _draw_head_bbox(self, img, headx, heady):
        cv2.circle(img, (int(headx), int(heady)), 2, (0, 0, 255), -1)

    def _estimate_head_positions(self, x1, y1, x2, y2, img):
        offsetY = getattr(config, 'offsetY', 0)
        offsetX = getattr(config, 'offsetX', 0)

        width = x2 - x1
        height = y2 - y1

        # Crop léger
        top_crop_factor = 0.05
        side_crop_factor = 0.05

        effective_y1 = y1 + height * top_crop_factor
        effective_height = height * (1 - top_crop_factor)

        effective_x1 = x1 + width * side_crop_factor
        effective_x2 = x2 - width * side_crop_factor
        effective_width = effective_x2 - effective_x1

        center_x = (effective_x1 + effective_x2) / 2
        headx_base = center_x + effective_width * (offsetX / 100)
        heady_base = effective_y1 + effective_height * (offsetY / 100)

        pixel_marginx = 40
        pixel_marginy = 10

        x1_roi = int(max(headx_base - pixel_marginx, 0))
        y1_roi = int(max(heady_base - pixel_marginy, 0))
        x2_roi = int(min(headx_base + pixel_marginx, img.shape[1]))
        y2_roi = int(min(heady_base + pixel_marginy, img.shape[0]))

        roi = img[y1_roi:y2_roi, x1_roi:x2_roi]
        cv2.rectangle(img, (x1_roi, y1_roi), (x2_roi, y2_roi), (0, 0, 255), 2)

        results = []
        detections = []
        try:
            detections = perform_detection(self.model, roi)
        except Exception as e:
            print("[perform_detection ROI error]", e)

        if not detections:
            # Sans détection → garder le head position avec offset
            results.append((headx_base, heady_base, (x1_roi, y1_roi, x2_roi, y2_roi)))
        else:
            for det in detections:
                x, y, w, h = det["bbox"]
                cv2.rectangle(img, (x1_roi + x, y1_roi + y), (x1_roi + x + w, y1_roi + y + h), (0, 255, 0), 2)

                # Position détection brute
                headx_det = x1_roi + x + w / 2
                heady_det = y1_roi + y + h / 2

                # Application de l’offset aussi sur la détection
                headx_det += effective_width * (offsetX / 100)
                heady_det += effective_height * (offsetY / 100)

                results.append((headx_det, heady_det, (x1_roi + x, y1_roi + y, w, h)))

        return results

    def _draw_body(self, img, x1, y1, x2, y2, conf):
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(img, f"Body {conf:.2f}", (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    def _aim_and_move(self, targets, frame, img):
        aim_enabled = getattr(config, "enableaim", False)
        selected_btn = getattr(config, "selected_mouse_button", None)

        best_target = min(targets, key=lambda t: t[2])
        cx, cy, _ = best_target
        center_x = frame.xres / 2.0
        center_y = frame.yres / 2.0
        distance_to_center = math.hypot(cx - center_x, cy - center_y)
        if distance_to_center > float(getattr(config, 'fovsize', self.fovsize)):
            return

        dx = cx - center_x
        dy = cy - center_y

        sens = float(getattr(config, "in_game_sens", self.in_game_sens))
        dpi = float(getattr(config, "mouse_dpi", self.mouse_dpi))
       

        cm_per_rev_base = 54.54
        cm_per_rev = cm_per_rev_base / max(sens, 0.01)

        count_per_cm = dpi / 2.54
        deg_per_count = 360.0 / (cm_per_rev * count_per_cm)

        ndx = dx * deg_per_count
        ndy = dy * deg_per_count

        mode = getattr(config, "mode", "Normal")
        if mode == "Normal":
            try:
                main_btn = getattr(config, "selected_mouse_button", None)
                sec_btn = getattr(config, "selected_sec_mouse_button", None)
                
                # Check which button is pressed and use appropriate speeds
                main_button_pressed = main_btn is not None and is_button_pressed(main_btn)
                sec_button_pressed = sec_btn is not None and is_button_pressed(sec_btn)
                
                if aim_enabled and (main_button_pressed or sec_button_pressed):
                    # Get the appropriate speeds based on which button is pressed
                    if main_button_pressed:
                        x_speed = float(getattr(config, "main_x_speed", self.main_x_speed))
                        y_speed = float(getattr(config, "main_y_speed", self.main_y_speed))
                        logger.debug(f"Using main button speeds: {x_speed}/{y_speed}")
                    else:
                        x_speed = float(getattr(config, "sec_x_speed", self.sec_x_speed))
                        y_speed = float(getattr(config, "sec_y_speed", self.sec_y_speed))
                        logger.debug(f"Using secondary button speeds: {x_speed}/{y_speed}")
                    
                    # Apply smoothing if within smoothing FOV
                    if distance_to_center < float(getattr(config, "normalsmoothfov", self.normalsmoothfov)):
                        ndx *= x_speed / max(float(getattr(config, "normalsmooth", self.normalsmooth)), 0.01)
                        ndy *= y_speed / max(float(getattr(config, "normalsmooth", self.normalsmooth)), 0.01)
                    else:
                        ndx *= x_speed
                        ndy *= y_speed
                    
                    ddx, ddy = self._clip_movement(ndx, ndy)
                    self.move_queue.put((ddx, ddy, 0.005))
            except Exception as e:
                logger.error(f"Error in aim movement: {e}", exc_info=True)

            # Triggerbot
            try:
                if getattr(config, "enabletb", False) and is_button_pressed(getattr(config, "selected_tb_btn", None)) or is_button_pressed(getattr(config, "selected_2_tb", None)):
                    cx0, cy0 = int(frame.xres // 2), int(frame.yres // 2)
                    r = int(getattr(config, "tbfovsize", self.tbfovsize))
                    x1, y1 = max(cx0 - r, 0), max(cy0 - r, 0)
                    x2, y2 = min(cx0 + r, frame.xres), min(cy0 + r, frame.yres)
                    roi = img[y1:y2, x1:x2]

                    debug_roi = roi.copy()
                    detections = perform_detection(self.model, debug_roi)
                    if detections:
                        for det in detections:
                            x, y, w, h = det["bbox"]
                            cv2.rectangle(debug_roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    try:
                        cv2.imshow("Triggerbot ROI", debug_roi)
                        cv2.waitKey(1)
                    except Exception:
                        pass

                    if detections or distance_to_center < float(getattr(config, "tbfovsize", self.tbfovsize)):
                        now = time.time()
                        tb_delay = float(getattr(config, "tbdelay", self.tbdelay))
                        tb_cooldown = float(getattr(config, "tbcooldown", self.tbcooldown))
                        
                        if now - self.last_tb_click_time >= tb_delay:
                            self.controller.click()
                            self.last_tb_click_time = now + tb_cooldown
            except Exception as e:
                print("[Triggerbot error]", e)

        elif mode == "Silent":
            dx_raw = int(dx)
            dy_raw = int(dy)
            dx_raw *= self.normal_x_speed
            dy_raw *= self.normal_y_speed
            threading.Thread(target=threaded_silent_move, args=(self.controller, dx_raw, dy_raw), daemon=True).start()


class ViewerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("CUPSY COLORBOT")
        self.geometry("400x700")

        # Language support
        self.lang_manager = LanguageManager()
        # Set initial language from config
        if hasattr(config, "language") and config.language in self.lang_manager.languages:
            self.lang_manager.set_language(config.language)
            logger.info(f"Initialized language to {config.language}")
        
        # Dicos pour MAJ UI <-> config
        self._slider_widgets = {}   # key -> {"slider": widget, "label": widget, "min":..., "max":..., "entry": widget}
        self._checkbox_vars = {}    # key -> tk.BooleanVar
        self._option_widgets = {}   # key -> CTkOptionMenu

        # NDI
        self.finder = Finder()
        self.finder.set_change_callback(self._on_finder_change)
        self.finder.open()

        self.receiver = Receiver(color_format=RecvColorFormat.RGBX_RGBA, bandwidth=RecvBandwidth.highest)
        self.video_frame = VideoFrameSync()
        self.audio_frame = AudioFrameSync()
        self.receiver.frame_sync.set_video_frame(self.video_frame)
        self.receiver.frame_sync.set_audio_frame(self.audio_frame)

        self.connected = False
        self.ndi_sources = []
        self.selected_source = None
        self.source_queue = queue.Queue()
        self.after(100, self._process_source_updates)
        # enlève la barre native
       

        # barre custom
        self.title_bar = ctk.CTkFrame(self, height=30, corner_radius=0)
        self.title_bar.pack(fill="x", side="top")

        self.title_label = ctk.CTkLabel(self.title_bar, text="CUPSY CB", anchor="w")
        self.title_label.pack(side="left", padx=10)

        # bouton fermer
        self.close_btn = ctk.CTkButton(self.title_bar, text="X", width=25, command=self.destroy)
        self.close_btn.pack(side="right", padx=2)

        # rendre la barre draggable
        self.title_bar.bind("<Button-1>", self.start_move)
        self.title_bar.bind("<B1-Motion>", self.do_move)
        
        # Tracker
        self.tracker = AimTracker(app=self, target_fps=80)

        # TabView
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(expand=True, fill="both", padx=20, pady=20)
        self.tab_general = self.tabview.add("⚙️ Général")
        self.tab_aimbot = self.tabview.add("🎯 Aimbot")
        self.tab_tb = self.tabview.add("🔫 Triggerbot")
        self.tab_config = self.tabview.add("💾 Config")

        self._build_general_tab()
        self._build_aimbot_tab()
        self._build_tb_tab()
        self._build_config_tab()

        
        # Status polling
        self.after(500, self._update_connection_status_loop)
        self._load_initial_config()

    # ---------- Helpers de mapping UI ----------
    def _register_slider(self, key, slider, label, vmin, vmax, is_float, entry=None):
        self._slider_widgets[key] = {
            "slider": slider, 
            "label": label, 
            "min": vmin, 
            "max": vmax, 
            "is_float": is_float, 
            "entry": entry,
            "entry_var": getattr(self, "_last_entry_var", None),
            "trace_callback": getattr(self, "_last_trace_callback", None)
        }

    def _load_initial_config(self):
        try:
            import json, os
            from detection import reload_model
            if os.path.exists("configs/default.json"):
                with open("configs/default.json", "r") as f:
                    data = json.load(f)

                self._apply_settings(data)


            else:
                print("doesn't exist")
        except Exception as e:
            print("Impossible de charger la config initiale:", e)




    def _set_slider_value(self, key, value):
        if key not in self._slider_widgets:
            return
        w = self._slider_widgets[key]
        vmin, vmax = w["min"], w["max"]
        is_float = w["is_float"]
        # Clamp
        try:
            v = float(value) if is_float else int(round(float(value)))
        except Exception:
            return
        v = max(vmin, min(v, vmax))
        w["slider"].set(v)
        # Rafraîchir label
        txt = f"{key.replace('_',' ').title()}: {v:.2f}" if is_float else f"{key.replace('_',' ').title()}: {int(v)}"
        # On garde le libellé humain (X Speed etc.) si déjà présent
        current = w["label"].cget("text")
        prefix = current.split(":")[0] if ":" in current else txt.split(":")[0]
        w["label"].configure(text=f"{prefix}: {v:.2f}" if is_float else f"{prefix}: {int(v)}")
        
        # Update entry if available - don't create a new StringVar, just update the value
        if w.get("entry") is not None and w.get("entry_var") is not None:
            # Temporarily remove trace to avoid recursive callbacks
            entry_var = w["entry_var"]
            traces = entry_var.trace_info()
            for trace_type, trace_id in traces:
                if trace_type == "write":
                    entry_var.trace_remove("write", trace_id)
                    
            # Update the value
            entry_var.set(f"{v:.2f}" if is_float else str(int(v)))
            
            # Re-add the trace
            entry_var.trace_add("write", w["trace_callback"])

    def _set_checkbox_value(self, key, value_bool):
        var = self._checkbox_vars.get(key)
        if var is not None:
            var.set(bool(value_bool))

    def _set_option_value(self, key, value_str):
        menu = self._option_widgets.get(key)
        if menu is not None and value_str is not None:
            menu.set(str(value_str))

    # -------------- Tab Config --------------
    def _build_config_tab(self):
        os.makedirs("configs", exist_ok=True)

        ctk.CTkLabel(self.tab_config, text="Choose a config:").pack(pady=5, anchor="w")

        self.config_option = ctk.CTkOptionMenu(self.tab_config, values=[], command=self._on_config_selected)
        self.config_option.pack(pady=5, fill="x")

        ctk.CTkButton(self.tab_config, text="💾 Save", command=self._save_config).pack(pady=10, fill="x")
        ctk.CTkButton(self.tab_config, text="💾 New Config", command=self._save_new_config).pack(pady=5, fill="x")
        ctk.CTkButton(self.tab_config, text="📂 Load config", command=self._load_selected_config).pack(pady=5, fill="x")


        self.config_log = ctk.CTkTextbox(self.tab_config, height=120)
        self.config_log.pack(pady=10, fill="both", expand=True)

        self._refresh_config_list()

    def start_move(self, event):
        self._x = event.x
        self._y = event.y

    def do_move(self, event):
        x = self.winfo_pointerx() - self._x
        y = self.winfo_pointery() - self._y
        self.geometry(f"+{x}+{y}")

    def _get_current_settings(self):
        return {
            # Backward compatibility
            "normal_x_speed": getattr(config, "main_x_speed", 0.5),
            "normal_y_speed": getattr(config, "main_y_speed", 0.5),
            
            # Main aimbot button parameters
            "main_x_speed": getattr(config, "main_x_speed", 0.5),
            "main_y_speed": getattr(config, "main_y_speed", 0.5),
            
            # Secondary aimbot button parameters
            "sec_x_speed": getattr(config, "sec_x_speed", 0.5),
            "sec_y_speed": getattr(config, "sec_y_speed", 0.5),
            
            # Common parameters
            "normalsmooth": getattr(config, "normalsmooth", 10),
            "normalsmoothfov": getattr(config, "normalsmoothfov", 10),
            "mouse_dpi" : getattr(config, "mouse_dpi", 800),
            "fovsize": getattr(config, "fovsize", 300),
            
            # FOV colors
            "fov_color": getattr(config, "fov_color", "white"),
            "fov_smooth_color": getattr(config, "fov_smooth_color", "cyan"),
            "tb_fov_color": getattr(config, "tb_fov_color", "white"),
            
            # Triggerbot parameters
            "tbfovsize": getattr(config, "tbfovsize", 70),
            "tbdelay": getattr(config, "tbdelay", 0.08),
            "tbcooldown": getattr(config, "tbcooldown", 0.5),
            
            # Other settings
            "in_game_sens": getattr(config, "in_game_sens", 7),
            "color": getattr(config, "color", "yellow"),
            "mode": getattr(config, "mode", "Normal"),
            "enableaim": getattr(config, "enableaim", False),
            "enabletb": getattr(config, "enabletb", False),
            "selected_mouse_button": getattr(config, "selected_mouse_button", 3),
            "selected_sec_mouse_button": getattr(config, "selected_sec_mouse_button", 4),
            "selected_tb_btn": getattr(config, "selected_tb_btn", 3),
            "language": self.lang_manager.current_language
        }

    def _apply_settings(self, data, config_name=None):
        """
        Applique un dictionnaire de settings sur le config global, le tracker et l'UI.
        Recharge le modèle si nécessaire.
        """
        try:
            # --- Appliquer sur config global ---
            for k, v in data.items():
                setattr(config, k, v)

            # --- Appliquer sur le tracker si l'attribut existe ---
            for k, v in data.items():
                if hasattr(self.tracker, k):
                    setattr(self.tracker, k, v)

            # --- Mettre à jour les sliders ---
            for k, v in data.items():
                if k in self._slider_widgets:
                    self._set_slider_value(k, v)

            # --- Mettre à jour les checkbox ---
            for k, v in data.items():
                if k in self._checkbox_vars:
                    self._set_checkbox_value(k, v)

            # --- Mettre à jour les OptionMenu ---
            for k, v in data.items():
                if k in self._option_widgets:
                    self._set_option_value(k, v)

            # --- Recharger le modèle si nécessaire ---
            from detection import reload_model
            self.tracker.model, self.tracker.class_names = reload_model()

            if config_name:
                self._log_config(f"Config '{config_name}' applied and model reloaded ✅")
            else:
                self._log_config(f"Config applied and model reloaded ✅")

        except Exception as e:
            self._log_config(f"[Erreur _apply_settings] {e}")


    def _save_new_config(self):
        from tkinter import simpledialog
        name = simpledialog.askstring("Config name", "Enter the config name:")
        if not name:
            self._log_config("Cancelled save (pas de nom fourni).")
            return
        data = self._get_current_settings()
        path = os.path.join("configs", f"{name}.json")
        try:
            os.makedirs("configs", exist_ok=True)
            with open(path, "w") as f:
                json.dump(data, f, indent=4)
            self._refresh_config_list()
            self.config_option.set(name)  # Sélectionner automatiquement
            self._log_config(f"New config'{name}' saved ✅")
        except Exception as e:
            self._log_config(f"[Erreur SAVE] {e}")



    def _load_selected_config(self):
        """
        Charge la config sélectionnée dans l'OptionMenu.
        """
        name = self.config_option.get()
        path = os.path.join("configs", f"{name}.json")
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self._apply_settings(data, config_name=name)
            self._log_config(f"Config '{name}' loaded 📂")
        except Exception as e:
            self._log_config(f"[Erreur LOAD] {e}")




    def _refresh_config_list(self):
        files = [f[:-5] for f in os.listdir("configs") if f.endswith(".json")]
        if not files:
            files = ["default"]
        current = self.config_option.get()
        self.config_option.configure(values=files)
        if current in files:
            self.config_option.set(current)
        else:
            self.config_option.set(files[0])


    def _on_config_selected(self, val):
        self._log_config(f"Selected config: {val}")

    def _save_config(self):
        name = self.config_option.get() or "default"
        data = self._get_current_settings()
        path = os.path.join("configs", f"{name}.json")
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=4)
            self._log_config(f"Config '{name}' sauvegardée ✅")
            self._refresh_config_list()
        except Exception as e:
            self._log_config(f"[Erreur SAVE] {e}")

    def _load_config(self):
        name = self.config_option.get() or "default"
        path = os.path.join("configs", f"{name}.json")
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self._apply_settings(data)
            self._log_config(f"Config '{name}' loaded 📂")
        except Exception as e:
            self._log_config(f"[Erreur LOAD] {e}")

    def _log_config(self, msg):
        self.config_log.insert("end", msg + "\n")
        self.config_log.see("end")

    # ----------------------- UI BUILDERS -----------------------
    def _build_general_tab(self):
        self.status_label = ctk.CTkLabel(self.tab_general, text=f"{self.lang_manager.get_text('general', 'status', 'Status')}: {self.lang_manager.get_text('general', 'disconnected', 'Disconnected')}")
        self.status_label.pack(pady=5, anchor="w")

        self.source_option = ctk.CTkOptionMenu(self.tab_general, values=["(searching...)"], command=self._on_source_selected)
        self.source_option.pack(pady=5, fill="x")

        ctk.CTkButton(self.tab_general, text=self.lang_manager.get_text('general', 'refresh_sources', "Refresh NDI Sources"), command=self._refresh_sources).pack(pady=5, fill="x")
        ctk.CTkButton(self.tab_general, text=self.lang_manager.get_text('general', 'connect_source', "Connect to Source"), command=self._connect_to_selected).pack(pady=5, fill="x")
        
        # MAKCU Connection Status
        self.makcu_status_frame = ctk.CTkFrame(self.tab_general)
        self.makcu_status_frame.pack(pady=5, fill="x")
        ctk.CTkLabel(self.makcu_status_frame, text=self.lang_manager.get_text('general', 'makcu_status', "MAKCU Status")).pack(side="left", padx=5)
        self.makcu_status_indicator = ctk.CTkLabel(self.makcu_status_frame, text="●", text_color="red")
        self.makcu_status_indicator.pack(side="right", padx=5)
        
        # Move and Click Test buttons
        test_frame = ctk.CTkFrame(self.tab_general)
        test_frame.pack(pady=5, fill="x")
        ctk.CTkButton(test_frame, text=self.lang_manager.get_text('general', 'move_test', "Move Test"), 
                     command=self._test_move, width=100).pack(side="left", padx=5, pady=5, expand=True, fill="x")
        ctk.CTkButton(test_frame, text=self.lang_manager.get_text('general', 'click_test', "Click Test"), 
                     command=self._test_click, width=100).pack(side="right", padx=5, pady=5, expand=True, fill="x")

        ctk.CTkLabel(self.tab_general, text=self.lang_manager.get_text('general', 'appearance', "Appearance")).pack(pady=5)
        ctk.CTkOptionMenu(self.tab_general, values=["Dark", "Light"], command=self._on_appearance_selected).pack(pady=5, fill="x")

        # Language selector
        ctk.CTkLabel(self.tab_general, text=self.lang_manager.get_text('general', 'language', "Language")).pack(pady=5)
        self.language_option = ctk.CTkOptionMenu(
            self.tab_general, 
            values=self.lang_manager.get_language_names(),
            command=self._on_language_selected
        )
        self.language_option.set(self.lang_manager.languages[self.lang_manager.current_language].get("language_name", "English"))
        self.language_option.pack(pady=5, fill="x")

        ctk.CTkLabel(self.tab_general, text=self.lang_manager.get_text('general', 'mode', "Mode")).pack(pady=5)
        self.mode_option = ctk.CTkOptionMenu(self.tab_general, values=["Normal"], command=self._on_mode_selected)
        self.mode_option.pack(pady=5, fill="x")
        self._option_widgets["mode"] = self.mode_option

        ctk.CTkLabel(self.tab_general, text=self.lang_manager.get_text('general', 'color', "Color")).pack(pady=5)
        self.color_option = ctk.CTkOptionMenu(self.tab_general, values=["yellow", "purple"], command=self._on_color_selected)
        self.color_option.pack(pady=5, fill="x")
        self._option_widgets["color"] = self.color_option

    def _build_aimbot_tab(self):
        # Create a scrollable frame inside the tab
        self.aimbot_scrollable_frame = ctk.CTkScrollableFrame(self.tab_aimbot)
        self.aimbot_scrollable_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Common settings
        ctk.CTkLabel(self.aimbot_scrollable_frame, text="--- Common Settings ---", font=("Arial", 14, "bold")).pack(pady=10)
        
        # In-game sens
        s, l, e = self._add_slider_with_label(self.aimbot_scrollable_frame, self.lang_manager.get_text('aimbot', 'in_game_sens', "In-game sens"), 0.1, 2000, float(getattr(config, "in_game_sens", 7)), self._on_config_in_game_sens_changed, is_float=True)
        self._register_slider("in_game_sens", s, l, 0.1, 2000, True, e)
        
        # Smoothing
        s, l, e = self._add_slider_with_label(self.aimbot_scrollable_frame, self.lang_manager.get_text('aimbot', 'smoothing', "Smoothing"), 1, 30, float(getattr(config, "normalsmooth", 10)), self._on_config_normal_smooth_changed, is_float=True)
        self._register_slider("normalsmooth", s, l, 1, 30, True, e)
        
        # Smoothing FOV
        s, l, e = self._add_slider_with_label(self.aimbot_scrollable_frame, self.lang_manager.get_text('aimbot', 'smoothing_fov', "Smoothing FOV"), 1, 30, float(getattr(config, "normalsmoothfov", 10)), self._on_config_normal_smoothfov_changed, is_float=True)
        self._register_slider("normalsmoothfov", s, l, 1, 30, True, e)
        
        # FOV Size
        s, l, e = self._add_slider_with_label(self.aimbot_scrollable_frame, self.lang_manager.get_text('aimbot', 'fov_size', "FOV Size"), 1, 1000, float(getattr(config, "fovsize", 300)), self._on_fovsize_changed, is_float=True)
        self._register_slider("fovsize", s, l, 1, 1000, True, e)
        
        # FOV Color
        ctk.CTkLabel(self.aimbot_scrollable_frame, text=self.lang_manager.get_text('aimbot', 'fov_color', "FOV Circle Color")).pack(pady=5, anchor="w")
        self.fov_color_option = ctk.CTkOptionMenu(
            self.aimbot_scrollable_frame,
            values=["white", "red", "green", "blue", "yellow", "cyan", "magenta", "orange", "purple"],
            command=self._on_fov_color_selected
        )
        self.fov_color_option.set(getattr(config, "fov_color", "white"))
        self.fov_color_option.pack(pady=5, fill="x")
        self._option_widgets["fov_color"] = self.fov_color_option
        
        # FOV Smooth Color
        ctk.CTkLabel(self.aimbot_scrollable_frame, text="FOV Smooth Circle Color").pack(pady=5, anchor="w")
        self.fov_smooth_color_option = ctk.CTkOptionMenu(
            self.aimbot_scrollable_frame,
            values=["white", "red", "green", "blue", "yellow", "cyan", "magenta", "orange", "purple"],
            command=self._on_fov_smooth_color_selected
        )
        self.fov_smooth_color_option.set(getattr(config, "fov_smooth_color", "cyan"))
        self.fov_smooth_color_option.pack(pady=5, fill="x")
        self._option_widgets["fov_smooth_color"] = self.fov_smooth_color_option

        # Enable Aim
        self.var_enableaim = tk.BooleanVar(value=getattr(config, "enableaim", False))
        ctk.CTkCheckBox(self.aimbot_scrollable_frame, text=self.lang_manager.get_text('aimbot', 'enable_aim', "Enable Aim"), variable=self.var_enableaim, command=self._on_enableaim_changed).pack(pady=6, anchor="w")
        self._checkbox_vars["enableaim"] = self.var_enableaim

        # Main Aimbot Button section
        ctk.CTkLabel(self.aimbot_scrollable_frame, text="--- Main Aimbot Button ---", font=("Arial", 14, "bold")).pack(pady=10)
        
        # Main Aimbot Button
        ctk.CTkLabel(self.aimbot_scrollable_frame, text=self.lang_manager.get_text('aimbot', 'aimbot_button', "Main Aimbot Button")).pack(pady=5, anchor="w")
        self.aimbot_button_option = ctk.CTkOptionMenu(
            self.aimbot_scrollable_frame,
            values=list(BUTTONS.values()),
            command=self._on_aimbot_button_selected
        )
        self.aimbot_button_option.pack(pady=5, fill="x")
        self._option_widgets["aimbot_button"] = self.aimbot_button_option
        
        # Main X Speed
        s, l, e = self._add_slider_with_label(self.aimbot_scrollable_frame, self.lang_manager.get_text('aimbot', 'main_x_speed', "Main X Speed"), 0.1, 2000, float(getattr(config, "main_x_speed", 0.5)), self._on_main_x_speed_changed, is_float=True)
        self._register_slider("main_x_speed", s, l, 0.1, 2000, True, e)
        
        # Main Y Speed
        s, l, e = self._add_slider_with_label(self.aimbot_scrollable_frame, self.lang_manager.get_text('aimbot', 'main_y_speed', "Main Y Speed"), 0.1, 2000, float(getattr(config, "main_y_speed", 0.5)), self._on_main_y_speed_changed, is_float=True)
        self._register_slider("main_y_speed", s, l, 0.1, 2000, True, e)
        
        # Secondary Aimbot Button section
        ctk.CTkLabel(self.aimbot_scrollable_frame, text="--- Secondary Aimbot Button ---", font=("Arial", 14, "bold")).pack(pady=10)
        
        # Secondary Aimbot Button
        ctk.CTkLabel(self.aimbot_scrollable_frame, text=self.lang_manager.get_text('aimbot', 'sec_aimbot_button', "Secondary Aimbot Button")).pack(pady=5, anchor="w")
        self.sec_aimbot_button_option = ctk.CTkOptionMenu(
            self.aimbot_scrollable_frame,
            values=list(BUTTONS.values()),
            command=self._on_sec_aimbot_button_selected
        )
        self.sec_aimbot_button_option.pack(pady=5, fill="x")
        self._option_widgets["sec_aimbot_button"] = self.sec_aimbot_button_option
        
        # Secondary X Speed
        s, l, e = self._add_slider_with_label(self.aimbot_scrollable_frame, self.lang_manager.get_text('aimbot', 'sec_x_speed', "Secondary X Speed"), 0.1, 2000, float(getattr(config, "sec_x_speed", 0.5)), self._on_sec_x_speed_changed, is_float=True)
        self._register_slider("sec_x_speed", s, l, 0.1, 2000, True, e)
        
        # Secondary Y Speed
        s, l, e = self._add_slider_with_label(self.aimbot_scrollable_frame, self.lang_manager.get_text('aimbot', 'sec_y_speed', "Secondary Y Speed"), 0.1, 2000, float(getattr(config, "sec_y_speed", 0.5)), self._on_sec_y_speed_changed, is_float=True)
        self._register_slider("sec_y_speed", s, l, 0.1, 2000, True, e)


    def _build_tb_tab(self):
        # Create a scrollable frame inside the tab
        self.tb_scrollable_frame = ctk.CTkScrollableFrame(self.tab_tb)
        self.tb_scrollable_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # TB FOV Size
        s, l, e = self._add_slider_with_label(self.tb_scrollable_frame, self.lang_manager.get_text('triggerbot', 'tb_fov_size', "TB FOV Size"), 1, 300, float(getattr(config, "tbfovsize", 70)), self._on_tbfovsize_changed, is_float=True)
        self._register_slider("tbfovsize", s, l, 1, 300, True, e)
        # TB Delay
        s, l, e = self._add_slider_with_label(self.tb_scrollable_frame, self.lang_manager.get_text('triggerbot', 'tb_delay', "TB Delay"), 0.0, 1.0, float(getattr(config, "tbdelay", 0.08)), self._on_tbdelay_changed, is_float=True)
        self._register_slider("tbdelay", s, l, 0.0, 1.0, True, e)
        # TB Cooldown
        s, l, e = self._add_slider_with_label(self.tb_scrollable_frame, self.lang_manager.get_text('triggerbot', 'tb_cooldown', "TB Cooldown"), 0.0, 5.0, float(getattr(config, "tbcooldown", 0.5)), self._on_tbcooldown_changed, is_float=True)
        self._register_slider("tbcooldown", s, l, 0.0, 5.0, True, e)

        # TB FOV Color
        ctk.CTkLabel(self.tb_scrollable_frame, text="TB FOV Circle Color").pack(pady=5, anchor="w")
        self.tb_fov_color_option = ctk.CTkOptionMenu(
            self.tb_scrollable_frame,
            values=["white", "red", "green", "blue", "yellow", "cyan", "magenta", "orange", "purple"],
            command=self._on_tb_fov_color_selected
        )
        self.tb_fov_color_option.set(getattr(config, "tb_fov_color", "white"))
        self.tb_fov_color_option.pack(pady=5, fill="x")
        self._option_widgets["tb_fov_color"] = self.tb_fov_color_option

        # Enable TB
        self.var_enabletb = tk.BooleanVar(value=getattr(config, "enabletb", False))
        ctk.CTkCheckBox(self.tb_scrollable_frame, text=self.lang_manager.get_text('triggerbot', 'enable_tb', "Enable TB"), variable=self.var_enabletb, command=self._on_enabletb_changed).pack(pady=6, anchor="w")
        self._checkbox_vars["enabletb"] = self.var_enabletb

        ctk.CTkLabel(self.tb_scrollable_frame, text=self.lang_manager.get_text('triggerbot', 'tb_button', "Triggerbot Button")).pack(pady=5, anchor="w")
        self.tb_button_option = ctk.CTkOptionMenu(
            self.tb_scrollable_frame,
            values=list(BUTTONS.values()),
            command=self._on_tb_button_selected
        )
        self.tb_button_option.pack(pady=5, fill="x")
        self._option_widgets["tb_button"] = self.tb_button_option


    # Generic slider helper (parent-aware)
    def _add_slider_with_label(self, parent, text, min_val, max_val, init_val, command, is_float=False):
        outer_frame = ctk.CTkFrame(parent)
        outer_frame.pack(padx=12, pady=6, fill="x")
        
        # Top row with label and entry
        top_frame = ctk.CTkFrame(outer_frame)
        top_frame.pack(fill="x")
        
        label = ctk.CTkLabel(top_frame, text=f"{text}: {init_val:.2f}" if is_float else f"{text}: {init_val}")
        label.pack(side="left")
        
        # Create entry for direct value input
        entry_var = tk.StringVar(value=f"{init_val:.2f}" if is_float else str(init_val))
        entry = ctk.CTkEntry(top_frame, width=60, textvariable=entry_var)
        entry.pack(side="right", padx=5)
        
        # Bottom row with slider
        slider_frame = ctk.CTkFrame(outer_frame)
        slider_frame.pack(fill="x", pady=(5, 0))
        
        steps = 100 if is_float else max(1, int(max_val - min_val))
        slider = ctk.CTkSlider(slider_frame, from_=min_val, to=max_val, number_of_steps=steps,
                              command=lambda v: self._slider_callback(v, label, text, command, is_float, entry_var))
        slider.set(init_val)
        slider.pack(fill="x", expand=True)
        
        # Entry callback
        def on_entry_change(var_name, index, mode):
            try:
                value = float(entry_var.get()) if is_float else int(float(entry_var.get()))
                value = max(min_val, min(max_val, value))
                # Temporarily remove slider callback to avoid recursive callbacks
                slider_command = slider.cget("command")
                slider.configure(command=None)
                slider.set(value)
                slider.configure(command=slider_command)
                command(value)
                label.configure(text=f"{text}: {value:.2f}" if is_float else f"{text}: {value}")
            except ValueError:
                pass  # Ignore invalid inputs
        
        # Add trace and store the callback function for later use
        trace_id = entry_var.trace_add("write", on_entry_change)
        
        # Return the widgets and also store the entry_var and trace_callback
        result = (slider, label, entry)
        # Store additional info for later use in _set_slider_value
        self._last_entry_var = entry_var
        self._last_trace_callback = on_entry_change
        
        return result

    def _slider_callback(self, value, label, text, command, is_float, entry_var=None):
        val = float(value) if is_float else int(round(value))
        label.configure(text=f"{text}: {val:.2f}" if is_float else f"{text}: {val}")
        
        if entry_var is not None:
            # Temporarily remove trace to avoid recursive callbacks
            traces = entry_var.trace_info()
            trace_ids = []
            for trace_type, trace_id in traces:
                if trace_type == "write":
                    entry_var.trace_remove("write", trace_id)
                    trace_ids.append((trace_type, trace_id))
            
            # Update entry value
            entry_var.set(f"{val:.2f}" if is_float else str(val))
            
            # Re-add traces
            for trace_type, trace_id in trace_ids:
                # Find the callback in our slider widgets
                for widget_dict in self._slider_widgets.values():
                    if widget_dict.get("entry_var") == entry_var and widget_dict.get("trace_callback"):
                        entry_var.trace_add("write", widget_dict["trace_callback"])
                        break
        
        command(val)

    # ----------------------- Callbacks -----------------------
    def _on_main_x_speed_changed(self, val):
        config.main_x_speed = val
        self.tracker.main_x_speed = val
        # For backward compatibility
        config.normal_x_speed = val
        self.tracker.normal_x_speed = val
        logger.debug(f"Main X Speed changed to {val}")

    def _on_main_y_speed_changed(self, val):
        config.main_y_speed = val
        self.tracker.main_y_speed = val
        # For backward compatibility
        config.normal_y_speed = val
        self.tracker.normal_y_speed = val
        logger.debug(f"Main Y Speed changed to {val}")
        
    def _on_sec_x_speed_changed(self, val):
        config.sec_x_speed = val
        self.tracker.sec_x_speed = val
        logger.debug(f"Secondary X Speed changed to {val}")
        
    def _on_sec_y_speed_changed(self, val):
        config.sec_y_speed = val
        self.tracker.sec_y_speed = val
        logger.debug(f"Secondary Y Speed changed to {val}")
        
    # For backward compatibility
    def _on_normal_x_speed_changed(self, val):
        self._on_main_x_speed_changed(val)

    def _on_normal_y_speed_changed(self, val):
        self._on_main_y_speed_changed(val)

    def _on_config_in_game_sens_changed(self, val):
        config.in_game_sens = val
        self.tracker.in_game_sens = val

    def _on_config_normal_smooth_changed(self, val):
        config.normalsmooth = val
        self.tracker.normalsmooth = val

    def _on_config_normal_smoothfov_changed(self, val):
        config.normalsmoothfov = val
        self.tracker.normalsmoothfov = val

    def _on_aimbot_button_selected(self, val):
        for key, name in BUTTONS.items():
            if name == val:
                config.selected_mouse_button = key
                break
        self._log_config(f"Aimbot button set to {val} ({key})")

    def _on_tb_button_selected(self, val):
        for key, name in BUTTONS.items():
            if name == val:
                config.selected_tb_btn = key
                break
        self._log_config(f"Triggerbot button set to {val} ({key})")
    
    def _on_sec_aimbot_button_selected(self, val):
        for key, name in BUTTONS.items():
            if name == val:
                config.selected_sec_mouse_button = key
                break
        self._log_config(f"Secondary aimbot button set to {val} ({key})")

    def _on_fovsize_changed(self, val):
        config.fovsize = val
        self.tracker.fovsize = val

    def _on_tbdelay_changed(self, val):
        config.tbdelay = val
        self.tracker.tbdelay = val
    
    def _on_tbcooldown_changed(self, val):
        config.tbcooldown = val
        if not hasattr(self.tracker, "tbcooldown"):
            self.tracker.tbcooldown = val
        else:
            self.tracker.tbcooldown = val

    def _on_tbfovsize_changed(self, val):
        config.tbfovsize = val
        self.tracker.tbfovsize = val
        
    def _on_language_selected(self, val):
        lang_key = self.lang_manager.get_language_by_name(val)
        self.lang_manager.set_language(lang_key)
        logger.info(f"Language changed to {val} ({lang_key})")
        self._log_config(f"Language changed to {val}")
        
        # Update all UI elements with the new language
        self._update_ui_language()
    
    def _update_ui_language(self):
        """Update all UI elements with the current language"""
        logger.debug("Updating UI with new language")
        
        try:
            # A simpler approach: just update the tab names in the segmented button
            # and keep the existing tab references
            
            # Get current tab
            current_tab = self.tabview.get()
            
            # Create mapping from old tab names to new tab names
            old_to_new_mapping = {}
            for tab_name in self.tabview._tab_dict.keys():
                if "⚙️" in tab_name:  # General tab
                    old_to_new_mapping[tab_name] = "⚙️ " + self.lang_manager.get_text('general', 'general', "Général")
                elif "🎯" in tab_name:  # Aimbot tab
                    old_to_new_mapping[tab_name] = "🎯 " + self.lang_manager.get_text('general', 'aimbot', "Aimbot")
                elif "🔫" in tab_name:  # Triggerbot tab
                    old_to_new_mapping[tab_name] = "🔫 " + self.lang_manager.get_text('general', 'triggerbot', "Triggerbot")
                elif "💾" in tab_name:  # Config tab
                    old_to_new_mapping[tab_name] = "💾 " + self.lang_manager.get_text('general', 'config', "Config")
            
            # Create new tab dictionary with updated names
            new_tab_dict = {}
            for old_name, tab_frame in self.tabview._tab_dict.items():
                if old_name in old_to_new_mapping:
                    new_name = old_to_new_mapping[old_name]
                    new_tab_dict[new_name] = tab_frame
                else:
                    new_tab_dict[old_name] = tab_frame  # Keep unchanged if not in mapping
            
            # Update tab names in segmented button
            new_values = list(new_tab_dict.keys())
            self.tabview._segmented_button.configure(values=new_values)
            
            # Update the tab dictionary
            self.tabview._tab_dict = new_tab_dict
            
            # Update tab references
            for tab_name, tab_frame in new_tab_dict.items():
                if "⚙️" in tab_name:  # General tab
                    self.tab_general = tab_frame
                elif "🎯" in tab_name:  # Aimbot tab
                    self.tab_aimbot = tab_frame
                elif "🔫" in tab_name:  # Triggerbot tab
                    self.tab_tb = tab_frame
                elif "💾" in tab_name:  # Config tab
                    self.tab_config = tab_frame
            
            # Try to restore the previously selected tab
            # Find the new name for the current tab
            new_current_tab = None
            for old_name, new_name in old_to_new_mapping.items():
                if old_name == current_tab:
                    new_current_tab = new_name
                    break
            
            # Set the current tab
            if new_current_tab:
                self.tabview.set(new_current_tab)
            else:
                # Default to first tab if we can't find a match
                self.tabview.set(new_values[0])
                
            # Now update the text of all widgets
            # Update General tab
            self.status_label.configure(text=f"{self.lang_manager.get_text('general', 'status', 'Status')}: " +
                                        (f"{self.lang_manager.get_text('general', 'connected', 'Connected')} - {self.selected_source}" 
                                         if self.connected else 
                                         self.lang_manager.get_text('general', 'disconnected', 'Disconnected')))
            
            # Update buttons in General tab
            for widget in self.tab_general.winfo_children():
                if isinstance(widget, ctk.CTkButton):
                    if "Refresh" in widget.cget("text"):
                        widget.configure(text=self.lang_manager.get_text('general', 'refresh_sources', "Refresh NDI Sources"))
                    elif "Connect" in widget.cget("text"):
                        widget.configure(text=self.lang_manager.get_text('general', 'connect_source', "Connect to Source"))
            
            # Update labels in General tab
            for widget in self.tab_general.winfo_children():
                if isinstance(widget, ctk.CTkLabel) and not widget == self.status_label:
                    if "Appearance" in widget.cget("text"):
                        widget.configure(text=self.lang_manager.get_text('general', 'appearance', "Appearance"))
                    elif "Language" in widget.cget("text"):
                        widget.configure(text=self.lang_manager.get_text('general', 'language', "Language"))
                    elif "Mode" in widget.cget("text"):
                        widget.configure(text=self.lang_manager.get_text('general', 'mode', "Mode"))
                    elif "Color" in widget.cget("text"):
                        widget.configure(text=self.lang_manager.get_text('general', 'color', "Color"))
                elif isinstance(widget, ctk.CTkFrame):
                    for child in widget.winfo_children():
                        if isinstance(child, ctk.CTkLabel) and "MAKCU" in child.cget("text"):
                            child.configure(text=self.lang_manager.get_text('general', 'makcu_status', "MAKCU Status"))
                        elif isinstance(child, ctk.CTkButton):
                            if "Move Test" in child.cget("text"):
                                child.configure(text=self.lang_manager.get_text('general', 'move_test', "Move Test"))
                            elif "Click Test" in child.cget("text"):
                                child.configure(text=self.lang_manager.get_text('general', 'click_test', "Click Test"))
            
            # Update Aimbot tab
            for key, widget_dict in self._slider_widgets.items():
                label = widget_dict["label"]
                current = label.cget("text")
                if ":" in current:
                    prefix = current.split(":")[0].strip()
                    value = current.split(":")[1].strip()
                    
                    if "X Speed" in prefix:
                        new_prefix = self.lang_manager.get_text('aimbot', 'x_speed', "X Speed")
                    elif "Y Speed" in prefix:
                        new_prefix = self.lang_manager.get_text('aimbot', 'y_speed', "Y Speed")
                    elif "In-game sens" in prefix:
                        new_prefix = self.lang_manager.get_text('aimbot', 'in_game_sens', "In-game sens")
                    elif "Smoothing FOV" in prefix:
                        new_prefix = self.lang_manager.get_text('aimbot', 'smoothing_fov', "Smoothing FOV")
                    elif "Smoothing" in prefix:
                        new_prefix = self.lang_manager.get_text('aimbot', 'smoothing', "Smoothing")
                    elif "FOV Size" in prefix:
                        new_prefix = self.lang_manager.get_text('aimbot', 'fov_size', "FOV Size")
                    elif "TB FOV Size" in prefix:
                        new_prefix = self.lang_manager.get_text('triggerbot', 'tb_fov_size', "TB FOV Size")
                    elif "TB Delay" in prefix:
                        new_prefix = self.lang_manager.get_text('triggerbot', 'tb_delay', "TB Delay")
                    elif "TB Cooldown" in prefix:
                        new_prefix = self.lang_manager.get_text('triggerbot', 'tb_cooldown', "TB Cooldown")
                    elif "Main X Speed" in prefix:
                        new_prefix = self.lang_manager.get_text('aimbot', 'main_x_speed', "Main X Speed")
                    elif "Main Y Speed" in prefix:
                        new_prefix = self.lang_manager.get_text('aimbot', 'main_y_speed', "Main Y Speed")
                    elif "Secondary X Speed" in prefix:
                        new_prefix = self.lang_manager.get_text('aimbot', 'sec_x_speed', "Secondary X Speed")
                    elif "Secondary Y Speed" in prefix:
                        new_prefix = self.lang_manager.get_text('aimbot', 'sec_y_speed', "Secondary Y Speed")
                    else:
                        new_prefix = prefix
                        
                    label.configure(text=f"{new_prefix}: {value}")
            
            # Update checkboxes in the scrollable frame
            for widget in self.aimbot_scrollable_frame.winfo_children():
                if isinstance(widget, ctk.CTkCheckBox):
                    if "Enable Aim" in widget.cget("text"):
                        widget.configure(text=self.lang_manager.get_text('aimbot', 'enable_aim', "Enable Aim"))
            
            for widget in self.tb_scrollable_frame.winfo_children():
                if isinstance(widget, ctk.CTkCheckBox):
                    if "Enable TB" in widget.cget("text"):
                        widget.configure(text=self.lang_manager.get_text('triggerbot', 'enable_tb', "Enable TB"))
            
            # Update button labels in the aimbot scrollable frame
            for widget in self.aimbot_scrollable_frame.winfo_children():
                if isinstance(widget, ctk.CTkLabel):
                    if "--- Common Settings ---" in widget.cget("text"):
                        widget.configure(text="--- Common Settings ---")
                    elif "--- Main Aimbot Button ---" in widget.cget("text"):
                        widget.configure(text="--- Main Aimbot Button ---")
                    elif "--- Secondary Aimbot Button ---" in widget.cget("text"):
                        widget.configure(text="--- Secondary Aimbot Button ---")
                    elif "Main Aimbot Button" in widget.cget("text"):
                        widget.configure(text=self.lang_manager.get_text('aimbot', 'aimbot_button', "Main Aimbot Button"))
                    elif "Secondary Aimbot Button" in widget.cget("text"):
                        widget.configure(text=self.lang_manager.get_text('aimbot', 'sec_aimbot_button', "Secondary Aimbot Button"))
                    elif "FOV Circle Color" in widget.cget("text"):
                        widget.configure(text=self.lang_manager.get_text('aimbot', 'fov_color', "FOV Circle Color"))
                    elif "FOV Smooth Circle Color" in widget.cget("text"):
                        widget.configure(text="FOV Smooth Circle Color")
            
            # Update labels in the triggerbot scrollable frame
            for widget in self.tb_scrollable_frame.winfo_children():
                if isinstance(widget, ctk.CTkLabel):
                    if "Triggerbot Button" in widget.cget("text"):
                        widget.configure(text=self.lang_manager.get_text('triggerbot', 'tb_button', "Triggerbot Button"))
                    elif "TB FOV Circle Color" in widget.cget("text"):
                        widget.configure(text="TB FOV Circle Color")
            
            # Update Config tab
            for widget in self.tab_config.winfo_children():
                if isinstance(widget, ctk.CTkLabel):
                    if "Choose" in widget.cget("text"):
                        widget.configure(text=self.lang_manager.get_text('config', 'choose_config', "Choose a config"))
                elif isinstance(widget, ctk.CTkButton):
                    if "Save" in widget.cget("text") and "New" not in widget.cget("text"):
                        widget.configure(text=self.lang_manager.get_text('config', 'save', "Save"))
                    elif "New Config" in widget.cget("text"):
                        widget.configure(text=self.lang_manager.get_text('config', 'new_config', "New Config"))
                    elif "Load" in widget.cget("text"):
                        widget.configure(text=self.lang_manager.get_text('config', 'load_config', "Load config"))
            
            logger.info("UI language update completed")
        except Exception as e:
            logger.error(f"Error updating UI language: {e}", exc_info=True)
        
    def _test_move(self):
        try:
            self.tracker.controller.move(100, 100)
            time.sleep(0.1)
            self.tracker.controller.move(-100, -100)
            self._log_config("Move test executed")
        except Exception as e:
            self._log_config(f"Move test error: {e}")
            
    def _test_click(self):
        try:
            self.tracker.controller.click()
            self._log_config("Click test executed")
        except Exception as e:
            self._log_config(f"Click test error: {e}")

    def _on_enableaim_changed(self):
        config.enableaim = self.var_enableaim.get()

    def _on_enabletb_changed(self):
        config.enabletb = self.var_enabletb.get()

    def _on_source_selected(self, val):
        self.selected_source = val

    def _on_appearance_selected(self, val):
        try:
            ctk.set_appearance_mode(val)
        except Exception:
            pass

    def _on_color_selected(self, val):
        config.color = val
        self.tracker.color = val
        logger.debug(f"Color changed to {val}")

    def _on_mode_selected(self, val):
        config.mode = val
        self.tracker.mode = val
        logger.debug(f"Mode changed to {val}")
        
    def _on_fov_color_selected(self, val):
        config.fov_color = val
        self.tracker.fov_color = val
        logger.debug(f"FOV color changed to {val}")
        
    def _on_fov_smooth_color_selected(self, val):
        config.fov_smooth_color = val
        self.tracker.fov_smooth_color = val
        logger.debug(f"FOV smooth color changed to {val}")
        
    def _on_tb_fov_color_selected(self, val):
        config.tb_fov_color = val
        self.tracker.tb_fov_color = val
        logger.debug(f"TB FOV color changed to {val}")

    # ----------------------- NDI helpers -----------------------
    def _process_source_updates(self):
        while not self.source_queue.empty():
            names = self.source_queue.get()
            self._apply_sources_to_ui(names)
        self.after(100, self._process_source_updates)

    def _refresh_sources(self):
        try:
            names = self.finder.get_source_names() or []
        except Exception:
            names = []
        if names:
            self.ndi_sources = names
            self.source_option.configure(values=names)
            self.source_option.set(names[0])
            self.selected_source = names[0]
            self.status_label.configure(text="Sources refreshed", text_color="green")
        else:
            self.source_option.configure(values=["(no sources)"])
            self.source_option.set("(no sources)")
            self.selected_source = None
            self.status_label.configure(text="No sources found", text_color="orange")

    def _connect_to_selected(self):
        if not self.ndi_sources:
            self.status_label.configure(text="No NDI sources available", text_color="orange")
            return
        if self.selected_source is None:
            self.selected_source = self.ndi_sources[0]
            self.source_option.set(self.selected_source)
        try:
            with self.finder.notify:
                src = self.finder.get_source(self.selected_source)
                self.receiver.set_source(src)
                self.connected = True
                self.status_label.configure(text=f"Connected to {self.selected_source}", text_color="green")
        except Exception as e:
            self.status_label.configure(text=f"Failed to connect: {e}", text_color="red")



    def _update_connection_status_loop(self):
        try:
            # NDI connection status
            is_conn = self.receiver.is_connected()
            self.connected = is_conn
            if is_conn:
                self.status_label.configure(
                    text=f"{self.lang_manager.get_text('general', 'status', 'Status')}: {self.lang_manager.get_text('general', 'connected', 'Connected')} - {self.selected_source}", 
                    text_color="green"
                )
            else:
                self.status_label.configure(
                    text=f"{self.lang_manager.get_text('general', 'status', 'Status')}: {self.lang_manager.get_text('general', 'disconnected', 'Disconnected')}", 
                    text_color="red"
                )
            
            # MAKCU connection status - this is a placeholder, replace with actual MAKCU connection check
            try:
                # Check if @makcu-py-lib is connected
                # For now, we'll just use a placeholder check - replace with actual implementation
                import importlib.util
                makcu_connected = False
                try:
                    # Try to import makcu if available
                    if importlib.util.find_spec("makcu") is not None:
                        makcu = importlib.import_module("makcu")
                        if hasattr(makcu, "is_connected") and callable(makcu.is_connected):
                            makcu_connected = makcu.is_connected()
                    
                    # If not available, check if mouse controller is working as a fallback
                    if not makcu_connected and hasattr(self.tracker, "controller") and self.tracker.controller:
                        makcu_connected = True
                except:
                    pass
                
                self.makcu_status_indicator.configure(text="●", text_color="green" if makcu_connected else "red")
            except Exception:
                self.makcu_status_indicator.configure(text="●", text_color="red")
                
        except Exception:
            pass
        self.after(500, self._update_connection_status_loop)

    def _on_finder_change(self):
        try:
            names = self.finder.get_source_names() or []
        except Exception:
            names = []
        self.source_queue.put(names)

    def _apply_sources_to_ui(self, names):
        self.ndi_sources = names
        if names:
            self.source_option.configure(values=names)
            self.source_option.set(names[0])
            self.selected_source = names[0]
        else:
            self.source_option.configure(values=["(no sources)"])
            self.source_option.set("(no sources)")
            self.selected_source = None

    def _on_close(self):
        try:
            self.tracker.stop()
        except Exception:
            pass
        try:
            if self.finder:
                self.finder.close()
        except Exception:
            pass
        self.destroy()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    ctk.set_appearance_mode("Dark")
    try:
        ctk.set_default_color_theme("themes/metal.json")
    except Exception:
        pass
    app = ViewerApp()
    app.protocol("WM_DELETE_WINDOW", app._on_close)
    app.mainloop()
