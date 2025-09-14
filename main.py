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
import logging

from cyndilib.finder import Finder
from cyndilib.receiver import Receiver
from cyndilib.wrapper.ndi_recv import RecvColorFormat, RecvBandwidth
from cyndilib.video_frame import VideoFrameSync
from cyndilib.audio_frame import AudioFrameSync

from config import config
from mouse import Mouse, is_button_pressed
from detection import load_model, perform_detection

# Setup logging
os.makedirs("data", exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,  # Increase log level for more detailed information
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s',
    handlers=[
        logging.FileHandler('data/bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


BUTTONS = {
    0: 'Left Mouse Button',
    1: 'Right Mouse Button',
    2: 'Middle Mouse Button',
    3: 'Side Mouse 4 Button',
    4: 'Side Mouse 5 Button'
}

def threaded_silent_move(controller, dx, dy):
    """Small move-restore for Silent mode."""
    controller.move(dx, dy)
    time.sleep(0.001)
    controller.click()
    time.sleep(0.001)
    controller.move(-dx, -dy)


class AimTracker:
    def __init__(self, app, target_fps=80):
        self.app = app
        # --- Params (with fallback values) ---
        self.normal_x_speed = float(getattr(config, "normal_x_speed", 0.5))
        self.normal_y_speed = float(getattr(config, "normal_y_speed", 0.5))
        self.normalsmooth = float(getattr(config, "normalsmooth", 10))
        self.normalsmoothfov = float(getattr(config, "normalsmoothfov", 10))
        self.mouse_dpi = float(getattr(config, "mouse_dpi", 800))
        self.fovsize = float(getattr(config, "fovsize", 300))
        self.tbfovsize = float(getattr(config, "tbfovsize", 70))
        self.tbdelay = float(getattr(config, "tbdelay", 0.08))
        self.last_tb_click_time = 0.0

        self.in_game_sens = float(getattr(config, "in_game_sens", 7))
        self.color = getattr(config, "color", "yellow")
        self.mode = getattr(config, "mode", "V1aim")
        self.always_aim = getattr(config, "always_aim", False)
        self.selected_mouse_button = getattr(config, "selected_mouse_button", 3),
        self.selected_tb_btn= getattr(config, "selected_tb_btn", 3)
        self.max_speed = float(getattr(config, "max_speed", 1000.0))
        
        # Flick related parameters
        self.flick_strength = float(getattr(config, "flick_strength", 1.0))
        self.enableflick = getattr(config, "enableflick", False)
        self.enablebezier = getattr(config, "enablebezier", False)
        self.enableflicktb = getattr(config, "enableflicktb", False)
        self.last_flick_time = 0.0
        self.flick_cooldown = float(getattr(config, "flick_cooldown", 0.1))  # Adjustable cooldown time
        
        # Silent related parameters
        self.silent_cooldown = float(getattr(config, "silent_cooldown", 0.1))  # Silent mode cooldown time
        self.last_silent_time = 0.0
        self.silent_strength = float(getattr(config, "silent_strength", 1.0))  # Silent mode strength
        self.enablesilent = getattr(config, "enablesilent", False)  # Enable Silent mode
        self.enablesilentbezier = getattr(config, "enablesilentbezier", False)  # Silent mode bezier curve
        self.enablesilenttb = getattr(config, "enablesilenttb", False)  # Silent mode auto shoot
        self.enableultrafast = getattr(config, "enableultrafast", True)  # Enable ultra fast mode
        self.enableextremefast = getattr(config, "enableextremefast", False)  # Enable extreme fast mode
        
        # Y offset for aiming adjustment
        self.y_offset = float(getattr(config, "y_offset", 0))
        
        # Silent mode related variables
        self.original_mouse_pos = None  # Record original mouse position
        self.silent_move_in_progress = False  # Mark if silent movement is in progress

        self.controller = Mouse()
        self.move_queue = queue.Queue(maxsize=50)
        self._move_thread = threading.Thread(target=self._process_move_queue, daemon=True)
        self._move_thread.start()

        self.model, self.class_names = load_model()
        print("Classes:", self.class_names)
        logger.info(f"AimTracker initialized with mode: {self.mode}, target FPS: {target_fps}")
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

    def _bezier_curve(self, start_x, start_y, end_x, end_y, steps=10):
        """Generate bezier curve path points"""
        # Calculate control points to create a natural curve
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2
        
        # Add some randomness to make the curve more natural
        import random
        offset_x = random.uniform(-20, 20)
        offset_y = random.uniform(-20, 20)
        
        control_x = mid_x + offset_x
        control_y = mid_y + offset_y
        
        points = []
        for i in range(steps + 1):
            t = i / steps
            # Quadratic bezier curve formula
            x = (1-t)**2 * start_x + 2*(1-t)*t * control_x + t**2 * end_x
            y = (1-t)**2 * start_y + 2*(1-t)*t * control_y + t**2 * end_y
            points.append((x, y))
        
        return points

    def _flick_to_target(self, target_x, target_y, center_x, center_y):
        """Execute flick movement to target"""
        now = time.time()
        if now - self.last_flick_time < self.flick_cooldown:
            return False  # In cooldown
        
        self.last_flick_time = now
        
        # Calculate movement distance
        dx = target_x - center_x
        dy = target_y - center_y
        
        # Apply flick strength
        dx *= self.flick_strength
        dy *= self.flick_strength
        
        if getattr(config, "enablebezier", False):
            # Use bezier curve
            points = self._bezier_curve(0, 0, dx, dy, steps=5)
            for i, (px, py) in enumerate(points):
                if i > 0:
                    prev_x, prev_y = points[i-1]
                    move_x = px - prev_x
                    move_y = py - prev_y
                    ddx, ddy = self._clip_movement(move_x, move_y)
                    self.move_queue.put((ddx, ddy, 0.001))
        else:
            # Direct movement
            ddx, ddy = self._clip_movement(dx, dy)
            self.move_queue.put((ddx, ddy, 0.001))
        
        logger.debug(f"Flick executed: target=({target_x:.1f}, {target_y:.1f}), move=({dx:.1f}, {dy:.1f})")
        return True

    def _silent_flick_to_target(self, target_x, target_y, center_x, center_y):
        """Execute silent flick movement to target and return to original position"""
        now = time.time()
        if now - self.last_silent_time < self.silent_cooldown:
            return False  # In cooldown
        
        self.last_silent_time = now
        
        # Record current mouse position
        try:
            import win32gui
            self.original_mouse_pos = win32gui.GetCursorPos()
            logger.debug(f"Silent mode: Recorded original mouse position: {self.original_mouse_pos}")
        except Exception as e:
            logger.error(f"Failed to get mouse position: {e}")
            self.original_mouse_pos = (0, 0)
        
        # Calculate movement distance
        dx = target_x - center_x
        dy = target_y - center_y
        
        # Calculate distance to target
        distance_to_target = math.hypot(dx, dy)
        logger.debug(f"Silent mode: Distance to target: {distance_to_target:.1f} pixels")
        
        # Apply silent strength
        dx *= self.silent_strength
        dy *= self.silent_strength
        
        if getattr(config, "enablesilentbezier", False):
            # Use bezier curve (reduce steps to improve speed)
            points = self._bezier_curve(0, 0, dx, dy, steps=3)
            for i, (px, py) in enumerate(points):
                if i > 0:
                    prev_x, prev_y = points[i-1]
                    move_x = px - prev_x
                    move_y = py - prev_y
                    ddx, ddy = self._clip_movement(move_x, move_y)
                    self.move_queue.put((ddx, ddy, 0.0001))  # Reduce delay
        else:
            # Direct movement (no delay)
            ddx, ddy = self._clip_movement(dx, dy)
            self.move_queue.put((ddx, ddy, 0.0001))  # Reduce delay
        
        # Shoot (auto shoot based on settings)
        if getattr(config, "enablesilenttb", False):
            # Remove delay, shoot directly
            self.controller.click()
        
        # Return to original position (no delay)
        self.controller.move(-dx, -dy)
        
        logger.debug(f"Silent flick executed: target=({target_x:.1f}, {target_y:.1f}), move=({dx:.1f}, {dy:.1f}), returned to original position")
        return True

    def _ultra_fast_silent_flick(self, target_x, target_y, center_x, center_y):
        """Ultra fast Silent mode - no delay"""
        now = time.time()
        if now - self.last_silent_time < self.silent_cooldown:
            return False  # In cooldown
        
        self.last_silent_time = now
        
        # Calculate movement distance and apply strength (combined calculation)
        dx = (target_x - center_x) * self.silent_strength
        dy = (target_y - center_y) * self.silent_strength
        
        # Use the most direct movement method - no delay, no logging
        ddx, ddy = self._clip_movement(dx, dy)
        
        # Instantly move to target
        self.controller.move(ddx, ddy)
        
        # Shoot (auto shoot based on settings)
        if getattr(config, "enablesilenttb", False):
            self.controller.click()
        
        # Instantly return to original position
        self.controller.move(-ddx, -ddy)
        
        return True

    def _extreme_fast_silent_flick(self, target_x, target_y, center_x, center_y):
        """Extreme speed Silent mode - fastest speed"""
        now = time.time()
        if now - self.last_silent_time < self.silent_cooldown:
            return False
        
        self.last_silent_time = now
        
        # Pre-calculate all values to reduce function calls
        dx = int((target_x - center_x) * self.silent_strength)
        dy = int((target_y - center_y) * self.silent_strength)
        
        # Directly limit range to avoid function calls
        max_speed = int(self.max_speed)
        dx = max(-max_speed, min(dx, max_speed))
        dy = max(-max_speed, min(dy, max_speed))
        
        # Instantly move to target
        self.controller.move(dx, dy)
        
        # Shoot (auto shoot based on settings)
        if getattr(config, "enablesilenttb", False):
            self.controller.click()
        
        # Instantly return to original position
        self.controller.move(-dx, -dy)
        
        return True

    def _silent_move_with_return(self, dx, dy, distance_to_target):
        """Silent mode movement and return to original position"""
        try:
            logger.debug(f"Silent move: Moving {dx}, {dy} pixels, distance to target: {distance_to_target:.1f}")
            
            # Move to target
            self.controller.move(dx, dy)
            time.sleep(0.001)
            
            # Shoot
            self.controller.click()
            time.sleep(0.001)
            
            # Return to original position
            self.controller.move(-dx, -dy)
            
            logger.debug(f"Silent move completed: moved to target and returned to original position")
            
        except Exception as e:
            logger.error(f"Silent move error: {e}")
        finally:
            # Reset state
            self.silent_move_in_progress = False
            self.original_mouse_pos = None

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
        if getattr(config, "enableaim", False):
            cv2.circle(img, (center_x, center_y), int(getattr(config, "fovsize", self.fovsize)), (255, 255, 255), 2)
            # Correct: cercle smoothing = normalsmoothFOV
            cv2.circle(img, (center_x, center_y), int(getattr(config, "normalsmoothfov", self.normalsmoothfov)), (51, 255, 255), 2)
        if getattr(config, "enabletb", False):
            cv2.circle(img, (center_x, center_y), int(getattr(config, "tbfovsize", self.tbfovsize)), (255, 255, 255), 2)

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
            detection_results, mask = perform_detection(self.model, bgr_img)
            cv2.imshow("MASK", mask)
            cv2.waitKey(1)
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
                    # Draw body
                    self._draw_body(bgr_img, x1, y1, x2, y2, conf)
                    # Estimate head positions in bbox
                    head_positions = self._estimate_head_positions(x1, y1, x2, y2, bgr_img)
                    for head_cx, head_cy, bbox in head_positions:
                        self._draw_head_bbox(bgr_img, head_cx, head_cy)
                        d = math.hypot(head_cx - frame.xres / 2.0, head_cy - frame.yres / 2.0)
                        targets.append((head_cx, head_cy, d))
                except Exception as e:
                    print("Error in _estimate_head_positions:", e)


        # Draw FOVs once per frame
        try:
            self._draw_fovs(bgr_img, frame)
        except Exception:
            pass

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

        # Light crop
        top_crop_factor = 0.10
        side_crop_factor = 0.10

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
            detections, mask = perform_detection(self.model, roi)
        except Exception as e:
            print("[perform_detection ROI error]", e)

        if not detections:
            # No detection → keep head position with offset
            results.append((headx_base, heady_base, (x1_roi, y1_roi, x2_roi, y2_roi)))
        else:
            for det in detections:
                x, y, w, h = det["bbox"]
                cv2.rectangle(img, (x1_roi + x, y1_roi + y), (x1_roi + x + w, y1_roi + y + h), (0, 255, 0), 2)

                # Raw detection position
                headx_det = x1_roi + x + w / 2
                heady_det = y1_roi + y + h / 2

                # Apply offset also on detection
                headx_det += effective_width * (offsetX / 100)
                heady_det += effective_height * (offsetY / 100)

                results.append((headx_det, heady_det, (x1_roi + x, y1_roi + y, w, h)))

        return results

    def _draw_body(self, img, x1, y1, x2, y2, conf):
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(img, f"Body {conf:.2f}", (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    def _aim_and_move(self, targets, frame, img):
        
        aim_enabled = getattr(config, "enableaim", False)
        always_aim = getattr(config, "always_aim", False)
        selected_btn = getattr(config, "selected_mouse_button", None)

        # Helper function to check if aiming should be active
        def should_aim():
            if not aim_enabled:
                return False
            if always_aim:
                return True
            return selected_btn is not None and is_button_pressed(selected_btn)

        center_x = frame.xres / 2.0
        center_y = frame.yres / 2.0
        # --- If no target, skip aimbot but continue triggerbot ---
        if not targets:
            cx, cy, distance_to_center = center_x, center_y, float("inf")
        else:
            # Select the best target
            best_target = min(targets, key=lambda t: t[2])
            cx, cy, _ = best_target
            distance_to_center = math.hypot(cx - center_x, cy - center_y)
            if distance_to_center > float(getattr(config, 'fovsize', self.fovsize)):
                return

        dx = cx - center_x
        dy = cy - center_y + self.y_offset

        sens = float(getattr(config, "in_game_sens", self.in_game_sens))
        dpi = float(getattr(config, "mouse_dpi", self.mouse_dpi))

        cm_per_rev_base = 54.54
        cm_per_rev = cm_per_rev_base / max(sens, 0.01)

        count_per_cm = dpi / 2.54
        deg_per_count = 360.0 / (cm_per_rev * count_per_cm)

        ndx = dx * deg_per_count
        ndy = dy * deg_per_count

        mode = getattr(config, "mode", "V1aim")
        if mode == "V1aim":
           
            try:
                
                # --- FLICK CHECK ---
                flick_enabled = getattr(config, "enableflick", False)
                if flick_enabled and should_aim() and targets:
                    # Check if aim key was just pressed (flick trigger)
                    if self._flick_to_target(cx, cy, center_x, center_y):
                        # If flick triggerbot is enabled, auto shoot
                        if getattr(config, "enableflicktb", False):
                            time.sleep(0.01)  # Brief delay to ensure movement completion
                            self.controller.click()
                            logger.debug("Flick triggerbot activated")
                        return  # Return directly after flick execution, skip normal aimbot
                
                # --- AIMBOT ---
                if should_aim() and targets:
                    if distance_to_center < float(getattr(config, "normalsmoothfov", self.normalsmoothfov)):
                       
                        ndx *= float(getattr(config, "normal_x_speed", self.normal_x_speed)) / max(float(getattr(config, "normalsmooth", self.normalsmooth)), 0.01)
                        ndy *= float(getattr(config, "normal_y_speed", self.normal_y_speed)) / max(float(getattr(config, "normalsmooth", self.normalsmooth)), 0.01)
                    else:
                        ndx *= float(getattr(config, "normal_x_speed", self.normal_x_speed))
                        ndy *= float(getattr(config, "normal_y_speed", self.normal_y_speed))
                    ddx, ddy = self._clip_movement(ndx, ndy)
                    self.move_queue.put((ddx, ddy, 0.005))
            except Exception:
                pass

            try:
                # --- Triggerbot parameters ---
                if getattr(config, "enabletb", False) and is_button_pressed(getattr(config, "selected_tb_btn", None)) or is_button_pressed(getattr(config, "selected_2_tb", None)):

                    # Screen center
                    cx0, cy0 = int(frame.xres // 2), int(frame.yres // 2)
                    ROI_SIZE = 5  # small square around center
                    x1, y1 = max(cx0 - ROI_SIZE, 0), max(cy0 - ROI_SIZE, 0)
                    x2, y2 = min(cx0 + ROI_SIZE, img.shape[1]), min(cy0 + ROI_SIZE, img.shape[0])
                    roi = img[y1:y2, x1:x2]

                    if roi.size == 0:
                        return  # safety check

                    # HSV conversion (ensure img is BGR)
                    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                    # HSV range for purple (adjust if necessary)
                    
                    HSV_UPPER = self.model[1]
                    HSV_LOWER = self.model[0]
                    
                    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

                    detected = cv2.countNonZero(mask) > 0
                    #print(f"ROI shape: {roi.shape}, NonZero pixels: {cv2.countNonZero(mask)}")

                    # Debug display
                    cv2.imshow("ROI", roi)
                    cv2.imshow("Mask", mask)
                    cv2.waitKey(1)

                    # Text on main image
                    if detected:
                        cv2.putText(img, "PURPLE DETECTED", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        now = time.time()
                        if now - self.last_tb_click_time >= float(getattr(config, "tbdelay", self.tbdelay)):
                            self.controller.click()
                            self.last_tb_click_time = now

            except Exception as e:
                print("[Triggerbot error]", e)


        elif mode == "V2aim":
            # V2aim - optimized aimbot logic
            try:
                # --- FLICK CHECK ---
                flick_enabled = getattr(config, "enableflick", False)
                if flick_enabled and should_aim() and targets:
                    # Check if aim key was just pressed (flick trigger)
                    if self._flick_to_target(cx, cy, center_x, center_y):
                        # If flick triggerbot is enabled, auto shoot
                        if getattr(config, "enableflicktb", False):
                            time.sleep(0.01)  # Brief delay to ensure movement completion
                            self.controller.click()
                            logger.debug("V2aim Flick triggerbot activated")
                        return  # Return directly after flick execution, skip normal aimbot
                
                # --- AIMBOT V2 ---
                if should_aim() and targets:
                    # V2aim uses more precise aiming algorithm
                    if distance_to_center < float(getattr(config, "normalsmoothfov", self.normalsmoothfov)):
                        # Use more precise calculation within smoothing FOV
                        smoothing_factor = max(float(getattr(config, "normalsmooth", self.normalsmooth)), 0.01)
                        # V2aim uses smoother movement curve
                        ndx *= float(getattr(config, "normal_x_speed", self.normal_x_speed)) / (smoothing_factor * 1.2)
                        ndy *= float(getattr(config, "normal_y_speed", self.normal_y_speed)) / (smoothing_factor * 1.2)
                    else:
                        # Use standard speed outside FOV but slightly reduced
                        ndx *= float(getattr(config, "normal_x_speed", self.normal_x_speed)) * 0.9
                        ndy *= float(getattr(config, "normal_y_speed", self.normal_y_speed)) * 0.9
                    
                    # V2aim uses more precise movement limits
                    ddx, ddy = self._clip_movement(ndx, ndy)
                    # Add tiny delay for smoother experience
                    self.move_queue.put((ddx, ddy, 0.003))
                    logger.debug(f"V2aim: Moving to target at ({cx:.1f}, {cy:.1f}), distance: {distance_to_center:.1f}")
            except Exception as e:
                logger.error(f"V2aim error: {e}")

            try:
                # --- Triggerbot (V2aim also supports triggerbot) ---
                if getattr(config, "enabletb", False) and is_button_pressed(getattr(config, "selected_tb_btn", None)) or is_button_pressed(getattr(config, "selected_2_tb", None)):
                    # Screen center
                    cx0, cy0 = int(frame.xres // 2), int(frame.yres // 2)
                    ROI_SIZE = 5  # small square around center
                    x1, y1 = max(cx0 - ROI_SIZE, 0), max(cy0 - ROI_SIZE, 0)
                    x2, y2 = min(cx0 + ROI_SIZE, img.shape[1]), min(cy0 + ROI_SIZE, img.shape[0])
                    roi = img[y1:y2, x1:x2]

                    if roi.size == 0:
                        return  # safety check

                    # HSV conversion (ensure img is BGR)
                    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                    # HSV range for purple (adjust if necessary)
                    HSV_UPPER = self.model[1]
                    HSV_LOWER = self.model[0]
                    
                    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

                    detected = cv2.countNonZero(mask) > 0

                    # Debug display
                    cv2.imshow("ROI", roi)
                    cv2.imshow("Mask", mask)
                    cv2.waitKey(1)

                    # Text on main image
                    if detected:
                        cv2.putText(img, "PURPLE DETECTED", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        now = time.time()
                        if now - self.last_tb_click_time >= float(getattr(config, "tbdelay", self.tbdelay)):
                            self.controller.click()
                            self.last_tb_click_time = now
                            logger.debug("V2aim: Triggerbot activated")

            except Exception as e:
                logger.error(f"V2aim triggerbot error: {e}")

        elif mode == "Flick":
            # Flick mode - instantly move to enemy head when pressing aim
            try:
                if should_aim() and targets:
                    # Execute flick movement to target
                    if self._flick_to_target(cx, cy, center_x, center_y):
                        # If flick triggerbot is enabled, auto shoot
                        if getattr(config, "enableflicktb", False):
                            time.sleep(0.01)  # Brief delay to ensure movement completion
                            self.controller.click()
                            logger.debug("Flick mode triggerbot activated")
                        else:
                            # If unchecked, move to enemy head instantly then perform normal aimbot
                            logger.debug("Flick mode: moved to target, continuing with normal aim")
            except Exception as e:
                logger.error(f"Flick mode error: {e}")

        elif mode == "Silent":
            # Silent mode - instantly move to target and return to original position
            try:
                silent_enabled = getattr(config, "enablesilent", False)
                if silent_enabled and should_aim() and targets:
                    # Choose speed mode based on settings
                    if getattr(config, "enableextremefast", False):
                        # Use extreme speed silent flick to move to target and return
                        if self._extreme_fast_silent_flick(cx, cy, center_x, center_y):
                            pass  # No logging to improve speed
                    elif getattr(config, "enableultrafast", True):
                        # Use ultra fast silent flick to move to target and return
                        if self._ultra_fast_silent_flick(cx, cy, center_x, center_y):
                            pass  # No logging to improve speed
                    else:
                        # Use normal silent flick to move to target and return
                        if self._silent_flick_to_target(cx, cy, center_x, center_y):
                            logger.debug("Silent mode: Flick executed and returned to original position")
            except Exception as e:
                logger.error(f"Silent mode error: {e}")



class ViewerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("CUPSY COLORBOT")
        self.geometry("400x700")

        # Dictionaries for UI <-> config updates
        self._slider_widgets = {}   # key -> {"slider": widget, "label": widget, "min":..., "max":...}
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
        # Remove native bar
       

        # Custom bar
        self.title_bar = ctk.CTkFrame(self, height=30, corner_radius=0)
        self.title_bar.pack(fill="x", side="top")

        self.title_label = ctk.CTkLabel(self.title_bar, text="CUPSY CB", anchor="w")
        self.title_label.pack(side="left", padx=10)

        # Close button
        self.close_btn = ctk.CTkButton(self.title_bar, text="X", width=25, command=self.destroy)
        self.close_btn.pack(side="right", padx=2)

        # Make bar draggable
        self.title_bar.bind("<Button-1>", self.start_move)
        self.title_bar.bind("<B1-Motion>", self.do_move)
        
        # Tracker
        self.tracker = AimTracker(app=self, target_fps=float(getattr(config, "target_fps", 80)))

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

    # ---------- UI mapping helpers ----------
    def _register_slider(self, key, slider, label, vmin, vmax, is_float):
        self._slider_widgets[key] = {"slider": slider, "label": label, "min": vmin, "max": vmax, "is_float": is_float}

    def _load_initial_config(self):
        try:
            import json, os
            from detection import reload_model
            if os.path.exists("configs/default.json"):
                with open("configs/default.json", "r") as f:
                    data = json.load(f)

                self._apply_settings(data)


            else:
                print("Config file doesn't exist")
        except Exception as e:
            print("Unable to load initial config:", e)




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
        # Refresh label
        txt = f"{key.replace('_',' ').title()}: {v:.2f}" if is_float else f"{key.replace('_',' ').title()}: {int(v)}"
        # Keep human-readable label (X Speed etc.) if already present
        current = w["label"].cget("text")
        prefix = current.split(":")[0] if ":" in current else txt.split(":")[0]
        w["label"].configure(text=f"{prefix}: {v:.2f}" if is_float else f"{prefix}: {int(v)}")

    def _set_checkbox_value(self, key, value_bool):
        var = self._checkbox_vars.get(key)
        if var is not None:
            var.set(bool(value_bool))

    def _set_option_value(self, key, value_str):
        menu = self._option_widgets.get(key)
        if menu is not None and value_str is not None:
            menu.set(str(value_str))

    def _set_btn_option_value(self, key, value_str):
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
            "normal_x_speed": getattr(config, "normal_x_speed", 0.5),
            "normal_y_speed": getattr(config, "normal_y_speed", 0.5),
            "normalsmooth": getattr(config, "normalsmooth", 10),
            "normalsmoothfov": getattr(config, "normalsmoothfov", 10),
            "mouse_dpi" : getattr(config, "mouse_dpi", 800),
            "fovsize": getattr(config, "fovsize", 300),
            "tbfovsize": getattr(config, "tbfovsize", 70),
            "tbdelay": getattr(config, "tbdelay", 0.08),
            "in_game_sens": getattr(config, "in_game_sens", 7),
            "color": getattr(config, "color", "yellow"),
            "mode": getattr(config, "mode", "V1aim"),
            "enableaim": getattr(config, "enableaim", False),
            "always_aim": getattr(config, "always_aim", False),
            "enabletb": getattr(config, "enabletb", False),
            "selected_mouse_button": getattr(config, "selected_mouse_button", 3),
            "selected_tb_btn": getattr(config, "selected_tb_btn", 3),
            "flick_strength": getattr(config, "flick_strength", 1.0),
            "enableflick": getattr(config, "enableflick", False),
            "enablebezier": getattr(config, "enablebezier", False),
            "enableflicktb": getattr(config, "enableflicktb", False),
            "flick_cooldown": getattr(config, "flick_cooldown", 0.1),
            "silent_cooldown": getattr(config, "silent_cooldown", 0.1),
            "silent_strength": getattr(config, "silent_strength", 1.0),
            "enablesilent": getattr(config, "enablesilent", False),
            "enablesilentbezier": getattr(config, "enablesilentbezier", False),
            "enablesilenttb": getattr(config, "enablesilenttb", False),
            "enableultrafast": getattr(config, "enableultrafast", True),
            "enableextremefast": getattr(config, "enableextremefast", False),
            "target_fps": getattr(config, "target_fps", 80),
            "y_offset": getattr(config, "y_offset", 0)

        }

    def _apply_settings(self, data, config_name=None):
        """
        Apply a settings dictionary to global config, tracker and UI.
        Reload model if necessary.
        """
        try:
            # --- Apply to global config ---
            for k, v in data.items():
                setattr(config, k, v)

            # --- Apply to tracker if attribute exists ---
            for k, v in data.items():
                if hasattr(self.tracker, k):
                    setattr(self.tracker, k, v)

            # --- Update sliders ---
            for k, v in data.items():
                if k in self._slider_widgets:
                    self._set_slider_value(k, v)

            # --- Update checkboxes ---
            for k, v in data.items():
                if k in self._checkbox_vars:
                    self._set_checkbox_value(k, v)

            # --- Update OptionMenus ---
            for k, v in data.items():
                if k in self._option_widgets:
                    self._set_option_value(k, v)

            # --- Update OptionMenus ---
            for k, v in data.items():
                if k == "selected_mouse_button" or k == "selected_tb_btn":

                    if k in self._option_widgets:
                        print(k, v)
                        
                        v = BUTTONS[v]
                        print(v)
                        self._set_btn_option_value(k, v)

            # --- Reload model if necessary ---
            from detection import reload_model
            self.tracker.model, self.tracker.class_names = reload_model()

            if config_name:
                self._log_config(f"Config '{config_name}' applied and model reloaded ✅")
            else:
                self._log_config(f"Config applied and model reloaded ✅")

        except Exception as e:
            self._log_config(f"[Error _apply_settings] {e}")


    def _save_new_config(self):
        from tkinter import simpledialog
        name = simpledialog.askstring("Config name", "Enter the config name:")
        if not name:
            self._log_config("Cancelled save (no name provided).")
            return
        data = self._get_current_settings()
        path = os.path.join("configs", f"{name}.json")
        try:
            os.makedirs("configs", exist_ok=True)
            with open(path, "w") as f:
                json.dump(data, f, indent=4)
            self._refresh_config_list()
            self.config_option.set(name)  # Automatically select
            self._log_config(f"New config '{name}' saved ✅")
        except Exception as e:
            self._log_config(f"[Error SAVE] {e}")



    def _load_selected_config(self):
        """
        Load the selected config from OptionMenu.
        """
        name = self.config_option.get()
        path = os.path.join("configs", f"{name}.json")
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self._apply_settings(data, config_name=name)
            self._log_config(f"Config '{name}' loaded 📂")
        except Exception as e:
            self._log_config(f"[Error LOAD] {e}")




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
            self._log_config(f"Config '{name}' saved ✅")
            self._refresh_config_list()
        except Exception as e:
            self._log_config(f"[Error SAVE] {e}")

    def _load_config(self):
        name = self.config_option.get() or "default"
        path = os.path.join("configs", f"{name}.json")
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self._apply_settings(data)
            self._log_config(f"Config '{name}' loaded 📂")
        except Exception as e:
            self._log_config(f"[Error LOAD] {e}")

    def _log_config(self, msg):
        self.config_log.insert("end", msg + "\n")
        self.config_log.see("end")

    # ----------------------- UI BUILDERS -----------------------
    def _build_general_tab(self):
        self.status_label = ctk.CTkLabel(self.tab_general, text="Status: Disconnected")
        self.status_label.pack(pady=5, anchor="w")
        
        # Target FPS Control
        s, l = self._add_slider_with_label(self.tab_general, "Target FPS", 30, 240, float(getattr(config, "target_fps", 80)), self._on_target_fps_changed, is_float=True)
        self._register_slider("target_fps", s, l, 30, 240, True)

        self.source_option = ctk.CTkOptionMenu(self.tab_general, values=["(searching...)"], command=self._on_source_selected)
        self.source_option.pack(pady=5, fill="x")

        ctk.CTkButton(self.tab_general, text="Refresh NDI Sources", command=self._refresh_sources).pack(pady=5, fill="x")
        ctk.CTkButton(self.tab_general, text="Connect to Source", command=self._connect_to_selected).pack(pady=5, fill="x")
        #ctk.CTkButton(self.tab_general, text="Toggle Rage Mode", command=self._toggle_rage).pack(pady=5, fill="x")

        ctk.CTkLabel(self.tab_general, text="Appearance").pack(pady=5)
        ctk.CTkOptionMenu(self.tab_general, values=["Dark", "Light"], command=self._on_appearance_selected).pack(pady=5, fill="x")

        ctk.CTkLabel(self.tab_general, text="Mode").pack(pady=5)
        self.mode_option = ctk.CTkOptionMenu(self.tab_general, values=["V1aim", "V2aim", "Silent", "Flick"], command=self._on_mode_selected)
        self.mode_option.pack(pady=5, fill="x")
        self._option_widgets["mode"] = self.mode_option

        ctk.CTkLabel(self.tab_general, text="Color").pack(pady=5)
        self.color_option = ctk.CTkOptionMenu(self.tab_general, values=["yellow", "purple"], command=self._on_color_selected)
        self.color_option.pack(pady=5, fill="x")
        self._option_widgets["color"] = self.color_option

    def _build_aimbot_tab(self):
        # X Speed
        s, l = self._add_slider_with_label(self.tab_aimbot, "X Speed", 0.1, 2000, float(getattr(config, "normal_x_speed", 0.5)), self._on_normal_x_speed_changed, is_float=True)
        self._register_slider("normal_x_speed", s, l, 0.1, 2000, True)
        # Y Speed
        s, l = self._add_slider_with_label(self.tab_aimbot, "Y Speed", 0.1, 2000, float(getattr(config, "normal_y_speed", 0.5)), self._on_normal_y_speed_changed, is_float=True)
        self._register_slider("normal_y_speed", s, l, 0.1, 2000, True)
        # In-game sens
        s, l = self._add_slider_with_label(self.tab_aimbot, "In-game sens", 0.1, 2000, float(getattr(config, "in_game_sens", 7)), self._on_config_in_game_sens_changed, is_float=True)
        self._register_slider("in_game_sens", s, l, 0.1, 2000, True)
        # Smoothing
        s, l = self._add_slider_with_label(self.tab_aimbot, "Smoothing", 1, 30, float(getattr(config, "normalsmooth", 10)), self._on_config_normal_smooth_changed, is_float=True)
        self._register_slider("normalsmooth", s, l, 1, 30, True)
        # Smoothing FOV
        s, l = self._add_slider_with_label(self.tab_aimbot, "Smoothing FOV", 1, 30, float(getattr(config, "normalsmoothfov", 10)), self._on_config_normal_smoothfov_changed, is_float=True)
        self._register_slider("normalsmoothfov", s, l, 1, 30, True)
        # FOV Size
        s, l = self._add_slider_with_label(self.tab_aimbot, "FOV Size", 1, 1000, float(getattr(config, "fovsize", 300)), self._on_fovsize_changed, is_float=True)
        self._register_slider("fovsize", s, l, 1, 1000, True)
        # Y Offset
        s, l = self._add_slider_with_label(self.tab_aimbot, "Y Offset", -100, 100, float(getattr(config, "y_offset", 0)), self._on_y_offset_changed, is_float=True)
        self._register_slider("y_offset", s, l, -100, 100, True)

        # Flick related control container (initially hidden)
        self.flick_frame = ctk.CTkFrame(self.tab_aimbot)
        self.flick_frame.pack(pady=10, fill="x", padx=10)
        
        # Flick Strength
        s, l = self._add_slider_with_label(self.flick_frame, "Flick Strength", 0.1, 5.0, float(getattr(config, "flick_strength", 1.0)), self._on_flick_strength_changed, is_float=True)
        self._register_slider("flick_strength", s, l, 0.1, 5.0, True)
        
        # Flick Cooldown
        s, l = self._add_slider_with_label(self.flick_frame, "Flick Cooldown", 0.01, 1.0, float(getattr(config, "flick_cooldown", 0.1)), self._on_flick_cooldown_changed, is_float=True)
        self._register_slider("flick_cooldown", s, l, 0.01, 1.0, True)

        # Enable Flick
        self.var_enableflick = tk.BooleanVar(value=getattr(config, "enableflick", False))
        ctk.CTkCheckBox(self.flick_frame, text="Enable Flick", variable=self.var_enableflick, command=self._on_enableflick_changed).pack(pady=6, anchor="w")
        self._checkbox_vars["enableflick"] = self.var_enableflick

        # Enable Bezier Curve
        self.var_enablebezier = tk.BooleanVar(value=getattr(config, "enablebezier", False))
        ctk.CTkCheckBox(self.flick_frame, text="Use Bezier Curve", variable=self.var_enablebezier, command=self._on_enablebezier_changed).pack(pady=6, anchor="w")
        self._checkbox_vars["enablebezier"] = self.var_enablebezier

        # Enable Flick Triggerbot
        self.var_enableflicktb = tk.BooleanVar(value=getattr(config, "enableflicktb", False))
        ctk.CTkCheckBox(self.flick_frame, text="Flick Auto Shoot", variable=self.var_enableflicktb, command=self._on_enableflicktb_changed).pack(pady=6, anchor="w")
        self._checkbox_vars["enableflicktb"] = self.var_enableflicktb
        
        # Initially hide Flick controls
        self.flick_frame.pack_forget()
        
        # Silent related control container (initially hidden)
        self.silent_frame = ctk.CTkFrame(self.tab_aimbot)
        self.silent_frame.pack(pady=10, fill="x", padx=10)
        
        # Silent Strength
        s, l = self._add_slider_with_label(self.silent_frame, "Silent Strength", 0.1, 5.0, float(getattr(config, "silent_strength", 1.0)), self._on_silent_strength_changed, is_float=True)
        self._register_slider("silent_strength", s, l, 0.1, 5.0, True)
        
        # Silent Cooldown
        s, l = self._add_slider_with_label(self.silent_frame, "Silent Cooldown", 0.01, 1.0, float(getattr(config, "silent_cooldown", 0.1)), self._on_silent_cooldown_changed, is_float=True)
        self._register_slider("silent_cooldown", s, l, 0.01, 1.0, True)

        # Enable Silent
        self.var_enablesilent = tk.BooleanVar(value=getattr(config, "enablesilent", False))
        ctk.CTkCheckBox(self.silent_frame, text="Enable Silent", variable=self.var_enablesilent, command=self._on_enablesilent_changed).pack(pady=6, anchor="w")
        self._checkbox_vars["enablesilent"] = self.var_enablesilent

        # Enable Silent Bezier Curve
        self.var_enablesilentbezier = tk.BooleanVar(value=getattr(config, "enablesilentbezier", False))
        ctk.CTkCheckBox(self.silent_frame, text="Use Bezier Curve", variable=self.var_enablesilentbezier, command=self._on_enablesilentbezier_changed).pack(pady=6, anchor="w")
        self._checkbox_vars["enablesilentbezier"] = self.var_enablesilentbezier

        # Enable Silent Auto Shoot
        self.var_enablesilenttb = tk.BooleanVar(value=getattr(config, "enablesilenttb", False))
        ctk.CTkCheckBox(self.silent_frame, text="Silent Auto Shoot", variable=self.var_enablesilenttb, command=self._on_enablesilenttb_changed).pack(pady=6, anchor="w")
        self._checkbox_vars["enablesilenttb"] = self.var_enablesilenttb

        # Enable Ultra Fast Mode
        self.var_enableultrafast = tk.BooleanVar(value=getattr(config, "enableultrafast", True))
        ctk.CTkCheckBox(self.silent_frame, text="Ultra Fast Mode (No Delay)", variable=self.var_enableultrafast, command=self._on_enableultrafast_changed).pack(pady=6, anchor="w")
        self._checkbox_vars["enableultrafast"] = self.var_enableultrafast

        # Enable Extreme Fast Mode
        self.var_enableextremefast = tk.BooleanVar(value=getattr(config, "enableextremefast", False))
        ctk.CTkCheckBox(self.silent_frame, text="Extreme Fast Mode (Fastest)", variable=self.var_enableextremefast, command=self._on_enableextremefast_changed).pack(pady=6, anchor="w")
        self._checkbox_vars["enableextremefast"] = self.var_enableextremefast
        
        # Initially hide Silent controls
        self.silent_frame.pack_forget()

        # Enable Aim
        self.var_enableaim = tk.BooleanVar(value=getattr(config, "enableaim", False))
        ctk.CTkCheckBox(self.tab_aimbot, text="Enable Aim", variable=self.var_enableaim, command=self._on_enableaim_changed).pack(pady=6, anchor="w")
        self._checkbox_vars["enableaim"] = self.var_enableaim

        # Always Aim
        self.var_always_aim = tk.BooleanVar(value=getattr(config, "always_aim", False))
        ctk.CTkCheckBox(self.tab_aimbot, text="Always Aim", variable=self.var_always_aim, command=self._on_always_aim_changed).pack(pady=6, anchor="w")
        self._checkbox_vars["always_aim"] = self.var_always_aim

        ctk.CTkLabel(self.tab_aimbot, text="Aimbot Button").pack(pady=5, anchor="w")
        self.aimbot_button_option = ctk.CTkOptionMenu(
            self.tab_aimbot,
            values=list(BUTTONS.values()),
            command=self._on_aimbot_button_selected
        )
        self.aimbot_button_option.pack(pady=5, fill="x")
        self._option_widgets["selected_mouse_button"] = self.aimbot_button_option


    def _build_tb_tab(self):
        # TB FOV Size
        s, l = self._add_slider_with_label(self.tab_tb, "TB FOV Size", 1, 300, float(getattr(config, "tbfovsize", 70)), self._on_tbfovsize_changed, is_float=True)
        self._register_slider("tbfovsize", s, l, 1, 300, True)
        # TB Delay
        s, l = self._add_slider_with_label(self.tab_tb, "TB Delay", 0.0, 1.0, float(getattr(config, "tbdelay", 0.08)), self._on_tbdelay_changed, is_float=True)
        self._register_slider("tbdelay", s, l, 0.0, 1.0, True)

        # Enable TB
        self.var_enabletb = tk.BooleanVar(value=getattr(config, "enabletb", False))
        ctk.CTkCheckBox(self.tab_tb, text="Enable TB", variable=self.var_enabletb, command=self._on_enabletb_changed).pack(pady=6, anchor="w")
        self._checkbox_vars["enabletb"] = self.var_enabletb

        ctk.CTkLabel(self.tab_tb, text="Triggerbot Button").pack(pady=5, anchor="w")
        self.tb_button_option = ctk.CTkOptionMenu(
            self.tab_tb,
            values=list(BUTTONS.values()),
            command=self._on_tb_button_selected
        )
        self.tb_button_option.pack(pady=5, fill="x")
        self._option_widgets["selected_tb_btn"] = self.tb_button_option


    # Generic slider helper (parent-aware)
    def _add_slider_with_label(self, parent, text, min_val, max_val, init_val, command, is_float=False):
        frame = ctk.CTkFrame(parent)
        frame.pack(padx=12, pady=6, fill="x")

        label = ctk.CTkLabel(frame, text=f"{text}: {init_val:.2f}" if is_float else f"{text}: {init_val}")
        label.pack(side="left")

        steps = 100 if is_float else max(1, int(max_val - min_val))
        slider = ctk.CTkSlider(frame, from_=min_val, to=max_val, number_of_steps=steps,
                               command=lambda v: self._slider_callback(v, label, text, command, is_float))
        slider.set(init_val)
        slider.pack(side="right", fill="x", expand=True)
        return slider, label

    def _slider_callback(self, value, label, text, command, is_float):
        val = float(value) if is_float else int(round(value))
        label.configure(text=f"{text}: {val:.2f}" if is_float else f"{text}: {val}")
        command(val)

    # ----------------------- Callbacks -----------------------
    def _on_normal_x_speed_changed(self, val):
        config.normal_x_speed = val
        self.tracker.normal_x_speed = val
        logger.info(f"X Speed changed to: {val}")
        print(f"[CONFIG] X Speed set to: {val}")

    def _on_normal_y_speed_changed(self, val):
        config.normal_y_speed = val
        self.tracker.normal_y_speed = val
        logger.info(f"Y Speed changed to: {val}")
        print(f"[CONFIG] Y Speed set to: {val}")

    def _on_config_in_game_sens_changed(self, val):
        config.in_game_sens = val
        self.tracker.in_game_sens = val
        logger.info(f"In-game sensitivity changed to: {val}")
        print(f"[CONFIG] In-game sensitivity set to: {val}")

    def _on_config_normal_smooth_changed(self, val):
        config.normalsmooth = val
        self.tracker.normalsmooth = val
        logger.info(f"Smoothing changed to: {val}")
        print(f"[CONFIG] Smoothing set to: {val}")

    def _on_config_normal_smoothfov_changed(self, val):
        config.normalsmoothfov = val
        self.tracker.normalsmoothfov = val
        logger.info(f"Smoothing FOV changed to: {val}")
        print(f"[CONFIG] Smoothing FOV set to: {val}")

    def _on_target_fps_changed(self, val):
        config.target_fps = val
        self.tracker._target_fps = val
        logger.info(f"Target FPS changed to: {val}")
        print(f"[CONFIG] Target FPS set to: {val}")

    def _on_y_offset_changed(self, val):
        config.y_offset = val
        self.tracker.y_offset = val
        logger.info(f"Y Offset changed to: {val}")
        print(f"[CONFIG] Y Offset set to: {val}")

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
                #self.tracker.selected_tb_btn = val
                break
        self._log_config(f"Triggerbot button set to {val} ({key})")


    def _on_fovsize_changed(self, val):
        config.fovsize = val
        self.tracker.fovsize = val
        logger.info(f"FOV Size changed to: {val}")
        print(f"[CONFIG] FOV Size set to: {val}")

    def _on_tbdelay_changed(self, val):
        config.tbdelay = val
        self.tracker.tbdelay = val
        logger.info(f"TB Delay changed to: {val}")
        print(f"[CONFIG] TB Delay set to: {val}")

    def _on_tbfovsize_changed(self, val):
        config.tbfovsize = val
        self.tracker.tbfovsize = val
        logger.info(f"TB FOV Size changed to: {val}")
        print(f"[CONFIG] TB FOV Size set to: {val}")

    def _on_enableaim_changed(self):
        config.enableaim = self.var_enableaim.get()

    def _on_always_aim_changed(self):
        config.always_aim = self.var_always_aim.get()
        logger.info(f"Always aim enabled: {config.always_aim}")

    def _on_enabletb_changed(self):
        config.enabletb = self.var_enabletb.get()

    def _on_flick_strength_changed(self, val):
        config.flick_strength = val
        self.tracker.flick_strength = val
        logger.info(f"Flick Strength changed to: {val}")
        print(f"[CONFIG] Flick Strength set to: {val}")

    def _on_flick_cooldown_changed(self, val):
        config.flick_cooldown = val
        self.tracker.flick_cooldown = val
        logger.info(f"Flick cooldown set to: {val}s")

    def _on_silent_cooldown_changed(self, val):
        config.silent_cooldown = val
        self.tracker.silent_cooldown = val
        logger.info(f"Silent cooldown set to: {val}s")

    def _on_silent_strength_changed(self, val):
        config.silent_strength = val
        self.tracker.silent_strength = val
        logger.info(f"Silent Strength changed to: {val}")
        print(f"[CONFIG] Silent Strength set to: {val}")

    def _on_enablesilent_changed(self):
        config.enablesilent = self.var_enablesilent.get()
        logger.info(f"Silent enabled: {config.enablesilent}")

    def _on_enablesilentbezier_changed(self):
        config.enablesilentbezier = self.var_enablesilentbezier.get()
        logger.info(f"Silent bezier curve enabled: {config.enablesilentbezier}")

    def _on_enablesilenttb_changed(self):
        config.enablesilenttb = self.var_enablesilenttb.get()
        logger.info(f"Silent auto shoot enabled: {config.enablesilenttb}")

    def _on_enableultrafast_changed(self):
        config.enableultrafast = self.var_enableultrafast.get()
        logger.info(f"Ultra fast mode enabled: {config.enableultrafast}")

    def _on_enableextremefast_changed(self):
        config.enableextremefast = self.var_enableextremefast.get()
        logger.info(f"Extreme fast mode enabled: {config.enableextremefast}")

    def _on_enableflick_changed(self):
        config.enableflick = self.var_enableflick.get()
        logger.info(f"Flick enabled: {config.enableflick}")

    def _on_enablebezier_changed(self):
        config.enablebezier = self.var_enablebezier.get()
        logger.info(f"Bezier curve enabled: {config.enablebezier}")

    def _on_enableflicktb_changed(self):
        config.enableflicktb = self.var_enableflicktb.get()
        logger.info(f"Flick triggerbot enabled: {config.enableflicktb}")

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

    def _on_mode_selected(self, val):
        config.mode = val
        self.tracker.mode = val
        logger.info(f"Mode changed to: {val}")
        
        # Hide all mode-specific controls
        self.flick_frame.pack_forget()
        self.silent_frame.pack_forget()
        
        # Show corresponding controls based on mode
        if val == "Flick":
            self.flick_frame.pack(pady=10, fill="x", padx=10)
        elif val == "Silent":
            self.silent_frame.pack(pady=10, fill="x", padx=10)

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
            is_conn = self.receiver.is_connected()
            self.connected = is_conn
            if is_conn:
                self.status_label.configure(text=f"Connected to {self.selected_source}", text_color="green")
            else:
                self.status_label.configure(text="Disconnected", text_color="red")
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
