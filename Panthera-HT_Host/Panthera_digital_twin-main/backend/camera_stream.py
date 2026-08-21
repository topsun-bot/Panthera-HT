"""Lazy RealSense D435i color stream for the Host digital-twin panel."""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


JPEG_QUALITY = 70
TARGET_FPS = 15
COLOR_ATTEMPTS = (
    (640, 480, 15),
    (640, 480, 30),
    (1280, 720, 15),
)


class CameraHub:
    def __init__(self):
        self._lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._pipeline = None
        self._thread = None
        self._running = False
        self._clients = 0
        self._jpeg = None
        self._serial = None
        self._error = None
        self._size = (0, 0)

    def status(self):
        with self._lock:
            return {
                "available": rs is not None,
                "running": self._running,
                "clients": self._clients,
                "serial": self._serial,
                "width": self._size[0],
                "height": self._size[1],
                "error": self._error,
            }

    def acquire(self):
        with self._lock:
            if rs is None:
                self._error = "pyrealsense2 is not installed"
                return False
            self._clients += 1
            if self._running:
                return True
            try:
                self._start_locked()
                return True
            except Exception as exc:
                self._clients = max(0, self._clients - 1)
                self._error = str(exc)
                self._stop_pipeline()
                return False

    def release(self):
        with self._lock:
            self._clients = max(0, self._clients - 1)
            if self._clients == 0 and self._running:
                self._running = False
                thread = self._thread
            else:
                thread = None
        if thread is not None:
            thread.join(timeout=2.0)
            with self._lock:
                self._stop_pipeline()

    def mjpeg_bytes(self):
        with self._frame_lock:
            return self._jpeg

    def _start_locked(self):
        devices = list(rs.context().query_devices())
        if not devices:
            raise RuntimeError("D435i not found. Check USB3 and reconnect the camera.")

        last_err = None
        for width, height, fps in COLOR_ATTEMPTS:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            try:
                profile = pipeline.start(config)
                self._pipeline = pipeline
                self._serial = profile.get_device().get_info(rs.camera_info.serial_number)
                self._size = (width, height)
                last_err = None
                break
            except Exception as exc:
                last_err = exc
                try:
                    pipeline.stop()
                except Exception:
                    pass
        if self._pipeline is None:
            raise RuntimeError(f"Failed to open D435i: {last_err}")

        for _ in range(10):
            self._pipeline.wait_for_frames(1000)

        self._error = None
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print(f"[camera] D435i started serial={self._serial} {self._size[0]}x{self._size[1]}")

    def _stop_pipeline(self):
        pipeline = self._pipeline
        self._pipeline = None
        self._running = False
        self._thread = None
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass
            print("[camera] D435i stopped")

    def _capture_loop(self):
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        interval = 1.0 / TARGET_FPS
        while self._running and self._pipeline is not None:
            loop_start = time.time()
            try:
                frames = self._pipeline.wait_for_frames(1000)
                color = frames.get_color_frame()
                if not color:
                    continue
                image = np.asanyarray(color.get_data())
                ok, buf = cv2.imencode(".jpg", image, encode_params)
                if ok:
                    with self._frame_lock:
                        self._jpeg = buf.tobytes()
            except Exception as exc:
                self._error = str(exc)
                time.sleep(0.2)
            sleep_time = interval - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)


camera_hub = CameraHub()


def mjpeg_generator():
    if not camera_hub.acquire():
        yield (
            b"--frame\r\nContent-Type: text/plain\r\n\r\n"
            + (camera_hub.status().get("error") or "camera unavailable").encode("utf-8")
            + b"\r\n"
        )
        return

    try:
        while True:
            jpeg = camera_hub.mjpeg_bytes()
            if jpeg:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode("ascii") + b"\r\n\r\n"
                    + jpeg
                    + b"\r\n"
                )
            time.sleep(1.0 / TARGET_FPS)
    finally:
        camera_hub.release()
