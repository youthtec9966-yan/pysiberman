import ctypes
import sys
import time
from pathlib import Path


class WindowContextManager:
    """
    Manages the context of the currently active window to allow for seamless switching
    between the digital human and other applications (e.g., video players).
    It handles:
    1. Capturing the current active window handle.
    2. Pausing media playback (via global media keys).
    3. Exiting fullscreen mode (via Esc key).
    4. Restoring the previous window to the foreground.
    5. Restoring fullscreen mode (via F11 key).
    6. Resuming media playback.
    """
    def __init__(self):
        self.last_hwnd = None
        self.user32 = ctypes.windll.user32
        
        # Virtual Key Codes
        self.VK_MEDIA_PLAY_PAUSE = 0xB3
        self.VK_SPACE = 0x20
        self.VK_ESCAPE = 0x1B
        self.VK_F11 = 0x7A 
        self.VK_RETURN = 0x0D
        self.VK_MENU = 0x12 # Alt key
        
        # Event Flags
        self.KEYEVENTF_KEYUP = 0x0002

    def _send_key(self, vk_code):
        """Simulate a single key press and release."""
        self.user32.keybd_event(vk_code, 0, 0, 0)
        self.user32.keybd_event(vk_code, 0, self.KEYEVENTF_KEYUP, 0)

    def capture_context(self):
        """
        Capture the current foreground window, pause media, and exit fullscreen.
        Must be called BEFORE the digital human window takes focus.
        """
        self.last_hwnd = self.user32.GetForegroundWindow()
        
        # Only capture if it's a valid window and not our own tray app (simplistic check)
        if not self.last_hwnd:
            return

        print(f"[WindowManager] Captured hwnd: {self.last_hwnd}")

        # 1. Pause Media (Try Space, then Media Key as backup)
        # Many web players use Space for play/pause
        self._send_key(self.VK_SPACE)
        # self._send_key(self.VK_MEDIA_PLAY_PAUSE)
        
        # 2. Exit Fullscreen (Try ESC)
        # Most players/browsers exit fullscreen with Esc.
        self._send_key(self.VK_ESCAPE)

    def restore_context(self):
        """
        Restore the previously captured window, re-enter fullscreen, and resume media.
        """
        if not self.last_hwnd:
            print("[WindowManager] No context to restore.")
            return

        print(f"[WindowManager] Restoring hwnd: {self.last_hwnd}")
        
        # 1. Restore Window State (if minimized)
        if self.user32.IsIconic(self.last_hwnd):
             self.user32.ShowWindow(self.last_hwnd, 9) # SW_RESTORE

        # 2. Bring to Foreground
        # Note: This might fail if the OS blocks it, but since we are the foreground process now,
        # handing off focus usually works.
        self.user32.SetForegroundWindow(self.last_hwnd)
        
        # Give a tiny delay for focus to switch
        time.sleep(0.2)
        
        # 3. Resume Media (Space) - Moved before fullscreen as requested
        self._send_key(self.VK_SPACE)
        
        time.sleep(0.2)

        # 4. Enter Fullscreen (Enter) - Changed from F11 to Enter as requested
        self._send_key(self.VK_RETURN)
        
        # Reset
        self.last_hwnd = None


def get_resource_path(relative_path: str) -> Path:
    """
    获取资源的绝对路径。
    这个函数是为了兼容开发环境和 PyInstaller 打包后的环境。
    PyInstaller 会把资源解压到一个临时目录 (sys._MEIPASS)。
    """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = Path.cwd()
        if getattr(sys, "frozen", False):
            base_path = Path(sys.executable).parent
            if (base_path / "_internal").exists():
                base_path = base_path / "_internal"
    p = Path(base_path) / relative_path
    if not p.exists():
        if (Path.cwd() / relative_path).exists():
            return Path.cwd() / relative_path
    return p
