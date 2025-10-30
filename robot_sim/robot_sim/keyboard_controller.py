from pynput import keyboard
import threading
import numpy as np


            
class KeyboardController:
    def __init__(self):
        print("请通过键盘控制:")
        print("前：↑")
        print("后：↓")
        print("左：←")
        print("右：→")
        print("上：g")
        print("下：b")
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.step_size = 0.002

    def start(self):
        self.listener_thread = threading.Thread(target=self.start_listener)
        self.listener_thread.daemon = True
        self.listener_thread.start()
        
    def start_listener(self):
        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            listener.join()

    def on_press(self, key):
        try:
            if key == keyboard.Key.up:
                self.x += self.step_size
            elif key == keyboard.Key.down:
                self.x -= self.step_size
            elif key == keyboard.Key.left:
                self.y += self.step_size
            elif key == keyboard.Key.right:
                self.y -= self.step_size

            if key.char == "g":
                self.z += self.step_size
            elif key.char == "b":
                self.z -= self.step_size
                
        except AttributeError:
            pass  # 其他特殊按键
        
    def on_release(self, key):
        if key == keyboard.Key.esc:
            print("退出键盘监听")
            return False  # 停止监听

    def state(self):
        return np.array([self.x, self.y, self.z])