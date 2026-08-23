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

class KeyboardController2:
    def __init__(self, state_num):
        print("请通过键盘控制:")
        print("前: 1  (后: Ctrl+1)")
        print("左: 2  (右: Ctrl+2)")
        print("上: 3  (下: Ctrl+3)")
        print("rot_x+: 4  (rot_x-: Ctrl+4)")
        print("rot_y+: 5  (rot_y-: Ctrl+5)")
        print("rot_z+: 6  (rot_z-: Ctrl+6)")
        print("按 Shift 键切换对象")
        
        self.states = np.zeros((state_num, 6))
        self.step_size = np.array([0.002, 0.002, 0.002, 0.02, 0.02, 0.02])
        self.ctrl_pressed = False
        self.state = 0
        print(f"\n当前为第{self.state}个状态")

    def start(self):
        self.listener_thread = threading.Thread(target=self.start_listener)
        self.listener_thread.daemon = True
        self.listener_thread.start()
        
    def start_listener(self):
        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            listener.join()

    def on_press(self, key):
        try:
            if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
                self.ctrl_pressed = True

            if key == keyboard.Key.shift_l or key == keyboard.Key.shift_r:
                self.state += 1
                if self.state==len(self.states):
                    self.state = 0
                print(f"转化为控制第{self.state}个状态")
            
            if key.char == '1':
                if not self.ctrl_pressed:
                    self.states[self.state, 0] += self.step_size[0]
                else:
                    self.states[self.state, 0] -= self.step_size[0]
            elif key.char == '2':
                if not self.ctrl_pressed:
                    self.states[self.state, 1] += self.step_size[1]
                else:
                    self.states[self.state, 1] -= self.step_size[1]
            elif key.char == '3':
                if not self.ctrl_pressed:
                    self.states[self.state, 2] += self.step_size[2]
                else:
                    self.states[self.state, 2] -= self.step_size[2]
            elif key.char == '4':
                if not self.ctrl_pressed:
                    self.states[self.state, 3] += self.step_size[3]
                else:
                    self.states[self.state, 3] -= self.step_size[3]
            elif key.char == '5':
                if not self.ctrl_pressed:
                    self.states[self.state, 4] += self.step_size[4]
                else:
                    self.states[self.state, 4] -= self.step_size[4]
            elif key.char == '6':
                if not self.ctrl_pressed:
                    self.states[self.state, 5] += self.step_size[5]
                else:
                    self.states[self.state, 5] -= self.step_size[5]
        except AttributeError:
            pass  # 其他特殊按键
        
    def on_release(self, key):
        if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
            self.ctrl_pressed = False
        elif key == keyboard.Key.esc:
            print("退出键盘监听")
            return False  # 停止监听

    def get_state(self):
        return self.states.copy()