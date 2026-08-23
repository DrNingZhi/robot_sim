import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from .keyboard_controller import KeyboardController2


class MarkerController:
    # 仅用于键盘控制两个marker点，作为机械臂末端跟踪位置，测试机械臂末端追踪性能
    def __init__(self, body_name1, body_name2):
        self.body_name1 = body_name1
        self.body_name2 = body_name2
        self.body_id1 = 0
        self.body_id2 = 0    
        self.init_pos1 = np.zeros(3)
        self.init_pos2 = np.zeros(3)
        self.init_quat1 = np.zeros(4)
        self.init_quat2 = np.zeros(4)
        self.pos1 = np.zeros(3)
        self.pos2 = np.zeros(3)
        self.quat1 = np.zeros(4)
        self.quat2 = np.zeros(4)
        
        self.keyboard_controller = KeyboardController2(2)
        self.keyboard_controller_started = False
        
    
    def step(self, model):
        if not self.keyboard_controller_started:
            # 获取初始的姿态
            self.body_id1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.body_name1)
            self.body_id2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.body_name2)
            self.init_pos1 = model.body_pos[self.body_id1].copy()
            self.init_pos2 = model.body_pos[self.body_id2].copy()
            self.init_quat1 = model.body_quat[self.body_id1].copy()
            self.init_quat2 = model.body_quat[self.body_id2].copy()
            self.keyboard_controller.start()
            self.keyboard_controller_started = True
        
        states = self.keyboard_controller.get_state()
        
        self.pos1 = self.init_pos1 + states[0, :3]
        self.pos2 = self.init_pos2 + states[1, :3]
        self.quat1 = self.rotation_superposition(self.init_quat1, states[0, 3:])
        self.quat2 = self.rotation_superposition(self.init_quat2, states[1, 3:])
        
        model.body_pos[self.body_id1] = self.pos1.copy()
        model.body_pos[self.body_id2] = self.pos2.copy()
        model.body_quat[self.body_id1] = self.quat1.copy()
        model.body_quat[self.body_id2] = self.quat2.copy()
        
    @staticmethod
    def rotation_superposition(quat, rpy):
        R0 = Rotation.from_quat(quat, scalar_first=True)
        R1 = Rotation.from_euler("XYZ", rpy)
        R = R0 * R1
        return R.as_quat(scalar_first=True)

    def get_transforms(self, model):
        pos1 = model.body_pos[self.body_id1].copy()
        pos2 = model.body_pos[self.body_id2].copy()
        quat1 = model.body_quat[self.body_id1].copy()
        quat2 = model.body_quat[self.body_id2].copy()
        
        T1 = np.eye(4)
        T1[:3, 3] = pos1
        T1[:3, :3] = Rotation.from_quat(quat1, scalar_first=True).as_matrix()
        T2 = np.eye(4)
        T2[:3, 3] = pos2
        T2[:3, :3] = Rotation.from_quat(quat2, scalar_first=True).as_matrix()
        
        return [T1, T2]
    