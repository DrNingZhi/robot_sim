import mujoco
import mink
import numpy as np
import time

class MinkModelG1:
    def __init__(self, model_file):
        self.model = mujoco.MjModel.from_xml_path(model_file)
        self.data = mujoco.MjData(self.model)
        self.mink_model = mink.Configuration(self.model)
        self.dt = self.model.opt.timestep
        self.dof = self.mink_model.nv
        self.joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(self.model.njnt)
        ]

        print(f"Load the file '{model_file}' successfully.")
        print(f"Dof of model: {self.dof}")
        print(f"Joint_names:")
        for j in self.joint_names:
            print(f" - {j}")
        
        self.solver = "daqp"
        
        self.tasks = []
        self.limits = []
        self.constraints = []

        self._setup_tasks()
        self._setup_limits()
        
        self.timer = 0
        self.solve_count = 0

    def _setup_tasks(self):
        self.tasks.append(
            mink.FrameTask(
                frame_name="wrist_l_marker",
                frame_type="body",
                position_cost=1.0,
                orientation_cost=1.0
            )
        )
        self.tasks.append(
            mink.FrameTask(
                frame_name="wrist_r_marker",
                frame_type="body",
                position_cost=1.0,
                orientation_cost=1.0
            )
        )
        self.tasks.append(
            mink.FrameTask(
                frame_name="foot_l_marker",
                frame_type="body",
                position_cost=1.0,
                orientation_cost=1.0
            )
        )
        self.tasks.append(
            mink.FrameTask(
                frame_name="foot_r_marker",
                frame_type="body",
                position_cost=1.0,
                orientation_cost=1.0
            )
        )
    
        self.tasks.append(
            mink.ComTask(cost=[1.0, 1.0, 0.0])
        )

        damping_cost = 0.1 * np.ones(self.dof)
        self.tasks.append(
            mink.DampingTask(self.model, damping_cost)
        )

        energy_task = mink.KineticEnergyRegularizationTask(cost=0.0001)
        energy_task.set_dt(self.dt)
        self.tasks.append(energy_task)

    def _setup_limits(self):
        self.limits.append(mink.ConfigurationLimit(self.model))

    def solve(self, q, frame_targets, com_target):
        if self.solve_count==0:
            self.timer = time.time()
        
        for i in range(len(frame_targets)):
            se3_target = mink.SE3.from_matrix(frame_targets[i])
            self.tasks[i].set_target(se3_target)
        
        self.mink_model.update(q)
        # mujoco.mj_step(self.mink_model.model, self.mink_model.data)
        
        com_task_id = len(frame_targets)
        self.tasks[com_task_id].set_target(com_target)

        try:
            vel = mink.solve_ik(
                self.mink_model,
                self.tasks, 
                self.dt, 
                self.solver, 
                limits=self.limits,
                constraints=self.constraints
            )
        except:
            print("Warning: solver error")
            vel = np.zeros(self.dof)
        self.solve_count += 1
        self.mink_model.integrate_inplace(vel, self.dt)

        if self.solve_count % 1000 == 0:
            mink_fps = int(self.solve_count / (time.time() - self.timer))
            print(f"mink fps: {mink_fps}")
        return self.mink_model.q.copy()