**robot_sim** is the code compilations of articles on Zhihu created by NingZhi: https://zhuanlan.zhihu.com/p/2018332504264820040

# Setup and Run examples

```
git clone git@github.com:DrNingZhi/robot_sim.git
cd robot_sim
```

## Run matlab examples

Open Matlab (version>=2021a), and run
| Script | Description (Link to Zhihu) |
| ---- | ---- |
| gait_plan_straight.m  | [机器人双足步态LIPM(直线)](https://zhuanlan.zhihu.com/p/603594171) |
| gait_plan_turn.m  | [机器人双足步态LIPM(转向)](https://zhuanlan.zhihu.com/p/603594171) |
| test_gait_mpc.m | [机器人双足步态MPC](https://zhuanlan.zhihu.com/p/603594171) |
| gait_mpc_generate.m; model/gait_model.slx  | [机器人双足步态MPC](https://zhuanlan.zhihu.com/p/603594171) |
| test_invdyn.m |  [机械臂逆动力学](https://zhuanlan.zhihu.com/p/624913809)  |
| test_opt_complex.m |  [优化算法-复合形法](https://zhuanlan.zhihu.com/p/553567636)  |
| test_opt_pso.m | [优化算法-多目标粒子群](https://zhuanlan.zhihu.com/p/554143499)  |
| lqr_continuous.m |  [LQR控制(连续)](https://zhuanlan.zhihu.com/p/22105893562)  |
| lqr_discrete.m |  [LQR控制(离散)](https://zhuanlan.zhihu.com/p/22105893562)  |
| mpc_demo.m | [MPC控制](https://zhuanlan.zhihu.com/p/22430677907)  |



## Run python examples

Firstly, install required python lib
```
pip install -r requirements.txt
cd robot_sim_py
```

Then, run the example:

|  Script   | Description (Link to Zhihu) |
|  ----  | ----  |
|ur5_gen_cir_traj.py | [机械臂运动规划](https://zhuanlan.zhihu.com/p/2022059758941734390) |
|ur5_joint_pd.py | [机械臂关节空间pd跟踪控制](https://zhuanlan.zhihu.com/p/1946345242811929221) |
|ur5_joint_grav_comp.py| [机械臂关节空间重力补偿控制](https://zhuanlan.zhihu.com/p/1946345242811929221) |
|ur5_joint_dyn_feed.py| [机械臂关节空间动力学补偿控制](https://zhuanlan.zhihu.com/p/1946345242811929221) |
|ur5_ee_jac_inv.py | [机械臂任务空间逆雅可比控制](https://zhuanlan.zhihu.com/p/1947982685357217203) |
|ur5_ee_jacT.py | [机械臂任务空间转置雅可比控制](https://zhuanlan.zhihu.com/p/1947982685357217203) |
|ur5_ee_jacT_grav.py | [机械臂任务空间转置雅可比+重力补偿控制](https://zhuanlan.zhihu.com/p/1947982685357217203) |
|ur5_ee_jacT_dyn.py | [机械臂任务空间转置雅可比+动力学补偿控制](https://zhuanlan.zhihu.com/p/1947982685357217203) |
|ur5_force.py | [机械臂末端力控制](https://zhuanlan.zhihu.com/p/1953863647915934385) |
|ur5_force_pos1.py | [机械臂末端力位混合控制](https://zhuanlan.zhihu.com/p/1953863647915934385) |
|ur5_force_pos2.py | [机械臂末端力位混合控制](https://zhuanlan.zhihu.com/p/1953863647915934385) |
|ur5_admittance.py | [机械臂末端导纳控制](https://zhuanlan.zhihu.com/p/1967358941055915966) |
|test_collision.py | [碰撞检测(待更新)](https://zhuanlan.zhihu.com/p/1977502768504792829) |
|待更新.py | [碰撞检测(待更新)](https://zhuanlan.zhihu.com/p/1985101999377760977) |
|panda_null_space.py | [机械臂零空间运动](https://zhuanlan.zhihu.com/p/1992710728243758668) |
|panda_null_space_obstacle.py | [机械臂零空间避障(待更新)](https://zhuanlan.zhihu.com/p/1997786441699304459) |
|test_Astar.py | [避障轨迹规划A*算法](https://zhuanlan.zhihu.com/p/1960099548056781292) |





