## 1. robotic arm control

Firstly, generate the trajectory to files for convenience.
```
python generate_trajectory.py
```

### (1) Joint position control

see https://zhuanlan.zhihu.com/p/1946345242811929221

Joint pd position:

```
python sim_arm_pos_pd.py
```

Joint pd position with gravity compensation:

```
python sim_arm_pos_pd_grav_comp.py
```

Joint pd position with dynamic feedforward:

```
python sim_arm_pos_dyn_feed.py
```

### (2) Cartesian position control

see https://zhuanlan.zhihu.com/p/1947982685357217203

End-effector Cartesian space control by inverse Jacobian method:
```
python sim_arm_ee_pos_jac_inv.py
```

End-effector Cartesian space control by transposed Jacobian method:
```
python sim_arm_ee_pos_jacT.py
```

End-effector Cartesian space control by transposed Jacobian method with gravity compensation:
```
python sim_arm_ee_pos_jacT_grav.py
```

End-effector Cartesian space control by transposed Jacobian method with dynamic feedforward:
```
python sim_arm_ee_pos_jacT_dyn.py
```

### (3) Force and position hybrid control

see https://zhuanlan.zhihu.com/p/1953863647915934385

End-effector force control, applying on a static block.
```
python sim_arm_force.py
```

End-effector force-position hybrid control. Apply constant force on block, and move a circle trajectory.

```
python sim_arm_force_pos1.py
```

End-effector force-position hybrid control. Apply constant force on a movable block, and move a circle trajectory.

```
python sim_arm_force_pos2.py
```