import numpy as np
import mujoco


def get_contact_force(model, data, geom_name, apply=True):
    # 获取作用在body_name上的作用力
    forces = []
    positions = []
    if data.ncon == 0:
        return forces, positions

    geom_id = model.geom(geom_name).id
    for i in range(data.ncon):
        if data.contact[i].geom1 == geom_id:
            p = data.contact[i].pos
            efc_address = data.contact[i].efc_address
            dim = data.contact[i].dim
            contact_force = data.efc_force[efc_address : (efc_address + dim)]
            if dim == 1:
                f = contact_force * data.contact[i].frame[:3]
            else:
                f = data.contact[i].frame.reshape(3, 3).T @ contact_force[0:3]
            forces.append(-f)
            positions.append(p)
        if data.contact[i].geom2 == geom_id:
            p = data.contact[i].pos
            efc_address = data.contact[i].efc_address
            dim = data.contact[i].dim
            contact_force = data.efc_force[efc_address : (efc_address + dim)]
            if dim == 1:
                f = contact_force * data.contact[i].frame[:3]
            else:
                f = data.contact[i].frame.reshape(3, 3).T @ contact_force[0:3]
            forces.append(f)
            positions.append(p)
    if apply:
        return np.array(forces), np.array(positions)
    else:
        return -np.array(forces), np.array(positions)


def show_contact_force(viewer, f, p, scale=0.1):
    viewer.user_scn.ngeom = 0
    num = 0
    for i in range(len(f)):
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[num],
            mujoco.mjtGeom.mjGEOM_LINE,
            np.zeros(3),
            np.zeros(3),
            np.zeros(9),
            np.array([0.0, 0.0, 1.0, 1.0]),
        )
        start_point = p[i]
        end_point = p[i] + scale * f[i]
        mujoco.mjv_connector(
            viewer.user_scn.geoms[num],
            mujoco.mjtGeom.mjGEOM_LINE,
            2.0,
            start_point,
            end_point,
        )
        num += 1
    viewer.user_scn.ngeom = num
