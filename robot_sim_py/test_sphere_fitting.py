from robot_sim.collision import SphereFittingCollision
import trimesh

model_file = "model/panda/meshes/link2.obj"
mesh = trimesh.load(model_file)
spheres = SphereFittingCollision(0, mesh)
spheres.cluster(50)

spheres.show(
    show_mesh=True,
    show_cluster=False,
    show_points=False,
    show_spheres=True,
)
