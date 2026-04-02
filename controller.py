from dm_control.mujoco import Camera as DmCamera

# write the same loop as before but with the new controller
timestep = 1e-4
run_time = 10.0

np.random.seed(0)
arena = OdorPlumeArena(
    output_dir / "plume_tcropped.hdf5",
    plume_simulation_fps=800,
    dimension_scale_factor=0.25,
)

# Define the fly
contact_sensor_placements = [
    f"{leg}{segment}"
    for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
    for segment in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
]

fly = Fly(
    enable_adhesion=True,
    draw_adhesion=True,
    enable_olfaction=True,
    enable_vision=False,
    contact_sensor_placements=contact_sensor_placements,
    # Here the opposite spawn position can be tried (65.0, 15.0, 0.25)
    spawn_pos=(65.0, 45.0, 0.25),
    spawn_orientation=(0, 0, -np.pi),
)

wind_dir = [1.0, 0.0]
ctrl = SimplePlumeNavigationController(timestep, wind_dir=wind_dir)

cam_params = {"mode":"fixed",
    "pos": (
                0.50 * arena.arena_size[0],
                0.15 * arena.arena_size[1],
                1.00 * arena.arena_size[1],
            ),
    "euler":(np.deg2rad(15), 0, 0), "fovy":60}

cam = Camera(
    attachment_point=arena.root_element.worldbody,
    camera_name=main_camera_name,
    timestamp_text = False,
    camera_parameters=cam_params
)

dm_cam = DmCamera(
    sim.physics,
    camera_id=cam.camera_id,
    width=cam.window_size[0],
    height=cam.window_size[1],
)
camera_matrix = dm_cam.matrix
arena_inflow_pos = np.array(inflow_pos) / arena.dimension_scale_factor * smoke_grid_size
target_inflow_radius = 5.0
inflow_x, inflow_y = get_inflow_circle(
    arena_inflow_pos,
    target_inflow_radius,
    camera_matrix,
)

sim = PlumeNavigationTask(
    fly=fly,
    arena=arena,
    cameras=[cam],
)

walking_icons = get_walking_icons()

obs, info = sim.reset(0)

for i in trange(np.rint(run_time / timestep).astype(int)):
    fly_orientation = obs["fly_orientation"][:2]
    fly_orientation /= np.linalg.norm(fly_orientation)
    close_to_boundary = is_close_to_boundary(obs["fly"][0][:2], arena.arena_size)
    dn_drive = ctrl.step(
        fly_orientation, obs["odor_intensity"], close_to_boundary, sim.curr_time
    )

    obs, reward, terminated, truncated, info = sim.step(dn_drive)

    icon = walking_icons[ctrl.curr_state][:, :, :3]
    rendered_img = sim.render()[0]
    rendered_img = render_overlay(
        rendered_img,
        ctrl.accumulated_evidence,
        fly_orientation,
        ctrl.target_angle,
        to_probability(ctrl.upwind_success),
        icon,
        cam.window_size,
        inflow_x,
        inflow_y,
    )

    if rendered_img is not None:
        cam._frames[-1] = rendered_img

    if np.linalg.norm(obs["fly"][0][:2] - arena_inflow_pos) < target_inflow_radius:
        print("The fly reached the inflow")
        break
    elif truncated:
        print("The fly went out of bound")
        break

    obs_list.append(obs)
