import carla
import cv2
import numpy as np
import time
import math

# ==============================================================================
# APEX ENGINE MATHEMATICS (Ported to Python/NumPy for CARLA integration)
# ==============================================================================

def rgb_to_gray_fixed_point(bgr):
    """1. Fixed-Point Grayscale"""
    b = bgr[:,:,0].astype(np.int32)
    g = bgr[:,:,1].astype(np.int32)
    r = bgr[:,:,2].astype(np.int32)
    return ((r * 77 + g * 150 + b * 29) >> 8).astype(np.uint8)

def perona_malik_diffusion(img, iters=2, lam=0.2, K=20.0):
    """2. PDE Anisotropic Diffusion"""
    img = img.astype(np.float32)
    K_sq = K * K
    for _ in range(iters):
        dN = np.zeros_like(img)
        dS = np.zeros_like(img)
        dE = np.zeros_like(img)
        dW = np.zeros_like(img)
        
        # Stencil differences
        dN[1:-1, 1:-1] = img[:-2, 1:-1] - img[1:-1, 1:-1]
        dS[1:-1, 1:-1] = img[2:, 1:-1] - img[1:-1, 1:-1]
        dE[1:-1, 1:-1] = img[1:-1, 2:] - img[1:-1, 1:-1]
        dW[1:-1, 1:-1] = img[1:-1, :-2] - img[1:-1, 1:-1]
        
        # Conductance
        cN = 1.0 / (1.0 + (dN**2) / K_sq)
        cS = 1.0 / (1.0 + (dS**2) / K_sq)
        cE = 1.0 / (1.0 + (dE**2) / K_sq)
        cW = 1.0 / (1.0 + (dW**2) / K_sq)
        
        img += lam * (cN*dN + cS*dS + cE*dE + cW*dW)
    return np.clip(img, 0, 255).astype(np.uint8)

# 3. MDP Setup
SPEED_SLOW, SPEED_FAST = 0, 1
FAR, MED, CLOSE = 0, 1, 2
ACCEL, MAINTAIN, BRAKE = 0, 1, 2

def mdp_next_state(dist, speed, action):
    next_speed = SPEED_FAST if action == ACCEL else (SPEED_SLOW if action == BRAKE else speed)
    next_dist = dist
    if next_speed == SPEED_FAST: next_dist += 1
    if action == BRAKE and dist > FAR: next_dist -= 1
    return max(0, min(2, next_dist)), next_speed

def mdp_reward(dist, speed, action):
    r = 0
    if dist == CLOSE and speed == SPEED_FAST: r -= 100
    if dist == CLOSE and action == BRAKE: r += 50
    if dist == FAR and action == ACCEL: r += 20
    return r

def mdp_value_iteration():
    V = np.zeros(6)
    policy = np.zeros(6, dtype=int)
    for _ in range(50):
        V_new = np.zeros(6)
        for d in range(3):
            for s in range(2):
                state = d * 2 + s
                max_q = -1e9
                for a in range(3):
                    nd, ns = mdp_next_state(d, s, a)
                    q = mdp_reward(d, s, a) + 0.9 * V[nd * 2 + ns]
                    if q > max_q:
                        max_q = q
                        policy[state] = a
                V_new[state] = max_q
        V = V_new
    return policy

mdp_policy = mdp_value_iteration()
bg_model = np.zeros((64, 64), dtype=np.float32)
frame_count = 0
maneuver_state = 'NORMAL'
maneuver_timer = 0

def process_frame(image, vehicle, carla_map):
    """CARLA Camera Callback Function"""
    global bg_model, frame_count, maneuver_state, maneuver_timer
    frame_count += 1
    
    # Extract BGRA raw pixels from CARLA
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    bgr = array[:, :, :3]
    
    # 1. Grayscale
    gray = rgb_to_gray_fixed_point(bgr)
    
    # 2. PDE Diffusion
    smoothed = perona_malik_diffusion(gray, iters=2, lam=0.2, K=20.0)
    
    # 3. SVD / EMA Background Subtraction
    small_gray = cv2.resize(smoothed, (64, 64)).astype(np.float32) / 255.0
    
    if frame_count == 1:
        bg_model = small_gray.copy()
    else:
        bg_model = 0.95 * bg_model + 0.05 * small_gray
        
    diff = np.abs(small_gray - bg_model)
    
    # Bottom half road analysis (Ignore shadows with higher pixel threshold)
    bottom_half = diff[32:, :]
    left_motion = np.sum(bottom_half[:, :21][bottom_half[:, :21] > 0.3])
    center_motion = np.sum(bottom_half[:, 21:42][bottom_half[:, 21:42] > 0.3])
    right_motion = np.sum(bottom_half[:, 42:][bottom_half[:, 42:] > 0.3])
    
    # 4. MDP Decision (Increased threshold for CARLA's high-res road textures)
    current_dist = CLOSE if center_motion > 20.0 else FAR
    current_speed = SPEED_FAST # Assume fast for simplification in script
    state = current_dist * 2 + current_speed
    chosen_action = mdp_policy[state]
    
    # Dynamic Evasion Logic
    turn_left = False
    turn_right = False
    if current_dist == CLOSE:
        if left_motion < 1.5 and left_motion <= right_motion:
            turn_left = True
        elif right_motion < 1.5:
            turn_right = True
            
    # Apply to CARLA Vehicle
    control = carla.VehicleControl()
    if chosen_action == ACCEL:
        control.throttle = 0.8
        control.brake = 0.0
    elif chosen_action == BRAKE:
        control.throttle = 0.0
        control.brake = 1.0
    else: # MAINTAIN
        control.throttle = 0.4
        control.brake = 0.0
        
    # --- NEW: CARLA Waypoint Steering ---
    waypoint = carla_map.get_waypoint(vehicle.get_location())
    next_wps = waypoint.next(4.0) # Look 4 meters ahead
    base_steer = 0.0
    if len(next_wps) > 0:
        wp_loc = next_wps[0].transform.location
        v_trans = vehicle.get_transform()
        v_loc = v_trans.location
        v_fwd = v_trans.get_forward_vector()
        
        # Vector to waypoint
        dx = wp_loc.x - v_loc.x
        dy = wp_loc.y - v_loc.y
        mag = math.sqrt(dx*dx + dy*dy)
        if mag > 0:
            dx /= mag
            dy /= mag
            
        # Cross product for steer direction (z-axis)
        cross_z = v_fwd.x * dy - v_fwd.y * dx
        base_steer = max(-1.0, min(1.0, cross_z * 1.5)) # Proportional steering
    else:
        # No waypoints ahead (End of road/wall) -> Trigger U-Turn
        base_steer = 0.0
        if maneuver_state == 'NORMAL':
            maneuver_state = 'UTURN'
            maneuver_timer = 80 # ~2.5 seconds of hard turning
            
    if turn_left:
        control.steer = max(-1.0, base_steer - 0.5) # Evasive swerve left
    elif turn_right:
        control.steer = min(1.0, base_steer + 0.5)  # Evasive swerve right
    else:
        control.steer = base_steer # Smooth road following
        
    # --- MANEUVER OVERRIDES (U-Turn & Anti-Stuck) ---
    if maneuver_state == 'UTURN':
        control.throttle = 0.5
        control.brake = 0.0
        control.steer = 1.0 # Hard right U-Turn
        maneuver_timer -= 1
        if maneuver_timer <= 0:
            maneuver_state = 'NORMAL'
            
    # Check if stuck (speed is near 0 but throttle is applied)
    vel = vehicle.get_velocity()
    speed = math.sqrt(vel.x**2 + vel.y**2)
    
    if speed < 0.1 and control.throttle > 0 and maneuver_state != 'REVERSE' and frame_count > 60:
        maneuver_state = 'REVERSE'
        maneuver_timer = 40 # ~1.3 seconds of reversing
        
    if maneuver_state == 'REVERSE':
        control.throttle = 0.6
        control.brake = 0.0
        control.steer = -1.0 # Steer left while reversing to pull away from wall
        control.reverse = True
        maneuver_timer -= 1
        if maneuver_timer <= 0:
            maneuver_state = 'NORMAL'
            control.reverse = False
            
    vehicle.apply_control(control)
    
    # Display view with Telemetry
    act_str = "ACCEL" if chosen_action == 0 else ("BRAKE" if chosen_action == 2 else "MAINTAIN")
    cv2.putText(smoothed, f"Motion: {center_motion:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2)
    cv2.putText(smoothed, f"Action: {act_str}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2)
    cv2.imshow("APEX Engine - CARLA View", smoothed)
    cv2.waitKey(1)

def main():
    print("Connecting to CARLA server...")
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    
        
    carla_map = world.get_map()
    blueprint_library = world.get_blueprint_library()
    
    # 1. Spawn Mini Cooper
    vehicle_bp = blueprint_library.find('vehicle.mini.cooper_s')
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    print(f"Spawned Vehicle: {vehicle.type_id}")
    
    # 2. Attach RGB Camera to vehicle hood
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_transform = carla.Transform(carla.Location(x=1.5, z=1.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    
    try:
        # Pass the camera data into your mathematical engine
        camera.listen(lambda image: process_frame(image, vehicle, world.get_map()))
        
        print("Engine running! Press Ctrl+C to exit.")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        camera.destroy()
        vehicle.destroy()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
