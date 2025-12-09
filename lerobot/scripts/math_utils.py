import numpy as np

def ee_to_joints(ee_actions: np.ndarray, seed_joints: np.ndarray = None) -> np.ndarray:
    """
    Convert end-effector actions to joint actions using inverse kinematics.
    
    Args:
        ee_actions: (chunk_size, 14) array of EE actions
                    Each timestep: [left_ee(7), right_ee(7)]
        seed_joints: (14,) initial joint positions for IK seeding
    
    Returns:
        joint_actions: (chunk_size, 14) array of joint actions
    """
    chunk_size = ee_actions.shape[0]
    joint_actions = np.zeros((chunk_size, 14), dtype=np.float32)
    
    # Initialize seeds
    if seed_joints is not None:
        left_seed = seed_joints[:6].tolist()  # First 6 joints only (no gripper)
        right_seed = seed_joints[7:13].tolist()
    else:
        # Default seed
        left_seed = [0.0, -0.96, 1.16, 0.0, -0.3, 0.0]
        right_seed = [0.0, -0.96, 1.16, 0.0, -0.3, 0.0]
    
    # Last valid solutions (for fallback on IK failure)
    last_left_joints = np.array(left_seed)
    last_right_joints = np.array(right_seed)
    
    print(f"Starting IK conversion for {chunk_size} timesteps...")
    
    for t in range(chunk_size):
        # Extract EE poses for this timestep
        left_ee = ee_actions[t, :7]  # [x, y, z, roll, pitch, yaw, gripper]
        right_ee = ee_actions[t, 7:14]
        
        # Split pose and gripper
        left_pose = left_ee[:6]  # [x, y, z, roll, pitch, yaw]
        left_gripper = left_ee[6]
        right_pose = right_ee[:6]
        right_gripper = right_ee[6]
        
        # Solve IK for left arm
        try:
            left_joints = calculate_inverse_kinematics(left_pose.tolist(), seed=left_seed)
            left_joints = np.array(left_joints)
            last_left_joints = left_joints  # Update last valid solution
            left_seed = left_joints.tolist()  # Update seed for next iteration
        except Exception as e:
            print(f"IK failed for left arm at timestep {t}: {e}. Using last valid joints.")
            left_joints = last_left_joints  # Use last valid solution
        
        # Solve IK for right arm
        try:
            right_joints = calculate_inverse_kinematics(right_pose.tolist(), seed=right_seed)
            right_joints = np.array(right_joints)
            last_right_joints = right_joints  # Update last valid solution
            right_seed = right_joints.tolist()  # Update seed for next iteration
        except Exception as e:
            print(f"IK failed for right arm at timestep {t}: {e}. Using last valid joints.")
            right_joints = last_right_joints  # Use last valid solution
        
        # Combine joints + gripper
        joint_actions[t, :6] = left_joints
        joint_actions[t, 6] = left_gripper
        joint_actions[t, 7:13] = right_joints
        joint_actions[t, 13] = right_gripper
        
        if t % 10 == 0:
            print(f"  IK progress: {t+1}/{chunk_size}")
    
    print(f"IK conversion complete: {chunk_size} timesteps")
    
    return joint_actions

def aloha_calculate_forward_kinematics_quaternion(joint_positions):
    """
    Calculate forward kinematics for the ALOHA VX300S robot arm.
    
    Args:
        joint_positions: List of 14 joint angles:
        [l_waist, l_shoulder, l_elbow, l_forearm_roll, l_wrist_angle, l_wrist_rotate, l_gripper, 
         r_waist, r_shoulder, r_elbow, r_forearm_roll, r_wrist_angle, r_wrist_rotate, r_gripper]

    reurns:
        array: [left_end_effector_pose (7,), left_gripper, right_end_effector_pose (7,), right_gripper]
    """
    # Split joint positions into left and right arms
    left_joints = joint_positions[:7]
    right_joints = joint_positions[7:14]
    left_ee = calculate_forward_kinematics_quaternion(left_joints)
    right_ee = calculate_forward_kinematics_quaternion(right_joints)
    left_gripper = joint_positions[6]
    right_gripper = joint_positions[13]
    
    return np.concatenate([left_ee, [left_gripper], right_ee, [right_gripper]])

def calculate_forward_kinematics(joint_positions):
    """
    Calculate forward kinematics for the ALOHA VX300S robot arm.
    
    Args:
        joint_positions: List of 6 joint angles [waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate]
        
    Returns:
        tuple: (position, rotation_matrix) of the end effector
               position: [x, y, z] in meters
               rotation_matrix: 3x3 rotation matrix
    """
    
    # Joint angles (first 6 joints only)
    j = joint_positions[:6]
    
    # DH-like parameters extracted from URDF (all distances in meters)
    # Base to waist: z = 0.079
    # Waist to shoulder: z = 0.04805  
    # Shoulder to elbow: x = 0.05955, z = 0.3
    # Elbow to forearm_roll: x = 0.2
    # Forearm_roll to wrist_angle: x = 0.1
    # Wrist_angle to wrist_rotate: x = 0.069744
    # Wrist_rotate to end effector: x = 0.042825 + 0.005675 = 0.0485
    
    def rotation_matrix_z(angle):
        """Rotation matrix around Z axis"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
    def rotation_matrix_y(angle):
        """Rotation matrix around Y axis"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    
    def rotation_matrix_x(angle):
        """Rotation matrix around X axis"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    
    def transform_matrix(rotation, translation):
        """Create 4x4 transformation matrix"""
        T = np.eye(4)
        T[:3, :3] = rotation
        T[:3, 3] = translation
        return T
    
    # Initialize transformation matrix (identity)
    T = np.eye(4)
    
    # Base to waist (fixed translation + waist rotation)
    T_base_waist = transform_matrix(
        rotation_matrix_z(j[0]),  # waist rotation around Z
        [0, 0, 0.079]  # base height
    )
    T = T @ T_base_waist
    
    # Waist to shoulder (fixed translation + shoulder rotation)
    T_waist_shoulder = transform_matrix(
        rotation_matrix_y(j[1]),  # shoulder rotation around Y
        [0, 0, 0.04805]  # shoulder height offset
    )
    T = T @ T_waist_shoulder
    
    # Shoulder to elbow (translation + elbow rotation)
    T_shoulder_elbow = transform_matrix(
        rotation_matrix_y(j[2]),  # elbow rotation around Y
        [0.05955, 0, 0.3]  # shoulder to elbow offset
    )
    T = T @ T_shoulder_elbow
    
    # Elbow to forearm_roll (translation + forearm_roll rotation)
    T_elbow_forearm = transform_matrix(
        rotation_matrix_x(j[3]),  # forearm_roll rotation around X
        [0.2, 0, 0]  # elbow to forearm offset
    )
    T = T @ T_elbow_forearm
    
    # Forearm_roll to wrist_angle (translation + wrist_angle rotation)
    T_forearm_wrist = transform_matrix(
        rotation_matrix_y(j[4]),  # wrist_angle rotation around Y
        [0.1, 0, 0]  # forearm to wrist offset
    )
    T = T @ T_forearm_wrist
    
    # Wrist_angle to wrist_rotate (translation + wrist_rotate rotation)
    T_wrist_rotate = transform_matrix(
        rotation_matrix_x(j[5]),  # wrist_rotate rotation around X
        [0.069744, 0, 0]  # wrist angle to rotate offset
    )
    T = T @ T_wrist_rotate
    
    # Wrist_rotate to end effector (fixed translation)
    T_rotate_ee = transform_matrix(
        np.eye(3),  # no rotation
        [0.0485, 0, 0]  # final offset to end effector
    )
    T = T @ T_rotate_ee
    
    # Extract position and rotation
    position = T[:3, 3]
    rotation_matrix = T[:3, :3]

    return position, rotation_matrix

def calculate_forward_kinematics_rpy(joint_positions):

    position, rotation_matrix = calculate_forward_kinematics(joint_positions)
    rpy = convert_rot_matrix_to_rpy(rotation_matrix)
    
    return np.array([position[0], position[1], position[2], rpy[0], rpy[1], rpy[2]])

def calculate_forward_kinematics_quaternion(joint_positions):
    position, rotation_matrix = calculate_forward_kinematics(joint_positions)
    # Convert rotation matrix to quaternion
    qw = np.sqrt(1 + rotation_matrix[0, 0] + rotation_matrix[1, 1] + rotation_matrix[2, 2]) / 2
    qx = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / (4 * qw)
    qy = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / (4 * qw)
    qz = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / (4 * qw)
    
    return np.array([position[0], position[1], position[2], qx, qy, qz, qw])

def convert_rot_matrix_to_rpy(rotation_matrix):
    """
    Convert a rotation matrix to roll, pitch, yaw angles.
    
    Args:
        rotation_matrix: 3x3 rotation matrix
    
    Returns:
        tuple: (roll, pitch, yaw) in radians
    """
    assert rotation_matrix.shape == (3, 3), "Input must be a 3x3 rotation matrix"
    
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    
    singular = sy < 1e-6
    
    if not singular:
        x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = 0
    
    return x, y, z

def convert_rpy_to_rot_matrix(roll, pitch, yaw):
    """
    Convert roll, pitch, yaw (Euler angles) to a 3x3 rotation matrix.
    
    Args:
        roll (float): Rotation around x-axis in radians
        pitch (float): Rotation around y-axis in radians  
        yaw (float): Rotation around z-axis in radians
        
    Returns:
        np.ndarray: 3x3 rotation matrix
    """
    # Rotation matrix around x-axis (roll)
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    # Rotation matrix around y-axis (pitch)
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    # Rotation matrix around z-axis (yaw)
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation matrix (order: R_z * R_y * R_x)
    R = R_z @ R_y @ R_x
    
    return R


def enforce_angle_limits(joint_angles):
    """
    Enforce joint angle limits for ALOHA VX300S robot arm.

    This is meant to correct errors in the ik solver.

    To do this the calculated joint angles are fed into this function which
    checks to see if a joint angle is outside of its limits. If it is, we either 
    subtract or add 2*pi*N to bring it within the limits.

    if the angle is still outside the limits after this correction, we return the 
    joint angle uncorrected as it originally came in.

    Args:
        joint_angles: List or array of exactly 6 joint angles

    Joint 'waist' limits: [-3.141582727432251, 3.141582727432251]
    Joint 'shoulder' limits: [-1.8500490188598633, 1.2566370964050293]
    Joint 'elbow' limits: [-1.7627825736999512, 1.6057028770446777]
    Joint 'forearm_roll' limits: [-3.141582727432251, 3.141582727432251]
    Joint 'wrist_angle' limits: [-1.8675023317337036, 2.2340214252471924]
    Joint 'wrist_rotate' limits: [-3.141582727432251, 3.141582727432251]
    """
    if len(joint_angles) != 6:
        raise ValueError(f"joint_angles must have exactly 6 elements, got {len(joint_angles)}")
    
    joint_names = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
    limits = {}
    limits[joint_names[0]] = (-3.141582727432251, 3.141582727432251)
    limits[joint_names[1]] = (-1.8500490188598633, 1.2566370964050293)
    limits[joint_names[2]] = (-1.7627825736999512, 1.6057028770446777)
    limits[joint_names[3]] = (-3.141582727432251, 3.141582727432251)
    limits[joint_names[4]] = (-1.8675023317337036, 2.2340214252471924)
    limits[joint_names[5]] = (-3.141582727432251, 3.141582727432251)

    corrected_angles = []
    for i, angle in enumerate(joint_angles):
        joint_name = joint_names[i]
        lower_limit, upper_limit = limits[joint_name]
        corrected_angle = angle

        # Adjust angle by adding or subtracting 2*pi*N to bring it within limits
        while corrected_angle < lower_limit:
            corrected_angle += 2 * np.pi
        while corrected_angle > upper_limit:
            corrected_angle -= 2 * np.pi

        # If still outside limits, revert to original angle
        if corrected_angle < lower_limit or corrected_angle > upper_limit:
            corrected_angle = angle

        corrected_angles.append(corrected_angle)
    
    return corrected_angles

def calculate_inverse_kinematics(
    desired_pose,
    seed=None,
    max_iter=100,
    tol=1e-3,#1e-4,
    alpha=0.2 #0.2
):
    """
    Analytic (with fallback to numerical) inverse kinematics for ALOHA VX300S 6-DoF arm.
    Args:
        desired_pose: [x, y, z, roll, pitch, yaw] (meters, radians)
        seed: Optional initial guess for joint angles (list of 6 floats)
        max_iter: Max iterations for numerical fallback
        tol: Tolerance for convergence
        alpha: Step size for numerical fallback
    Returns:
        joint_positions: List of 6 joint angles [waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate]
    """
    # Analytic IK for 6-DoF arm with spherical wrist
    # This is a simplified version and may not cover all edge cases
    # If analytic fails, fallback to numerical (gradient descent)

    # DH parameters (from FK)
    d1 = 0.079 + 0.04805  # base to shoulder
    a2 = 0.05955
    d2 = 0.3
    a3 = 0.2
    a4 = 0.1
    a5 = 0.069744
    a6 = 0.0485

    # Desired end effector
    px, py, pz, roll, pitch, yaw = desired_pose
    R06 = convert_rpy_to_rot_matrix(roll, pitch, yaw)

    # Use seed or default
    if seed is None:
        q = np.array([0.0, -0.96, 1.16, 0.0, -0.3, 0.0])
    else:
        q = np.array(seed)

    # Compute wrist center
    p_ee = np.array([px, py, pz])
    wc = p_ee - R06 @ np.array([a6, 0, 0])

    # 1. Solve for waist (q1)
    q1 = np.arctan2(wc[1], wc[0])

    # 2. Shoulder/elbow (q2, q3)
    # Project wrist center into base frame
    r = np.sqrt(wc[0]**2 + wc[1]**2) - a2
    s = wc[2] - d1
    D = (r**2 + s**2 - d2**2 - a3**2) / (2 * d2 * a3)
    if np.abs(D) > 1.0:
        # Out of reach, fallback to numerical
        print("FALLING BACK TO NUMERICAL IK - 1!")
        _num_solution = _ik_numerical(desired_pose, q, max_iter, tol, alpha)
        return enforce_angle_limits(_num_solution)
    q3 = np.arctan2(-np.sqrt(1 - D**2), D)  # elbow-down
    # Law of cosines for q2
    phi1 = np.arctan2(s, r)
    phi2 = np.arctan2(a3 * np.sin(q3), d2 + a3 * np.cos(q3))
    q2 = phi1 - phi2

    # 3. Forward kinematics to wrist
    q_analytic = np.array([q1, q2, q3, 0, 0, 0])
    T03 = _fk_aloha_3(q_analytic[:3])
    R03 = T03[:3, :3]
    R36 = R03.T @ R06

    # 4. Wrist orientation (q4, q5, q6)
    # R36 = rot_x(q4) @ rot_y(q5) @ rot_x(q6)
    # Decompose R36
    q5 = np.arccos(R36[0, 0])
    if np.abs(np.sin(q5)) < 1e-6:
        q4 = 0
        q6 = np.arctan2(-R36[1, 2], R36[1, 1])
    else:
        q4 = np.arctan2(R36[1, 0], R36[2, 0])
        q6 = np.arctan2(R36[0, 1], -R36[0, 2])

    joints = np.array([q1, q2, q3, q4, q5, q6])

    # Validate with FK
    # Use RPY FK (returns a flat 6D pose vector) for error checking
    fk_pose = calculate_forward_kinematics_rpy(joints)
    err = np.linalg.norm(fk_pose[:3] - p_ee) + np.linalg.norm(fk_pose[3:] - np.array([roll, pitch, yaw]))
    if err > 1e-2:
        # Fallback to numerical
        #print(f"FALLING BACK TO NUMERICAL IK - 2! ERR IN POS: {np.linalg.norm(fk_pose[:3] - p_ee)}, ERR IN ORIENTATION: {np.linalg.norm(fk_pose[3:] - np.array([roll, pitch, yaw]))}, ERR total: {err}")
        _num_solution = _ik_numerical(desired_pose, q, max_iter, tol, alpha)
        return enforce_angle_limits(_num_solution)
    else:
        #print(f"ACCEPTING ANALYTICAL IK SOLUTION. ERR IN POS: {np.linalg.norm(fk_pose[:3] - p_ee)}, ERR IN ORIENTATION: {np.linalg.norm(fk_pose[3:] - np.array([roll, pitch, yaw]))}, ERR total: {err}")
    return enforce_angle_limits(joints.tolist())


def _fk_aloha_3(joints):
    """FK to wrist (frame 3) for ALOHA arm, for analytic IK."""
    j1, j2, j3 = joints
    # Waist
    T1 = np.eye(4)
    T1[:3, :3] = np.array([
        [np.cos(j1), -np.sin(j1), 0],
        [np.sin(j1), np.cos(j1), 0],
        [0, 0, 1]
    ])
    T1[:3, 3] = [0, 0, 0.079]
    # Shoulder
    T2 = np.eye(4)
    T2[:3, :3] = np.array([
        [np.cos(j2), 0, np.sin(j2)],
        [0, 1, 0],
        [-np.sin(j2), 0, np.cos(j2)]
    ])
    T2[:3, 3] = [0, 0, 0.04805]
    # Elbow
    T3 = np.eye(4)
    T3[:3, :3] = np.array([
        [np.cos(j3), 0, np.sin(j3)],
        [0, 1, 0],
        [-np.sin(j3), 0, np.cos(j3)]
    ])
    T3[:3, 3] = [0.05955, 0, 0.3]
    T = T1 @ T2 @ T3
    return T


def _ik_numerical(desired_pose, seed, max_iter, tol, alpha):
    """Simple numerical IK fallback using gradient descent on pose error."""
    q = np.array(seed, dtype=np.float64)
    for i in range(max_iter):
        fk = calculate_forward_kinematics_rpy(q)  # 6D pose
        err_pos = desired_pose[:3] - fk[:3]
        err_rpy = desired_pose[3:] - fk[3:]
        err = np.concatenate([err_pos, err_rpy])
        if np.linalg.norm(err) < tol:
            return q.tolist()
        # Numerical Jacobian
        J = np.zeros((6, 6))
        eps = 1e-6
        for j in range(6):
            dq = np.zeros(6)
            dq[j] = eps
            fk2 = calculate_forward_kinematics_rpy(q + dq)
            derr = np.concatenate([(fk2[:3] - fk[:3]), (fk2[3:] - fk[3:])]) / eps
            J[:, j] = derr
        # Gradient step
        dq = alpha * np.linalg.pinv(J) @ err
        q += dq
    return q.tolist()


if __name__ == "__main__":
    # Example usage

    # Test FK/IK round-trip
    print("\n=== Forward Kinematics Test ===")
    joint_positions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    fk_rpy = calculate_forward_kinematics_rpy(joint_positions)
    print("Input Joint Positions:", joint_positions)
    print("FK End Effector Pose (RPY):", fk_rpy)
    print("FK End Effector Rotation Matrix:\n", convert_rpy_to_rot_matrix(*fk_rpy[3:]))

    print("\n=== Inverse Kinematics Test (Analytic + Numerical Fallback) ===")
    ik_result = calculate_inverse_kinematics(fk_rpy.tolist())
    print("IK Joint Solution:", ik_result)
    fk_from_ik_rpy = calculate_forward_kinematics_rpy(ik_result)
    print("FK from IK Solution (should match target):", fk_from_ik_rpy)
    pos_err = np.linalg.norm(fk_rpy[:3] - fk_from_ik_rpy[:3])
    rpy_err = np.linalg.norm(np.unwrap(fk_rpy[3:]) - np.unwrap(fk_from_ik_rpy[3:]))
    print(f"Position error: {pos_err:.6f} m, RPY error: {rpy_err:.6f} rad")

    # Test with a random pose
    print("\n=== IK Test for Random Pose ===")
    target_pose = [0.3, 0.1, 0.4, 0.0, 1.0, 0.0]
    print("Target Pose:", target_pose)
    ik_result2 = calculate_inverse_kinematics(target_pose)
    print("IK Joint Solution:", ik_result2)
    fk_from_ik2_rpy = calculate_forward_kinematics_rpy(ik_result2)
    print("FK from IK Solution (RPY):", fk_from_ik2_rpy)
    pos_err2 = np.linalg.norm(np.array(target_pose[:3]) - fk_from_ik2_rpy[:3])
    rpy_err2 = np.linalg.norm(np.unwrap(target_pose[3:]) - np.unwrap(fk_from_ik2_rpy[3:]))
    print(f"Position error: {pos_err2:.6f} m, RPY error: {rpy_err2:.6f} rad")

    # Test for a singular configuration (e.g., all joints zero, arm fully extended)
    print("\n=== IK Test for Singular Configuration (All Zeros) ===")
    singular_joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    singular_pose_rpy = calculate_forward_kinematics_rpy(singular_joints)
    print("Singular Joint Positions:", singular_joints)
    print("FK Pose (RPY):", singular_pose_rpy)
    ik_singular = calculate_inverse_kinematics(singular_pose_rpy.tolist())
    print("IK Solution for Singular Pose:", ik_singular)
    fk_from_ik_singular_rpy = calculate_forward_kinematics_rpy(ik_singular)
    print("FK from IK Solution (RPY):", fk_from_ik_singular_rpy)
    pos_err_sing = np.linalg.norm(singular_pose_rpy[:3] - fk_from_ik_singular_rpy[:3])
    rpy_err_sing = np.linalg.norm(np.unwrap(singular_pose_rpy[3:]) - np.unwrap(fk_from_ik_singular_rpy[3:]))
    print(f"Position error: {pos_err_sing:.6f} m, RPY error: {rpy_err_sing:.6f} rad")

    # Test: Same pose, different seeds
    print("\n=== IK Test: Same Pose, Different Seeds ===")
    target_pose = [0.3, 0.1, 0.4, 0.0, 1.0, 0.0]
    seed1 = [0.0, -0.96, 1.16, 0.0, -0.3, 0.0]
    seed2 = [np.pi/2, -1.0, 1.0, 0.0, 0.0, 0.0]
    print("Target Pose:", target_pose)
    print("Seed 1:", seed1)
    ik1 = calculate_inverse_kinematics(target_pose, seed=seed1)
    print("IK Solution (Seed 1):", ik1)
    fk1_rpy = calculate_forward_kinematics_rpy(ik1)
    print("FK from IK1 (RPY):", fk1_rpy)
    print("Seed 2:", seed2)
    ik2 = calculate_inverse_kinematics(target_pose, seed=seed2)
    print("IK Solution (Seed 2):", ik2)
    fk2_rpy = calculate_forward_kinematics_rpy(ik2)
    print("FK from IK2 (RPY):", fk2_rpy)
    print("Difference between IK1 and IK2:", np.array(ik1) - np.array(ik2))
    print("Position error IK1:", np.linalg.norm(np.array(target_pose[:3]) - fk1_rpy[:3]))
    print("Position error IK2:", np.linalg.norm(np.array(target_pose[:3]) - fk2_rpy[:3]))