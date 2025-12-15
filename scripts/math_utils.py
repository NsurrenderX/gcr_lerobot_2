import numpy as np


def _calculate_fk_position_and_rotation(joint_positions):
    """Return the end-effector position and rotation matrix."""

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

def calculate_forward_kinematics(joint_positions):
    """Return [x, y, z, roll, pitch, yaw] for the ALOHA VX300S arm."""

    position, rotation_matrix = _calculate_fk_position_and_rotation(joint_positions)
    rpy = convert_rot_matrix_to_rpy(rotation_matrix)
    return np.array([position[0], position[1], position[2], rpy[0], rpy[1], rpy[2]], dtype=np.float64)


def calculate_forward_kinematics_pose(joint_positions):
    """Return the raw end-effector position and rotation matrix."""

    return _calculate_fk_position_and_rotation(joint_positions)


def calculate_forward_kinematics_rpy(joint_positions):
    """Backwards compatible alias for calculate_forward_kinematics."""

    return calculate_forward_kinematics(joint_positions)


def calculate_forward_kinematics_quaternion(joint_positions):
    """Forward kinematics outputting position plus quaternion orientation."""

    pose_rpy = calculate_forward_kinematics(joint_positions)
    quat = convert_rpy_to_quaternion(pose_rpy[3], pose_rpy[4], pose_rpy[5])
    return np.concatenate([pose_rpy[:3], quat])

def convert_rot_matrix_to_quaternion(rotation_matrix):
    assert rotation_matrix.shape == (3, 3), "Input must be a 3x3 rotation matrix"
    m = rotation_matrix
    trace = np.trace(m)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (m[2, 1] - m[1, 2]) * s
        qy = (m[0, 2] - m[2, 0]) * s
        qz = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s

    quat = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm == 0:
        return quat
    return quat / norm

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


def convert_rpy_to_quaternion(roll, pitch, yaw):
    """Convert roll, pitch, yaw (Euler angles) to quaternion [x, y, z, w]."""
    half_roll = roll * 0.5
    half_pitch = pitch * 0.5
    half_yaw = yaw * 0.5

    cr = np.cos(half_roll)
    sr = np.sin(half_roll)
    cp = np.cos(half_pitch)
    sp = np.sin(half_pitch)
    cy = np.cos(half_yaw)
    sy = np.sin(half_yaw)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    quat = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm == 0:
        return quat
    return quat / norm


def convert_quaternion_to_rpy(qx, qy, qz, qw):
    """Convert quaternion [x, y, z, w] to roll, pitch, yaw."""
    quat = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm == 0:
        raise ValueError("Quaternion has zero magnitude; cannot convert to Euler angles.")
    qx, qy, qz, qw = quat / norm

    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (qw * qy - qz * qx)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def calculate_inverse_kinematics_raw(
    desired_pose,
    seed=None,
    max_iter=100,
    tol=1e-4,
    alpha=0.2,
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
        return _ik_numerical(desired_pose, q, max_iter, tol, alpha)
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
    fk_pose = calculate_forward_kinematics(joints)
    err = np.linalg.norm(fk_pose[:3] - p_ee) + np.linalg.norm(fk_pose[3:] - np.array([roll, pitch, yaw]))
    if err > 1e-2:
        # Fallback to numerical
        return _ik_numerical(desired_pose, q, max_iter, tol, alpha)
    return joints.tolist()

def calculate_inverse_kinematics(
    desired_pose,
    seed=None,
    max_iter=100,
    tol=1e-4,
    alpha=0.2,
):
    """
    Wrapper for inverse kinematics that ensures joint limits are respected.
    Joint limits for ALOHA VX300S 6-DoF arm:
    """
    raw_solution = calculate_inverse_kinematics_raw(
        desired_pose,
        seed=seed,
        max_iter=max_iter,
        tol=tol,
        alpha=alpha,
    )

    '''
    'waist' limits: [-3.141582727432251, 3.141582727432251]
    'shoulder' limits: [-1.8500490188598633, 1.2566370964050293]
    'elbow' limits: [-1.7627825736999512, 1.6057028770446777]
    'forearm_roll' limits: [-3.141582727432251, 3.141582727432251]
    'wrist_angle' limits: [-1.8675023317337036, 2.2340214252471924]
    'wrist_rotate' limits: [-3.141582727432251, 3.141582727432251]
    '''
    # Joint limits
    joint_limits = [
        (-3.141582727432251, 3.141582727432251),  # waist
        (-1.8500490188598633, 1.2566370964050293),  # shoulder
        (-1.7627825736999512, 1.6057028770446777),  # elbow
        (-3.141582727432251, 3.141582727432251),  # forearm_roll
        (-1.8675023317337036, 2.2340214252471924),  # wrist_angle
        (-3.141582727432251, 3.141582727432251),  # wrist_rotate
    ]

    adjusted_solution = []
    two_pi = 2 * np.pi
    for joint, (lower, upper) in zip(raw_solution, joint_limits):
        if lower <= joint <= upper:
            adjusted_solution.append(joint)
            continue

        candidate = joint
        if joint < lower:
            while True:
                candidate += two_pi
                if lower <= candidate <= upper:
                    break
                if candidate > upper:
                    candidate = joint
                    break
        else:
            while True:
                candidate -= two_pi
                if lower <= candidate <= upper:
                    break
                if candidate < lower:
                    candidate = joint
                    break

        adjusted_solution.append(candidate)

    return adjusted_solution


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


def _ik_numerical(desired_pose, seed, max_iter, tol, alpha, damping=1e-3):
    q = np.array(seed, dtype=np.float64)
    target = np.asarray(desired_pose, dtype=np.float64)
    for _ in range(max_iter):
        fk = calculate_forward_kinematics(q)
        err = target - fk
        if np.linalg.norm(err) < tol:
            return q.tolist()

        J = np.zeros((6, 6))
        eps = 1e-6
        for j in range(6):
            dq = np.zeros(6)
            dq[j] = eps
            J[:, j] = (calculate_forward_kinematics(q + dq) - fk) / eps

        # damped least squares step
        JTJ = J.T @ J + damping * np.eye(6)
        dq = alpha * np.linalg.solve(JTJ, J.T @ err)
        q += dq
    return q.tolist()





if __name__ == "__main__":
    # Lightweight smoke tests that double as usage examples.

    # Test FK/IK round-trip
    print("\n=== Forward Kinematics Test ===")
    joint_positions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    fk_result = calculate_forward_kinematics(joint_positions)
    assert fk_result.shape == (6,), "FK should return a 6D pose"
    fk_quat = calculate_forward_kinematics_quaternion(joint_positions)
    assert fk_quat.shape == (7,), "Quaternion FK must be xyz+quat"
    assert np.isclose(np.linalg.norm(fk_quat[3:]), 1.0, atol=1e-6), "Quaternion must be unit length"
    pose_pos, pose_rot = calculate_forward_kinematics_pose(joint_positions)
    assert pose_pos.shape == (3,) and pose_rot.shape == (3, 3), "Pose helper must return position and rotation"
    quat_from_rot = convert_rot_matrix_to_quaternion(pose_rot)
    assert np.allclose(quat_from_rot, fk_quat[3:], atol=1e-6), "Matrix/quat conversions should match FK quaternion"
    rpy_roundtrip = convert_quaternion_to_rpy(*quat_from_rot)
    assert np.allclose(rpy_roundtrip, fk_result[3:], atol=1e-6), "Quaternion to RPY should invert FK orientation"
    print("Input Joint Positions:", joint_positions)
    print("FK End Effector Position:", fk_result[:3])
    print("FK End Effector Rotation (RPY):", fk_result[3:])

    print("\n=== Inverse Kinematics Test (Analytic + Numerical Fallback) ===")
    ik_result = calculate_inverse_kinematics(fk_result)
    fk_from_ik = calculate_forward_kinematics(ik_result)
    pos_err = np.linalg.norm(fk_result[:3] - fk_from_ik[:3])
    rpy_err = np.linalg.norm(np.unwrap(fk_result[3:]) - np.unwrap(fk_from_ik[3:]))
    assert pos_err < 1e-4 and rpy_err < 1e-3, "FK/IK round-trip should be tight"
    print(f"Position error: {pos_err:.6f} m, RPY error: {rpy_err:.6f} rad")

    # Test with a random pose
    print("\n=== IK Test for Random Pose ===")
    target_pose = [0.3, 0.1, 0.4, 0.0, 1.0, 0.0]
    ik_result2 = calculate_inverse_kinematics(target_pose)
    fk_from_ik2 = calculate_forward_kinematics(ik_result2)
    pos_err2 = np.linalg.norm(np.array(target_pose[:3]) - fk_from_ik2[:3])
    rpy_err2 = np.linalg.norm(np.unwrap(target_pose[3:]) - np.unwrap(fk_from_ik2[3:]))
    assert pos_err2 < 5e-3 and rpy_err2 < 5e-3, "Random pose IK should converge"
    print(f"Position error: {pos_err2:.6f} m, RPY error: {rpy_err2:.6f} rad")

    # Test for a singular configuration (e.g., all joints zero, arm fully extended)
    print("\n=== IK Test for Singular Configuration (All Zeros) ===")
    singular_joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    singular_pose = calculate_forward_kinematics(singular_joints)
    ik_singular = calculate_inverse_kinematics(singular_pose)
    fk_from_ik_singular = calculate_forward_kinematics(ik_singular)
    pos_err_sing = np.linalg.norm(singular_pose[:3] - fk_from_ik_singular[:3])
    rpy_err_sing = np.linalg.norm(np.unwrap(singular_pose[3:]) - np.unwrap(fk_from_ik_singular[3:]))
    assert pos_err_sing < 5e-3 and rpy_err_sing < 5e-3, "Singular pose IK should stay stable"
    print(f"Position error: {pos_err_sing:.6f} m, RPY error: {rpy_err_sing:.6f} rad")

    # Test: Same pose, different seeds
    print("\n=== IK Test: Same Pose, Different Seeds ===")
    seed1 = [0.0, -0.96, 1.16, 0.0, -0.3, 0.0]
    seed2 = [np.pi / 2, -1.0, 1.0, 0.0, 0.0, 0.0]
    ik1 = calculate_inverse_kinematics(target_pose, seed=seed1)
    ik2 = calculate_inverse_kinematics(target_pose, seed=seed2)
    fk1 = calculate_forward_kinematics(ik1)
    fk2 = calculate_forward_kinematics(ik2)
    err1 = np.linalg.norm(np.array(target_pose[:3]) - fk1[:3])
    err2 = np.linalg.norm(np.array(target_pose[:3]) - fk2[:3])
    assert err1 < 5e-3 and err2 < 5e-3, "IK should converge regardless of reasonable seed"
    print("Seed 1/2 position errors:", err1, err2)

    print("\nAll math_utils smoke tests passed.")
