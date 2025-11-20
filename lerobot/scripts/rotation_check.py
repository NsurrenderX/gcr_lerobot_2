from lerobot.common.datasets.rotation_convert import (
    matrix_to_quaternion,
    quaternion_to_matrix,
    matrix_to_euler,
    ortho6d_to_matrix,
    matrix_to_ortho6d,
)

from scipy.spatial.transform import Rotation as R

import numpy as np
import torch

if __name__ == "__main__":
    example_matrix = np.array([
        [0.97026573, -0.13453148,  0.20116464],
        [-0.19265776, -0.93243551,  0.30566249],
        [0.14645183, -0.33532977, -0.93064667]
    ])
    
    example_quat = R.from_matrix(example_matrix).as_quat()
    print("Example quat: ", example_quat)