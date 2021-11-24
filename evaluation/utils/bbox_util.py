import numpy as np

__all__ = ['get_corners']


def get_corners(center, size, yaw):
    """Get the coordinates of the 4 points of the rotation box.

    Args:
        center (np.ndarray): Bbox centers with shape (n, 2).
        size (np.ndarray): Bbox shape with shape (n, 2), represents the W and L in lidar coordinate systems.
        yaw (np.ndarray): Bbox yaw with shape (n,), same as rotation_y in kitti-format, ranges in
            [-pi, pi].

    Returns:
        all_corners (np.ndarray): Bbox corners matrix, with shape (n, 4, 2).

    """
    yaw_cos = np.cos(yaw)
    yaw_sin = np.sin(yaw)

    all_corners = []
    length = center.shape[0]
    for idx in range(length):
        rot = np.asmatrix([[yaw_cos[idx], -yaw_sin[idx]], [yaw_sin[idx], yaw_cos[idx]]])
        plain_pts = np.asmatrix([
            [0.5 * size[idx][0], 0.5 * size[idx][1]], [0.5 * size[idx][0], -0.5 * size[idx][1]],
            [-0.5 * size[idx][0], -0.5 * size[idx][1]], [-0.5 * size[idx][0], 0.5 * size[idx][1]]
        ])
        tran_pts = np.asarray(rot * plain_pts.transpose())  # 2x2 @ 2x4 -> 2x4
        tran_pts = tran_pts.transpose()  # 2x4 -> 4x2

        tran_pts = tran_pts[[3, 2, 1, 0], :]

        corners = np.arange(8).astype(np.float32).reshape(4, 2)
        for i in range(4):
            corners[i][0] = center[idx][0] + tran_pts[i % 4][0]
            corners[i][1] = center[idx][1] + tran_pts[i % 4][1]
        all_corners.append(corners)

    if len(all_corners) != 0:
        all_corners = np.stack(all_corners, axis = 0)
    else:
        all_corners = np.array(all_corners).reshape(-1, 4, 2)

    return all_corners

