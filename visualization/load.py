import numpy as np
from pyntcloud import PyntCloud

__all__ = ['load_pointcloud']


def load_pointcloud(path):
    if path.endswith('bin'):
        lidar_points = np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)
    elif path.endswith('pcd'):
        scan = PyntCloud.from_file(str(path))
        lidar_points = np.concatenate([scan.xyz, scan.points.intensity.values.reshape(-1, 1)], axis=1)
    else:
        raise NotImplementedError()
    return lidar_points.astype(np.float32)
