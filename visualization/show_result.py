from os import path as osp

import cv2
import mmcv
import numpy as np
import trimesh

from .image_vis import draw_corners


def _write_ply(points, out_filename):
    """Write points into ``ply`` format for meshlab visualization.

    Args:
        points (np.ndarray): Points in shape (N, dim).
        out_filename (str): Filename to be saved.
    """
    N = points.shape[0]
    fout = open(out_filename, 'w')
    for i in range(N):
        if points.shape[1] == 6:
            c = points[i, 3:].astype(int)
            fout.write(
                'v %f %f %f %d %d %d\n' %
                (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))

        else:
            fout.write('v %f %f %f\n' %
                       (points[i, 0], points[i, 1], points[i, 2]))
    fout.close()


def _write_oriented_bbox(scene_bbox, out_filename):
    """Export oriented (around Z axis) scene bbox to meshes.

    Args:
        scene_bbox(list[ndarray] or ndarray): xyz pos of center and
            3 lengths (dx,dy,dz) and heading angle around Z axis.
            Y forward, X right, Z upward. heading angle of positive X is 0,
            heading angle of positive Y is 90 degrees.
        out_filename(str): Filename.
    """

    def heading2rotmat(heading_angle):
        rotmat = np.zeros((3, 3))
        rotmat[2, 2] = 1
        cosval = np.cos(heading_angle)
        sinval = np.sin(heading_angle)
        rotmat[0:2, 0:2] = np.array([[cosval, -sinval], [sinval, cosval]])
        return rotmat

    def convert_oriented_box_to_trimesh_fmt(box):
        ctr = box[:3]
        lengths = box[3:6]
        trns = np.eye(4)
        trns[0:3, 3] = ctr
        trns[3, 3] = 1.0
        trns[0:3, 0:3] = heading2rotmat(box[6])
        box_trimesh_fmt = trimesh.creation.box(lengths, trns)
        return box_trimesh_fmt

    if len(scene_bbox) == 0:
        scene_bbox = np.zeros((1, 7))
    scene = trimesh.scene.Scene()
    for box in scene_bbox:
        scene.add_geometry(convert_oriented_box_to_trimesh_fmt(box))

    mesh_list = trimesh.util.concatenate(scene.dump())
    # save to ply file
    trimesh.io.export.export_mesh(mesh_list, out_filename, file_type='ply')

    return


def show_result(imgs, points, gt_bboxes, pred_bboxes, projections, out_dir,
                filename, intrinsics=None, extrinsics=None, snapshot=False,
                show_open3d=False):
    """Convert results into format that is directly readable for meshlab.

    Args:
        imgs (list): list of multi-view images
        points (np.ndarray): Points.
        gt_bboxes (:obj:`LiDARInstance3DBoxes` cpu): Ground truth boxes.
        pred_bboxes (:obj:`LiDARInstance3DBoxes` cpu): Predicted boxes.
        projections (list): list of multi-vew projection matrix
        out_dir (str): Path of output directory
        filename (str): Filename of the current frame.
        intrinsics (np.array): intrinsics.
        extrinsics (np.array): extrinsics.
        snapshot (bool): whether save snapshot of visualization using open3d.
        show_open3d (bool): whether show visualization using open3d.
    """
    # print ('imgs ', len(imgs), imgs[0].shape)
    # print ('projections ', projections.shape)
    result_path = osp.join(out_dir, filename)
    mmcv.mkdir_or_exist(result_path)

    # multiview visualize
    for i, img in enumerate(imgs):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if gt_bboxes is not None:
            gt_corners = gt_bboxes.corners.numpy()
            for gt_corner in gt_corners:
                draw_corners(img, gt_corner, (0, 255, 0), projections[i])
        if pred_bboxes is not None:
            pred_corners = pred_bboxes.corners.numpy()
            for pred_corner in pred_corners:
                draw_corners(img, pred_corner, (255, 0, 0), projections[i])

        cv2.imshow('img ', img)
        cv2.waitKey(0)

    if show_open3d:
        from .open3d_vis import Visualizer

        vis = Visualizer(points, intrinsics, extrinsics)
        if pred_bboxes is not None:
            vis.add_bboxes(bbox3d=pred_bboxes.tensor.numpy())  # green
        if gt_bboxes is not None:
            vis.add_bboxes(bbox3d=gt_bboxes.tensor.numpy(),
                           bbox_color=(0, 0, 1))  # blue
        show_path = osp.join(result_path,
                             f'{filename}_online.png') if snapshot else None
        vis.show(show_path)

    return

