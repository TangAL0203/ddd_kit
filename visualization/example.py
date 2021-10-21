from .load import load_pointcloud
from .show_result import show_result


def show(self, results, out_dir, index):
    data_info = self.data_infos[index]
    print('data_info sample id ', data_info['image']['image_idx'])
    det_results = results['det_results']

    batch = len(det_results)
    for i in range(batch):
        # image
        imgs = self.load_image(data_info['image']['image_path'])['imgs']
        # lidar
        pts_filename = data_info['point_cloud']['path']
        points = load_pointcloud(pts_filename)[:, :3]
        # pred
        pred_bboxes_3d, scores, labels = list(det_results[i].values())
        # label
        label = self.load_annotations(data_info)

        projections = label['lidar2img']
        intrinsics = label['intrinsics'][0]
        extrinsics = label['extrinsics'][0]

        gt_bboxes_3d = label['gt_bboxes_3d']
        # visualize
        show_result(imgs, points, gt_bboxes_3d, pred_bboxes_3d.to('cpu'),
                    projections, out_dir=out_dir,
                    filename=osp.split(pts_filename)[-1].split('.')[0],
                    intrinsics=intrinsics, extrinsics=extrinsics,
                    snapshot=True, show_open3d=True)

