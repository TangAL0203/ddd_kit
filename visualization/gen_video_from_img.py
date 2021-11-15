import os
import os.path as osp
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm


root_dir = 'path_to_img_dir'

framesize = (5760, 2160)  # WH
fps = 3
time = 15  # seconds
fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
save = 'fps{}_{}s_3d.avi'.format(fps, time)

videoWriter = cv2.VideoWriter(save, fourcc, fps, framesize)


all_imgs = []
imgs = os.listdir(osp.join(root_dir))
sorted_imgs = sorted([os.path.join(osp.join(root_dir), img) for img in imgs])


num_img = time * fps
assert num_img < len(sorted_imgs)

# layout
# img4, img0, img2
# img5, img3, img1

for ii in tqdm(range(num_img)):
    imgs = []

    for file in sorted(os.listdir(sorted_imgs[ii])):
        if file.endswith('jpg'):
            imgs.append(cv2.imread(osp.join(sorted_imgs[ii], file)))

    front = [imgs[4], imgs[0], imgs[2]]
    front = [np.concatenate(front, 1).astype(np.uint8)]
    back = [imgs[5], imgs[3], imgs[1]]
    back = [np.concatenate(back, 1).astype(np.uint8)]
    frame = np.concatenate(front+back, 0).astype(np.uint8)

    videoWriter.write(frame)

videoWriter.release()
cv2.destroyAllWindows()
