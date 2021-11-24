import cv2
import numpy as np
import pytest
from bbox_util import get_corners


def test_get_corners():
    img = np.ones((1000, 1000, 3))
    img.fill(255)
    center = np.array([
        [250, 250],
        [750, 250],
        [750, 750],
        [250, 750],
    ])
    size = np.array([
        [50, 100],
        [40, 80],
        [50, 100],
        [40, 80],
    ])
    yaw = np.array([-np.pi/4, -3*np.pi/4, 3*np.pi/4, np.pi/4])  # same format as kitti
    corners = get_corners(center, size, yaw)
    # draw origin point
    cv2.circle(img, center=(499, 499), radius=3, color=(0, 255, 0), thickness=2)
    for idx, corner in enumerate(corners):
        # drow center point
        cv2.circle(img, center=(int(center[idx][0]), int(center[idx][1])), radius=3,
                   color=(0, 0, 255), thickness=2)
        # drow rect
        rect = cv2.minAreaRect(corner)
        box = np.int0(cv2.boxPoints(rect))
        cv2.drawContours(img, [box], 0, (0, 0, 255), 3)
        # cv2.polylines(img, [box], True, (0, 0, 255), 3)

    cv2.imshow('img', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    test_get_corners()
