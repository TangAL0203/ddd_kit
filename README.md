### collect common 3d code
This repo is a code collector, including some code commonly used for cver.<br />


#### video

1. video generation<br />
```python
from ddd import video
video.frames2video(frame_dir='path2img', 'temp.avi', fps=15)
```


#### img
1. io<br />
```python
from ddd import image
img = image.io.imread(img_path, flag='color', channel_order='bgr')
image.io.imwrite(img, file_path='xxx/yyy.jpg')
```
2. colorspace convert<br />
```python
from ddd.image import colorspace
# then get func you need to do the conversion, e.g.
bgr_img = colorspace.gray2bgr(img)
```
