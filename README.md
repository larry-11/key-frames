# key-frames
 Key frame extraction of video by OpticalFlow method and KMeans

# DataSet

The dataset we used was published by USTC, a Sign Language Dataset.

link:[](http://home.ustc.edu.cn/~pjh/openresources/slr/index.html)

# Requirement

- opencv-python
- sklearn

# Usage

```
python test.py   
  -h, --help      show this help message and exit
  -k K            the number of k-means
  -thresh THRESH  the threshold of OpticalFlow value
  -src SRC        the folder of input videos
  -des DES        the folder of output frames
```

# Result

shown in img/0.avi