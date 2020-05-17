# key-frames
 Key frame extraction of video by OpticalFlow method and KMeans

# DataSet

The dataset we used was published by USTC, a Sign Language Dataset.

[link](http://home.ustc.edu.cn/~pjh/openresources/slr/index.html)

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

## OpticalFlow:

![](https://github.com/larry-11/key-frames/blob/master/img/0.avi/OpticalFlow.png)

## KeyFrame:

![](https://github.com/larry-11/key-frames/blob/master/img/0.avi/26.jpg)

![](https://github.com/larry-11/key-frames/blob/master/img/0.avi/50.jpg)

![](https://github.com/larry-11/key-frames/blob/master/img/0.avi/67.jpg)

Shown in img/0.avi