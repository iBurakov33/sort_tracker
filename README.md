# tensorrt_yolov5_tracker

This is a project to deploy object tracking algorithm with yolov5 and TensorRT. Sort and Deep-sort algorithm are used to track the objects.
Thanks for the contribution of [tensorrtx](https://github.com/wang-xinyu/tensorrtx) and [sort-cpp](https://github.com/yasenh/sort-cpp)!
Now this project implements the tracki algorithm with yolov5 and sort(c++ version). **Deep-sort, multi-thread acceleration and python implemention** will be updated soon. Please give a star if this project helps ^_^

![example result of object tracking](https://github.com/AsakusaRinne/tensorrt_yolov5_tracker/blob/main/example.gif)
## Table of Contents

- [ToDo](#ToDo)
- [Install](#install)
- [Usage](#usage)
- [Reference](#Reference)
- [License](#license)

## ToDo
***1.Deep-sort tracker.***

***2.Multi-thread acceleration for c++.***

***3.Python implemention.***


## Install

```
$ git https://github.com/AsakusaRinne/tensorrt_yolov5_tracker.git
$ cd tensorrt_yolov5_tracker
$ mkdir build
$ cd build
$ cmake ..
$ make
```

## Usage
You could change the batch size in ``yolov5/yolov5.h``
You could change the input image size in ``yolov5/yololayer.h``

### SORT tuning for very low FPS (1-2)

At 1-2 FPS, object movement between consecutive frames can be large, so pure IoU matching in SORT becomes less stable. In this repository, the default tracker parameters are optimized for denser frame streams, so it is recommended to tune them for sparse video:

1. **Increase track lifetime without detections**  
   In `sort/include/utils.h`, increase `kMaxCoastCycles` from `1` to `3-8` so tracks are not dropped immediately when one frame is missed.

2. **Lower the minimum detector confidence**  
   In `sort/include/utils.h`, reduce `kMinConfidence` (for example, from `0.6` to `0.4-0.5`) to keep more candidate detections for association.

3. **Use a lower IoU association threshold**  
   In `sort/src/tracker.cpp`, `AssociateDetectionsToTrackers(...)` is called with default `iou_threshold = 0.1`. For low FPS, decreasing to `0.01-0.05` may help preserve identities when displacement is large.

4. **Relax Kalman process noise for velocity terms**  
   In `sort/src/track.cpp`, matrix `Q_` controls model uncertainty. Increasing velocity-related noise (last 4 diagonal values) helps the filter adapt to larger jumps between frames.

5. **Improve detector-side robustness first**  
   With 1-2 FPS, tracking quality depends heavily on detection quality. Use larger input resolution, stronger model, and class-specific confidence tuning to minimize missed detections.

6. **If possible, fuse appearance features (DeepSORT/BoT-SORT)**  
   This codebase currently uses geometric SORT matching. At very low FPS, appearance-based re-identification usually gives a significant gain in ID stability.

Quick practical starting point for low FPS:

```cpp
// sort/include/utils.h
constexpr int kMaxCoastCycles = 5;
constexpr int kMinHits = 1;
constexpr float kMinConfidence = 0.45;
```

Then reduce IoU threshold in association to around `0.03`, evaluate ID switches, and iterate.


#### Exact `sort/src/track.cpp` replacement for Kalman `Q_` (recommended start)

Replace this block in `Track::Track()`:

```cpp
kf_.Q_ <<
       1, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0.01, 0, 0, 0,
        0, 0, 0, 0, 0, 0.01, 0, 0,
        0, 0, 0, 0, 0, 0, 0.0001, 0,
        0, 0, 0, 0, 0, 0, 0, 0.0001;
```

with:

```cpp
kf_.Q_ <<
       1, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 1.0, 0, 0, 0,
        0, 0, 0, 0, 0, 1.0, 0, 0,
        0, 0, 0, 0, 0, 0, 0.05, 0,
        0, 0, 0, 0, 0, 0, 0, 0.05;
```

If IDs still switch often, increase the last 4 diagonal terms by ~1.5-2x; if boxes jitter too much, reduce them by ~30-50%.

We provide the support for building models of yolov5-3.1 and yolov5-4.0 in our project. The default branch is corresponding to yolov5-4.0. Please change to ``c269f65`` if you want to use yolov5-3.1 models to build engines.

If other versions are wanted, please kindly refer to [tensorrtx](https://github.com/wang-xinyu/tensorrtx) to build the engine and then copy to the folder of this project.


### Build engine
At first, please put the ``yolov5/gen_wts.py`` in the folder of corresponding version of [**Yolov5**](https://github.com/ultralytics/yolov5). For example, if you use the model of yolov5-4.0, please download the release of yolov5-4.0 and put the ``gen_wts.py`` in its folder. Then run the code to convert model from ``.pt`` to ``.wts``. Then please use the following instructions to build your engines.
```
$ sudo ./Tracker -s [.wts filename] [.engine filename] [s, m, l, or x] # build yolov5 model of [s, m, l, or x]

$ sudo ./Tracker -s  [.wts filename] [.engine filename] [depth] [width] # build yolov5 model of scales which are not s, m, l, or x
**for example**
$ sudo ./Tracker -s  ../yolov5/weights/yolov5.wts ../engines/yolov5.engine 0.17 0.25
```
### Process video
```
$ sudo ./Tracker -v [video filename] [engine filename]
```
The output video will be in the folder ``output``
Note that the track time includes the time of drawing boxes on images and saving them to the disk, which makes it a little long. You can delete the codes for drawing and saving if they are not needed to accelerate the process.

## Reference
[**tensorrtx**](https://github.com/wang-xinyu/tensorrtx)

[**Yolov5**](https://github.com/ultralytics/yolov5)

[**sort-cpp**](https://github.com/yasenh/sort-cpp)

## License
[MIT license](https://mit-license.org/)
