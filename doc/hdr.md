## 目标
在这个章节，我们将学会：
- 学习如何从曝光序列生成和显示HDR图像
- 使用曝光融合合并曝光序列。

## 理论
高动态范围成像（HDRI或HDR）是一种用于成像和摄影的技术，其目的是再现比标准数字成像或摄影技术更大的动态光度范围。虽然人眼可以适应各种光线条件，但大多数成像设备每通道使用8位，因此我们只能使用256级。当我们拍摄真实世界的场景时，明亮的区域可能曝光过度，而黑暗的区域可能曝光不足，所以我们不能用一次曝光来捕捉所有的细节。HDR图像处理每个通道使用8位以上的图像（通常为32位浮点值），允许更宽的动态范围。

获取HDR图像有多种方法，但最常见的方法是使用不同曝光值拍摄的场景照片。要组合这些曝光，了解相机的响应函数是很有用的，并且有算法可以估计它。合并HDR图像后，必须将其转换回8位才能在常规显示器上查看。这个过程叫做色调映射。当场景或相机的对象在镜头之间移动时，会产生额外的复杂性，因为具有不同曝光的图像应该被注册和对齐。

在本教程中，我们展示了从曝光序列生成和显示HDR图像的两种算法（Debevec、Robertson），并演示了一种称为曝光融合（Mertens）的替代方法，该方法生成低动态范围图像且不需要曝光时间数据。此外，我们还估计了摄像机响应函数（CRF），这对于许多计算机视觉算法都具有重要价值。HDR流水线的每一步都可以使用不同的算法和参数来实现，所以请看一下参考手册，了解它们的全部内容。

## 曝光顺序HDR
在本教程中，我们将看到以下场景，其中我们有4个曝光图像，曝光时间为：15秒、2.5秒、1/4秒和1/30秒。（你可以从[维基百科](https://en.wikipedia.org/wiki/High-dynamic-range_imaging)下载图片）
![img](./images/hdr/exposures.jpg)

### 1. 将曝光图像加载到列表中
第一个阶段只是将所有图像加载到一个列表中。另外，我们还需要常规HDR算法的曝光时间。注意数据类型，因为图像应该是1通道或3通道8位(np.uint8)曝光时间必须float32 单位秒。

```
import cv2 as cv
import numpy as np
# Loading exposure images into a list
img_fn = ["img0.jpg", "img1.jpg", "img2.jpg", "img3.jpg"]
img_list = [cv.imread(fn) for fn in img_fn]
exposure_times = np.array([15.0, 2.5, 0.25, 0.0333], dtype=np.float32)
```

### 2. 将曝光合并到HDR图像
在这个阶段，我们将曝光序列合并成一个HDR图像，显示了OpenCV中的两种可能性。第一种方法是Debevec，第二种是Robertson。注意，HDR图像的类型是float32，而不是uint8，因为它包含所有曝光图像的完整动态范围。
```
# Merge exposures to HDR image
merge_debevec = cv.createMergeDebevec()
hdr_debevec = merge_debevec.process(img_list, times=exposure_times.copy())
merge_robertson = cv.createMergeRobertson()
hdr_robertson = merge_robertson.process(img_list, times=exposure_times.copy())
```

### 3. 高清图像(Tonemap HDR image)
我们将32位浮点HDR数据映射到范围[0..1]。实际上，在某些情况下，值可以大于1或小于0，因此请注意，我们稍后将不得不剪裁数据以避免溢出。

```
# Tonemap HDR image
tonemap1 = cv.createTonemap(gamma=2.2)
res_debevec = tonemap1.process(hdr_debevec.copy())
```

### 4. 使用Mertens融合的合并暴露
在这里，我们展示了另一种算法来合并曝光图像，其中我们不需要曝光时间。我们也不需要使用任何色调映射算法，因为Mertens算法已经给出了[0..1]范围内的结果。
```
# Exposure fusion using Mertens
merge_mertens = cv.createMergeMertens()
res_mertens = merge_mertens.process(img_list)
```

### 5. 转换为8位并保存
为了保存或显示结果，我们需要将数据转换为[0..255]范围内的8位整数。
```
# Convert datatype to 8-bit and save
res_debevec_8bit = np.clip(res_debevec*255, 0, 255).astype('uint8')
res_robertson_8bit = np.clip(res_robertson*255, 0, 255).astype('uint8')
res_mertens_8bit = np.clip(res_mertens*255, 0, 255).astype('uint8')
cv.imwrite("ldr_debevec.jpg", res_debevec_8bit)
cv.imwrite("ldr_robertson.jpg", res_robertson_8bit)
cv.imwrite("fusion_mertens.jpg", res_mertens_8bit)
```

## 结果
您可以看到不同的结果，但请考虑每个算法都有额外的参数，您应该适合这些参数以获得所需的结果。最佳实践是尝试不同的方法，看看哪种方法对您的场景最有效。

## 摄像机响应估值函数
相机响应函数（CRF）给出了场景辐射与测量强度值之间的关系。CRF-if在一些计算机视觉算法中具有重要意义，包括HDR算法。在这里我们估计相机的反向响应函数，并将其用于HDR合并。
```
# Estimate camera response function (CRF)
cal_debevec = cv.createCalibrateDebevec()
crf_debevec = cal_debevec.process(img_list, times=exposure_times)
hdr_debevec = merge_debevec.process(img_list, times=exposure_times.copy(), response=crf_debevec.copy())
cal_robertson = cv.createCalibrateRobertson()
crf_robertson = cal_robertson.process(img_list, times=exposure_times)
hdr_robertson = merge_robertson.process(img_list, times=exposure_times.copy(), response=crf_robertson.copy())
```

相机响应函数由每个颜色通道的256长度向量表示。对于这个序列，我们得到了以下估计：

