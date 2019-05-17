# img dtype survey

## knowledges

- [图像数据类型及颜色空间转换](https://www.cnblogs.com/denny402/p/5122328.html)


## questions

- what is the default dtype of img readed by opencv?
- Can it be changed?
- what will happen if we read a image in `grayscale` flag?
- what will happen if we convert a rgb to gray?
- what is the behaviour of `plt.imshow`?
- what is the behaviour of `tf.summary.image`

## answers

On the one hand, the img read by opencv always have `uint8` dtype whether the `grayscale` flag is specified or not. I *don't know* what the behaviour is if you specify other flags, currently I am only testing on *no flags* or *cv2.IMREAD_GRAYSCALE*.

On the other hand, gray img converted by opencv `cv2.cv2Color` still yield `uint8` datatype.

Besides, I test different cases for `plt` package, and it shows some interesting results:

Before we start, we shall confirm that a gray img *does not* contains the 3rd channel no matter it is directly read or converted by opencv. So one obvious and significant difference betwwen rgb and gray is that the contains different number of channels.

The function `plt.imshow()` would either accept a rgb or gray. When it accepts an rgb img as param, it will check the dtype of the img. If the img is of `float32`, then it assumes all entries in the numpy matrix shall fall in \[0, 1\] and it will clip any values out of range. If the img is of `uint8`, then it assumes all entries in the numpy matrix shall fall in \[0, 255\] and it will clip any values out of range.

When it comes to the gray img, things become interesting. It turns out that no matter how you change the dtype or scale the numpy matrix, the `plt.imshow` will kindly accepts and shows the same output. By the way, don't forget to set `cmap` as `gray` or `rainbow` , or you will see a weird green image.

`tf.summary.image` seems to automatically scale the image into the cardinal range, in the case of `float32`.
