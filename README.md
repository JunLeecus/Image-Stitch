# 图像拼接代码介绍
本代码采用SIFT（可修改为ORB、SURF）提取关键点和特征向量进行点匹配，
而后将图片分割成多个32*32小块，用L2-Net生成特征向量以进行块匹配，两
种方法得到的特征向量进行拼接，共同作为图像的描述向量。
图像处理及拼接部分参考 https://github.com/zhaobenx/Image-stitcher
深度学习生成特征向量参考 https://github.com/virtualgraham/L2-Net-Python-Keras

##用法
修改`stitch.py`文件中477、478行
```python
img1 = cv2.imread("../example/1-left.jpeg")
img2 = cv2.imread("../example/1-right.jpeg")
```
修改图片目录后，运行`stitch.py`

## 主入口
```python
matcher = Matcher(img1, img2, Method.SIFT)
matcher.match(show_match=True)
sticher = Sticher(img1, img2, matcher)
sticher.stich()
```

分为两部分，`Matcher`和`Sticher`，分别用作图像的内容识别及图像的拼接
