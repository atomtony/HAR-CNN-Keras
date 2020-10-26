# 模型转换

## keras模型转onnx

```
python keras2onnx_demo.py
```

## onnx模型转ncnn

前提是已经编译ncnn（版本ncnn-20181228），然后执行以下命令

```
/home/jht/Downloads/ncnn-20181228/build/tools/onnx/onnx2ncnn \
/home/jht/github/HAR-CNN-Keras/model/onnx/model.onnx \
/home/jht/github/HAR-CNN-Keras/model/ncnn/model.param \
/home/jht/github/HAR-CNN-Keras/model/ncnn/model.bin
```


