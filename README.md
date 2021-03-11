# Pix2Vox

![](https://ai-studio-static-online.cdn.bcebos.com/2c150c364b4945ad89a93a9d012deeb30c699bd8f4ee4475ba81f62fa7f5af73)

This is a PaddlePaddle2.0 implementation of the paper [《Pix2Vox: Context-aware 3D Reconstruction from Single and Multi-view Images》](https://arxiv.org/pdf/1901.11153.pdf) 

See the [official repo](https://github.com/hzxie/Pix2Vox) in Pytorch, as well as overview of the method.

AI Studio Notebook.

**differences**

1. In order to maintain the same network structure as the original paper, this experiment used vgg16_bn without pre-trained(PaddlePaddle has vgg16 model pre-trained on ImageNet, but no pre-trained vgg16_bn model. ).

This may have affected the experimental results.

# Dataset

Use the same dataset as mentioned in the official repo.

--ShapeNet rendered images [http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz](http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz)

--ShapeNet voxelized models [http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz](http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz)

--Pix3D images & voxelized models: [http://pix3d.csail.mit.edu/data/pix3d.zip](http://pix3d.csail.mit.edu/data/pix3d.zip)

The dataset is already mounted in this notebook.


```python
!unzip -oq data/data67155/dataset.zip
```

# Install Python Denpendencies


```python
%cd work/Pix2Vox-F/
!pip install -r requirements.txt
```

# Pix2Vox-F

![](https://ai-studio-static-online.cdn.bcebos.com/0c30a61d6c594cd28616662beb2bbd4615ddec73fa85421fbb7f22e6ea425bbc)



```python
%cd work/Pix2Vox-F/
```


```python
# train
!python runner.py
```


```python
# test
!python3 runner.py --test --weights=/path/to/best_checkpoint
```

# Pix2Vox-A

![](https://ai-studio-static-online.cdn.bcebos.com/3ad4eb99a01d4c5fbf225468f67ba0b86517bdb3e8f74fc68756657abb8d6bff)



```python
%cd work/Pix2Vox-A/
```


```python
# train
!python runner.py
```


```python
# test
!python3 runner.py --test --weights=/path/to/best_checkpoint
```
