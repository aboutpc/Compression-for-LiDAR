# Compression-for-LiDAR

## demo
We support structure demo of proposed model, for ease of use, the demo is deployed in ipynb file, including training and running.

## measurements
We use three main measurements to evaluate the performance of our method: PSNR,SSIM,RMSE. We choose skimage library for concrete implements.

## recovery
The process of point cloud recovery during the training of model is illustrated:

![process of recovery](https://github.com/aboutpc/Compression-for-LiDAR/blob/main/fig/recovery.png)

The point cloud will be gradually recovered from the edge contour to the interior detail.

This process will provide some inspirations.

Note that the value of loss in the figure will be different in different training configuration.

## Other

During our experiments, we also get some unexpected results, like super resolution.

![super resolution](https://github.com/aboutpc/Compression-for-LiDAR/blob/main/fig/super_resolution.png)
