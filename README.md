# Compression-for-LiDAR

## demo
We support demo of example models, including the baseline model and the model introduced with curvature (In this demo, k=4). They are pretrained in several scenarios.

For simplicity, the demo is deployed in ipynb file.

## visualization
The recovered result of demo can be visualized in the ipynb file, there are also many point cloud processing libraries and softwares optional to visualize the results.  

## measurements
We use three main measurements to evaluate the performance of our method: PSNR,SSIM,RMSE. We choose skimage library for concrete implements.

## recovery
The process of point cloud recovery during the training of model is illustrated:

![process of recovery](https://github.com/aboutpc/Compression-for-LiDAR/blob/main/fig/recovery.png)

The point cloud will be gradually recovered from the edge contour to the interior detail.

This process will provide some inspirations.

Note that the value of loss in the figure will be different in different training configuration.

## Loss and Curvature

For ablation studies, we also have some results of different changes to the network.

![Loss and Curvature](https://github.com/aboutpc/Compression-for-LiDAR/blob/main/ablation_study/loss_and_curvature.png)

## Normalization

We also test the results of normalization, but we do not pay much attention in this part.

![Normalization](https://github.com/aboutpc/Compression-for-LiDAR/blob/main/ablation_study/normalization.png)

## Other

During our experiments, we also get some unexpected results, like super resolution.

![super resolution](https://github.com/aboutpc/Compression-for-LiDAR/blob/main/fig/super_resolution.png)
