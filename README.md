# In-Air-VS-Underwater
This is a project repo specially created for visualizing and comparing different models' performance for underwater datasets. We will include the following:
- Visualization for different encoders
- Visualization of the prediction for models under different settings
- Evaluation for models on different datasets
- Metric explanation on meanings
- Introduction of different datasets
- All related reference links (papers and websites)

<details>
<summary>Datasets</summary>

### Underwater Datasets

#### FLSea Stereo Dataset
- **Size**: Comprises 4 distinct stereo subsets with each subset containing thousands of image pairs, totaling over 7337(3803+2362+867+305) synchronized stereo image pairs. The dataset ptovides both RGB images and dense depth maps.
- **Issues**: The distribution of the subset is pretty skew, as it contains two extreme small subsets(smaller than 1000), and the total number of samples is quite small to be directly used for training and finetuning. On the other hand, depth maps were generated based on SFM techniques, so it contains large missing parts within the depth maps. The most important issue is that it is not a close-up dataset as expected.
- **Content**: Consists of high-resolution RGB underwater images acquired in shallow Mediterranean waters near Israel. Each stereo pair is accompanied by detailed calibration data—including intrinsic and extrinsic camera parameters—and ground truth depth maps generated using photogrammetry. The dataset features diverse underwater scenes showcasing coral reefs, marine flora and fauna, and various natural and man-made structures.
- **Purpose**: Designed to support research in underwater computer vision tasks such as depth estimation, 3D reconstruction, visual odometry, SLAM, obstacle detection, and autonomous underwater navigation. We decide to use it for accessing understandbility of the model for underwater rather than understandbility for close-up scenes.
- **Source**: [FLSea Dataset Publication](https://arxiv.org/abs/2302.12772) | [GTS.AI FLSea Stereo Dataset](https://gts.ai/dataset-download/flsea-stereo-dataset/)
- **Sample Images**:
  
<table>
  <tr>
    <td align="center">First Subset, img id 000006 left</td>
    <td align="center">First Subset, img id 000006 right</td>
    <td align="center">First Subset, img id 000006 absolute left depth</td>
    <td align="center">First Subset, img id 000006 normalized left depth</td>
  </tr>
  <tr>
    <td align="center"><img src="images/01_000006_L.png" alt="First Subset, img id 000006 left" width="400"/></td>
    <td align="center"><img src="images/01_000006_R.png" alt="First Subset, img id 000006 right" width="400"/></td>
    <td align="center"><img src="images/LFT_01_000006_abs_depth.png" alt="First Subset, img id 000006 left" width="400"/></td>
    <td align="center"><img src="images/LFT_01_000006_abs_depth_colored_pure.png" alt="First Subset, img id 000006 right" width="400"/></td>
  </tr>
</table>

### In-Air Stereo Datasets
#### TartanAir (A Dataset to Push the Limits of Visual SLAM)
- **Size**: Approximately 1 million frames (around 4 TB of data) collected from 1037 long motion sequences. It provides left and right RGB image pairs and corresponding depth maps. It also provides pose of the given cameras which may be useful in the future. 
- **Issue**: Similarly, the depth range of Tartan-Air is still too large. For exmaple, the given sample shown below is of 1.94~77.81 meters.
- **Content**: Multi-modal sensor data captured in photo-realistic simulated environments using Unreal Engine and AirSim. Includes synchronized stereo RGB images, depth maps, segmentation labels, optical flow, LiDAR point clouds, and precise camera poses under diverse conditions such as varying lighting, weather, and dynamic scenes. This dataset is much more larger than the FlSEA dataset mentioned above, which has already shown with good performance of monocular depth estimzatin for underwater environment. In other words, it could be used for training some SOTA models from scratch or doing funtuning.
- **Purpose**: To serve as a challenging benchmark for advancing Visual SLAM and robot navigation algorithms by providing extensive, diverse, and high-fidelity data that mimics real-world complexities.
- **Source**: [TartanAir Dataset Website](http://theairlab.org/tartanair-dataset/) | [Paper](https://arxiv.org/abs/2003.14338)
- **Sample Images**:
  
<table>
  <tr>
    <td align="center">amusement, Easy, P001, id000000, left</td>
    <td align="center">amusement, Easy, P001, id000000, right</td>
    <td align="center">amusement, Easy, P001, id000000, left absolute depth</td>
    <td align="center">amusement, Easy, P001, id000000, normalized left depth</td>
  </tr>
  <tr>
    <td align="center"><img src="images/000000_left.png" alt="First Subset, img id 000006 left" width="400"/></td>
    <td align="center"><img src="images/000000_right.png" alt="First Subset, img id 000006 right" width="400"/></td>
    <td align="center"><img src="images/000000_left_depth_raw.png" alt="First Subset, img id 000006 left" width="400"/></td>
    <td align="center"><img src="images/000000_left_depth_norm_color.png" alt="First Subset, img id 000006 right" width="400"/></td>
  </tr>
</table>

#### ScanNet
- **Size**: 1513 scanned indoor scenes with over 2.5 million RGB-D frames. Compared with Tartan-Air, the previous one mainly forcusing on out-door scenes and ScanNet focus more on in-door world, which is more likely to satisfied our close-up settings. The depth maps could be rendered into 16-bit png format, which scale is of millimeter. The sample shown beloew is of depth range 1159mm~3763mm(within 4m)
- **Content**: Rich indoor scene data captured with RGB-D sensors, including RGB images, depth maps, and 3D reconstructed meshes. Each scan is accompanied by detailed semantic annotations for objects and surfaces. 
- **Issues**: ScanNet does not originally provide stereo pairs, but it provides the trajectory of the camera, which is more limited. Besides, it would be a good source for unrectified stereo depth/disparity estimation. The depth maps also got missing values which is shown below(the missing proportion is smaller than FlSEA).
- **Purpose**: To facilitate research in 3D reconstruction, semantic segmentation, object recognition, and overall scene understanding in complex indoor environments.
- **Source**: [Project Website](http://www.scan-net.org/) | [Paper](https://arxiv.org/abs/1702.04405)
- **Sample Images**:
  
<table>
  <tr>
    <td align="center">scene0370_00, id0, the first frame</td>
    <td align="center">scene0370_00, id100, the thrid frame</td>
    <td align="center">scene0370_00, id0 16-bit depth</td>
    <td align="center">scene0370_00, id0 8-bit</td>
    <td align="center">scene0370_00, id0 normalized depth</td>
  </tr>
  <tr>
    <td align="center"><img src="images/0.jpg" alt="First Subset, img id 000006 left" width="400"/></td>
    <td align="center"><img src="images/100.jpg" alt="First Subset, img id 000006 left" width="400"/></td>
    <td align="center"><img src="images/0.png" alt="First Subset, img id 000006 right" width="400"/></td>
    <td align="center"><img src="images/0_depth_raw.png" alt="First Subset, img id 000006 left" width="400"/></td>
    <td align="center"><img src="images/0_colored_pure.png" alt="First Subset, img id 000006 right" width="400"/></td>
  </tr>
</table>


#### Middlebury Stereo Dataset
- **Size**: Approximately 8–15 high-resolution stereo image pairs, with the overall data volume typically under 1 GB.
- **Content**: Consists of rectified stereo RGB image pairs captured in controlled indoor settings, each accompanied by dense and accurate ground truth disparity maps (which can be converted to depth information using camera calibration parameters).
- **Purpose**: Serves as a classical benchmark for stereo matching and depth estimation algorithms, enabling precise evaluation of disparity calculation and 3D reconstruction methods.
- **Source**: [Middlebury Stereo Benchmark](https://vision.middlebury.edu/stereo/)
- **Samples**:
  This dataset only got a few samples, which could be directly preview online.
#### Others:
- [Scene Flow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
- [Falling Things](https://research.nvidia.com/publication/2018-06_Falling-Things)
- [HR-VS](https://drive.google.com/file/d/1SgEIrH_IQTKJOToUwR1rx4-237sThUqX/view)
- [CREStereo Dataset](https://github.com/megvii-research/CREStereo/blob/master/dataset_download.sh)
- [InStereo2K](https://github.com/YuhuaXu/StereoDataset)
- [Middlebury](https://vision.middlebury.edu/stereo/data/)
- [Sintel Stereo](http://sintel.is.tue.mpg.de/stereo)
- [ETH3D](https://www.eth3d.net/datasets#low-res-two-view-training-data)
Here we did not introduce KITTI series datasets as they are really well known.

### In-Air Table-Top Datasets
#### StereOBJ-1M (Large-scale Stereo Image Dataset for 6D Object Pose Estimation)
- **Size**: Over 393K stereo image frames and more than 1.5M 6D pose annotations, collected from 182 scenes across 11 different environments. It is a table-top pose estimation dataset, which would be a good choice for accessing the performance for close-up environment.
- **Content**: Stereo RGB images capturing 18 diverse objects—including symmetric, transparent, and reflective items—under challenging conditions such as occlusion, specularity, and varying illumination. Each image is fully annotated with accurate 6D object pose data, along with instance masks and bounding boxes.
- **Issues**: This dataset does not originally provide ground truth depth, which would be a challenge to train supervised models and doing evaluatio on unsupervised models. As most unsupervised models still doing evaluation on groud truth depth. The stereo pairs are stored in a single image.
- **Purpose**: Designed to advance research in 6D object pose estimation by providing a large-scale, challenging benchmark that addresses common real-world issues like transparency, occlusion, and environmental variations in stereo imagery.
- **Source**: [arXiv:2109.10115](https://arxiv.org/abs/2109.10115) | [Project Website](https://sites.google.com/view/stereobj-1m)
- **Sample Images**:
  
<table>
  <tr>
    <td align="center"><img src="images/562eec275ab81985045d2b5ebfacc5c.jpg" alt="First Subset, img id 000006 left" width="400"/></td>
    <td align="center"><img src="images/711e2bf7950c55f37104332a4b09081.jpg" alt="First Subset, img id 000006 left" width="400"/></td>
  </tr>
</table>

#### STIOS Dataset (Stereo Instances on Surfaces)
- **Size**: The dataset comprises a large collection of high-resolution stereo image pairs; exact numbers can be found on the official webpage, but it typically contains on the order of hundreds to thousands of stereo pairs. It contains two similar subset that capture the same scene with two different cameras. We could regarded as the same. In each subset, there are 8 different scenes with 24 different stereo images each. Besides, the dataset is of table-top settings.
- **Content**: This dataset features high-quality stereo RGB images captured from various surfaces in diverse environments. Each stereo pair is accompanied by detailed annotations including object instance segmentation masks and, where available, depth information, targeting challenging conditions such as occlusion, transparency, and specular reflections. In summary, it contains normals, pcd, depth, RGB, and instanca annotations.
- **Issues**: The overall dataset size is the smallest one up to know, which is only able to use it for accessing the performance of the model(testing only)
- **Purpose**: Designed to advance research in stereo vision, object instance segmentation, and 6D object pose estimation, the STIOS dataset provides a challenging benchmark for developing and evaluating algorithms under real-world conditions.
- **Source**: [Dataset Website](https://www.dlr.de/en/rm/research/publications-and-downloads/datasets/stereoinstancesonsurfaces)
- **Sample Images**:
  <table>
  <tr>
    <td align="center">left image</td>
    <td align="center">right image</td>
    <td align="center">depth map</td>
    <td align="center">pcd map</td>
    <td align="center">normals map</td>
  </tr>
  <tr>
    <td align="center"><img src="images/15b4fee0c421c0520b06d07e474247b.png" alt="First Subset, img id 000006 left" width="400"/></td>
    <td align="center"><img src="images/b28c9aa03bdab0743ab662306387a55.png" alt="First Subset, img id 000006 left" width="400"/></td>
    <td align="center"><img src="images/99bc389f7d8470c15e1db0d1782da20.png" alt="First Subset, img id 000006 right" width="400"/></td>
    <td align="center"><img src="images/a6f636a51592bfe61e01d73e6bd205d.png" alt="First Subset, img id 000006 left" width="400"/></td>
    <td align="center"><img src="images/8d0a71afb047aaf03d4d5bc969bd589.png" alt="First Subset, img id 000006 right" width="400"/></td>
  </tr>
</table>
</details>

<details>
<summary>Evaluation Metrics</summary>
This section describes the evaluation metrics used to assess the performance of depth estimation and stereo matching algorithms.
Besides, during evaluation, we may only evaluate areas that got non-zero groud turth or within a specific value range in order to aviod make evaluation on meaningless areas and raise with high bias due to ground truth abnormal distribution.
For exmaple, we may only evaluate the metrics that within 5m, and masked the rest of the image.
### Standard Depth Metrics

Our evaluation follows standard metrics used in depth estimation literature:

| Metric | Description | Formula | Better | Meaning |
|--------|-------------|---------|--------|---------|
| abs_rel | Absolute Relative Error | $\frac{1}{N} \sum_{i=1}^{N} \frac{\|d_i - \hat{d}_i\|}{\hat{d}_i}$ | Lower | Measures the average relative depth error, normalized by the true depth. Less sensitive to errors in far regions compared to absolute metrics. |
| sq_rel | Squared Relative Error | $\frac{1}{N} \sum_{i=1}^{N} \frac{\|d_i - \hat{d}_i\|^2}{\hat{d}_i}$ | Lower | Emphasizes larger depth errors by squaring the difference. Particularly sensitive to outliers and severe estimation errors. |
| rms | Root Mean Squared Error | $\sqrt{\frac{1}{N} \sum_{i=1}^{N} \|d_i - \hat{d}_i\|^2}$ | Lower | Measures the average magnitude of depth errors in metric units (e.g., meters). Gives higher weight to larger errors. |
| log_rms | Log Root Mean Squared Error | $\sqrt{\frac{1}{N} \sum_{i=1}^{N} \|\log(d_i) - \log(\hat{d}_i)\|^2}$ | Lower | Measures errors in logarithmic space, making it more sensitive to depth errors in close regions while being more tolerant to errors in distant regions. |
| a1 | Threshold Accuracy (delta < 1.25) | $\%$ of $\max(\frac{d_i}{\hat{d}_i}, \frac{\hat{d}_i}{d_i}) < 1.25$ | Higher | Percentage of pixels where the relative error is within 25%. Indicates high-quality depth predictions. |
| a2 | Threshold Accuracy (delta < 1.25²) | $\%$ of $\max(\frac{d_i}{\hat{d}_i}, \frac{\hat{d}_i}{d_i}) < 1.25^2$ | Higher | Percentage of pixels where the relative error is within 56.25%. Provides a more relaxed accuracy measure. |
| a3 | Threshold Accuracy (delta < 1.25³) | $\%$ of $\max(\frac{d_i}{\hat{d}_i}, \frac{\hat{d}_i}{d_i}) < 1.25^3$ | Higher | Percentage of pixels where the relative error is within 95.31%. Identifies regions with significant errors. |
| scale | Scaling Factor | $\text{median}(\hat{d}) / \text{median}(d)$ | - | The ratio used to align prediction and ground truth depths for scale-ambiguous methods (like monocular depth estimation). Not a performance metric but used for analysis. |

where $d_i$ is the predicted depth and $\hat{d}_i$ is the ground truth depth.

### Stereo-Specific Metrics
Note that for stereo matching, we could also use those depth metrics mentioned above, just modify it into comparision between disparity will be enough.
For evaluating stereo matching algorithms, we include:

| Metric | Description | Formula | Better | Meaning |
|--------|-------------|---------|--------|---------|
| EPE-all | End-Point Error | $\frac{1}{N} \sum_{i=1}^{N} \|disp_i - \hat{disp}_i\|$ | Lower | Average absolute disparity error in pixels. Directly measures the accuracy of disparity estimation without converting to depth. |
| D1-all | Disparity Error Rate | $\%$ of pixels where $\|disp_i - \hat{disp}_i\| > 3$ AND $\|disp_i - \hat{disp}_i\| / \hat{disp}_i > 0.05$ | Lower | Percentage of pixels with "significant" disparity errors (>3px absolute AND >5% relative). This is the standard error metric for the KITTI Stereo benchmark. |


### Image Synthesis Metrics

For evaluating image reconstruction quality:

| Metric | Description | Formula | Better | Meaning |
|--------|-------------|---------|--------|---------|
| PSNR | Peak Signal-to-Noise Ratio | $20 \cdot \log_{10}\left(\frac{MAX_I}{\sqrt{MSE}}\right)$ | Higher | Measures the ratio between the maximum possible signal power and the noise power. Higher values indicate better quality. Generally, values above 30dB indicate good reconstruction. |
| SSIM | Structural Similarity Index | $\frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$ | Higher | Measures the perceived similarity between images by considering luminance, contrast, and structure. Values range from 0 to 1, with 1 indicating perfect similarity. More aligned with human perception than PSNR. |
| photo_rmse | Photometric RMSE | $\sqrt{\frac{1}{N} \sum_{i=1}^{N} \|I_i - \hat{I}_i\|^2}$ | Lower | Root mean squared error between pixel values in the reconstructed and ground truth images. Directly measures pixel-wise accuracy of image reconstruction. |

where $I_i$ is the reconstructed image and $\hat{I}_i$ is the ground truth image.

</details>
