# 🎨 ComfyUI-Mosaica

Create colorful mosaic images in [ComfyUI](https://github.com/comfyanonymous/ComfyUI) by computing label images and applying lookup tables.

## Workflow Examples

### K-Means

Generate an image using a stable diffusion model and apply the [k-means clustering algorithm](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) to convert it to a label image. The average color of each cluster is applied to the image's labels and a colorized image is returned.

![kmeans-example](https://github.com/Mason-McGough/ComfyUI-Mosaica/blob/main/workflows/kmeans-example.png)

### Mean Shift

Generate an image using a stable diffusion model and apply the [mean shift clustering algorithm](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html) to convert it to a label image. The average color of each cluster is applied to the image's labels and a colorized image is returned.

![mean-shift-example](https://github.com/Mason-McGough/ComfyUI-Mosaica/blob/main/workflows/mean-shift-example.png)

### Watershed

Generate an image using a stable diffusion model and apply the [watershed segmentation algorithm](https://docs.opencv.org/4.x/d3/d47/group__imgproc__segmentation.html) to convert it to a label image. The average color of each cluster is applied to the image's labels and a colorized image is returned.

![watershed-example](https://github.com/Mason-McGough/ComfyUI-Mosaica/blob/main/workflows/watershed-example.png)

### Random LUT

Apply a randomly generated lookup table of RGB colors to colorize the label image from the mean shift clustering node.

![random-lut-example](https://github.com/Mason-McGough/ComfyUI-Mosaica/blob/main/workflows/random-lut-example.png)

### Load LUT from Matplotlib

Apply a lookup table from Matplotlib to colorize the label image.

![load-lut-from-matplotlib-example](https://github.com/Mason-McGough/ComfyUI-Mosaica/blob/main/workflows/load-lut-from-matplotlib-example.png)

### Label img2img

Apply an img2img with light denoising to the colorized label image.

![label-img2img-example](https://github.com/Mason-McGough/ComfyUI-Mosaica/blob/main/workflows/label-img2img-example.png)

## Nodes

* Mean Shift - Apply the [Mean Shift clustering algorithm](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html) to an image.
* Apply LUT To Label Image - Converts a label image into an RGB image by applying a RGB lookup table (LUT).
* Random LUT - Randomly generate a LUT of RGB colors.
* Load LUT From Matplotlib - Load an RGB LUT from Matplotlib.

## To do

- [ ] implement `LoadLUTFromFile` node
- [ ] implement `MedianFilter` node
- [x] implement `KMeans` node
- [x] implement `Watershed` node
- [ ] implement `Resize Label Image` node
- [ ] add support for Segment Anything labels
- [ ] write unit tests
- [ ] use LAB space in RandomLUT for better perceptual uniformity
- [ ] add random seed option to `RandomLUT`
