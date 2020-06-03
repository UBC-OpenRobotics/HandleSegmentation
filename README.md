# HandleSegmentation

## Workflow

### Dataset Preparation
After collecting pictures and videos of a bag, place them inside `/collected_data`.

Edit `config.json` to change model and data collection properties. Of note is the `"dev_split": 0.15` definition which sets what percentage of the dataset to use for testing.

Navigate to `/scripts` and run `python prepare_collected_data.py -r ../` to prepare an image dataset. For more details on `prepare_collected_data.py` refer to the [Scripts](#Scripts) section.

`prepare_collected_data.py` will prompt for image augmentation. These are basic rotations and horizontal flipping. Once the script runs, the images will be resized, grayscaled, and some augmented. They will be found in `/train/Images` and `/dev/Images`.

### Image Labelling
Segmentation requires masks as ground truths. These can be generated in [LableMe](http://labelme.csail.mit.edu/Release3.0/). 

For mask  generation, upload the dataset images and label the handle for each one. Ensure you use the mask tool, not the polygon tool which is default. After labelling download annotations and save these images in `/train/Masks` and `/dev/Masks` accordingly.

## Scripts

_prepare_collected_data.py_ - Script that takes images or videos found in `/collected_data` and creates an image dataset. It also allows for basic image augmentation. It takes three arguments:
<ul>
  <li> -r, --root_path is required and is the path of the root folder of the HandleSegmentation repo        </li>
  <li>-a, --angle_max is the maximum angle of rotation applied during augmentation. the default is angle_max=15</li>
  <li>-rp, --rotation_percent is the percentage of augmented images to flip horizontally, the default is rotation_percent=0.15</li>
</ul>
