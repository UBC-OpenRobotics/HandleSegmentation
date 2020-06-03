# HandleSegmentation

## Workflow

### Dataset Preparation
After collecting pictures and videos of a bag, place them inside `/collected_data`.

Edit `config.json` to change model and data collection properties. Of note is the `"dev_split": 0.15` definition which sets what percentage of the dataset to use for testing (equivalent to validation split).

Navigate to `/scripts` and run `python prepare_collected_data.py -r ../` to prepare an image dataset. For more details on `prepare_collected_data.py` refer to the [Scripts](#Scripts) section.

`prepare_collected_data.py` will prompt for image augmentation. These are basic rotations and horizontal flipping. Once the script runs, the images will be resized, grayscaled, and some augmented. They will be found in `/train/Images` and `/dev/Images`.

### Image Labelling
Segmentation requires masks as ground truths. These can be generated in [LableMe](http://labelme.csail.mit.edu/Release3.0/) or in [MakeSense.ai](https://www.makesense.ai/).

For mask  generation in LableMe, upload the dataset images and label the handle for each one. Ensure you use the mask tool, not the polygon tool which is default. After labelling download annotations and save these images in `/train/Masks` and `/dev/Masks` accordingly.

For MakeSense, it is easy to drag and drop images into the interface. Label the train and dev set seperately. Using the polygon tool, trace the outline of the handles. After this is done, export lables as VGG JSON. These are sets of points. To convert to masks, which the unet model expects, `cd scripts` and run `python vgg_json_to_mask.py -r ../ -j <path-to-train-json> -d train` for the training set and `python vgg_json_to_mask.py -r ../ -j <path-to-dev-json> -d dev` for the dev set.

Ensure that you change the `"label_method"` parameter in `config.json` to either `"polygon"` if the labelling method used was the VGG JSON output from MakeSense, or `"mask"` if the labels were made in LabelMe. This parameter controls whether to apply morphological transforms to the masks before processing, which LabelMe outputs require.

## Configuration
Most of the configuration for the model is found in `config.json` in the root folder. This section outlies the available parameters and their function. A default `config.json` may look like:

	"base_path":"/home/francisco/openrobotics_ws/src/HandleSegmentation",
	"dataset_path": "collected_data/",
	"figure_path": "figures/",
	"input_w": 384,
	"input_h": 384,
	"collection_frame_skip": 5,
	"dev_split": 0.15,
	"train_img_dir": "train/Images/",
	"dev_img_dir": "dev/Images",
	"model_path": "unet_model",
	"train_mask_dir": "train/Masks/",
	"dev_mask_dir": "dev/Masks",
	"epochs": 500,
	"batch_size": 8,
	"label_method":"polygon"


_base_path_ - Absolute path to the root folder of HandleSegmentation

_dataset_path_ - Relative path to collected data folder, `default="collected_data/"`

_figure_path_ - Relative path to figure folder, `default="figures/"`

_input_w_ - UNet model input width, `default=384`

_input_h_ - UNet model input height, `default=384`

_collection_frame_skip_ - If videos are present in the collected data folder a.k.a _dataset_path_ , then this parameter controls how many frames to skip before saving a frame, `default=5`

_dev_split_ - percentage of collected and augmented data to seperate to dev set, equivalent to validation split, `default=0.15`

_train_img_dir_ - Relative path to where training images are stored, `default="train/Images/"`

_dev_img_dir_ - Relative path to where dev images are stored, `default="dev/Images/"`

_model_path_ - Relative path to save model, as well as model name, `default=unet_model`

_train_mask_dir_ - Relative path to where training labels are stored, `default="train/Masks/"`

_dev_mask_dir_ - Relative path to where dev labels are stored, `default="dev/Masks/"`

_epochs_ - Number of epochs to train, `default=500`

_batch_size_ - Number of images per batch, `default=8`

_label_method_ - One of `polygon` or `mask`, corresponds to labelling method used, refer to [Image Labelling](#Image-Labelling)



## Scripts

_prepare_collected_data.py_ - Script that takes images or videos found in `/collected_data` and creates an image dataset. It also allows for basic image augmentation. It takes three arguments:
<ul>
  <li> -r, --root_path is required and is the path of the root folder of the HandleSegmentation repo        </li>
  <li>-a, --angle_max is the maximum angle of rotation applied during augmentation. the default is angle_max=15</li>
  <li>-rp, --rotation_percent is the percentage of augmented images to flip horizontally, the default is rotation_percent=0.15</li>
</ul>

_vgg_json_to_mask.py_ - Script that takes image annotation in VGG JSON format from MakeSense and converts them into masks:
<ul>
  <li> -r, --root_path is required and is the path of the root folder of the HandleSegmentation repo        </li>
  <li> -j, --json_file is the path to the json file being processed</li>
  <li>-d, --dest is the destination, so either dev or train</li>
</ul>
