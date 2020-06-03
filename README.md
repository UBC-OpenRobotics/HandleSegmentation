# HandleSegmentation

## Scripts

_prepare_collected_data.py_ - Script that takes images or videos found in `collected_data` and creates an image dataset. It also allows for basic image augmentation. It takes three arguments:
<ul>
  <li> -r, --root_path is required and is the path of the root folder of the HandleSegmentation repo        </li>
  <li>-a, --angle_max is the maximum angle of rotation applied during augmentation. the default is angle_max=15</li>
  <li>-rp, --rotation_percent is the percentage of augmented images to flip horizontally, the default is rotation_percent=0.15</li>
</ul>
