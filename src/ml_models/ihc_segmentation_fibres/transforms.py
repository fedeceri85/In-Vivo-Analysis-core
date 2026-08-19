from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd,  EnsureTyped, Lambdad
# Padding of different size images
from monai.transforms import SpatialPadd, RandSpatialCropd,RandCropByPosNegLabeld # RandCrop preserve a even split between negative and positive predominant images
import torch

from monai.transforms import MapTransform, DeleteItemsd,RandRotate90d, RandScaleIntensityd, RandShiftIntensityd, RandFlipd
import tifffile
import numpy as np

class LoadTiffd(MapTransform):

    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):

        d = dict(data)

        for key in self.keys:
            d[key] = tifffile.imread(d[key])

        return d
    
class BinarizeMaskd(MapTransform):

    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)

        for key in self.keys:
            d[key] = (d[key] > 0).astype(np.float32)

        return d


class FilterByAnnotationd(MapTransform):

    def __init__(self, mask_key="mask", annotation_key="annotation",annotation_indices=[1,2]):
        super().__init__([mask_key, annotation_key])
        self.mask_key = mask_key
        self.annotation_key = annotation_key
        self.annotation_indices = annotation_indices

    def __call__(self, data):

        d = dict(data)

        mask = d[self.mask_key]
        ann = d[self.annotation_key]

        output = np.zeros_like(mask)

        for obj_id in np.unique(mask):

            if obj_id == 0:
                continue

            obj_pixels = mask == obj_id

            # majority annotation of this object
            label = np.bincount(
                ann[obj_pixels].astype(int)
            ).argmax()

            if (label in self.annotation_indices):
                output[obj_pixels] = obj_id

        d[self.mask_key] = output

        return d

def get_transforms_binary_classification(
        num_samples,
        spatial_size,
        pos,
        neg,
        annotation_indices,
        flip_vertical,
        flip_vertical_prob,
        flip_horizontal,
        flip_horizontal_prob,
        rotate90,
        rotate90_prob,
        rotate90_max_k,
        intensity_scale,
        intensity_scale_prob,
        intensity_scale_factors,
        intensity_shift,
        intensity_shift_prob,
        intensity_shift_offsets,
):
    spatial_size = tuple(spatial_size)

    transforms_list = [

        LoadTiffd(
            keys=["image", "label", "annotation"]
        ),

        # Convert instance labels 0,1,2,3... -> binary 0/1
        # Lambdad(
        #     keys=["label"],
        #     func=binarize_mask
        # ),


        FilterByAnnotationd(
            mask_key="label",
            annotation_key="annotation",
            annotation_indices=annotation_indices
        ),

        #Don't need annotations after this, remove it so we don't have to crop it 
        DeleteItemsd(
            keys=["annotation"]

        ),

        BinarizeMaskd(
            keys=["label"]
        ),

        EnsureChannelFirstd(
            keys=["image", "label"],
            channel_dim="no_channel"
        ),


        ScaleIntensityd(
            keys=["image"]
        ),

        EnsureTyped(
            keys=["image", "label"],
            dtype=torch.float32
        ),
        #Ensure that all images are the same size, pad with zeros if necessary
        SpatialPadd(
            keys=["image", "label"],
            spatial_size=spatial_size,
        ),

        #Crop the images to a fixed size, with a random position, but ensure that at least one positive label is included in the crop
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=spatial_size,
            pos=pos,
            neg=neg,
            num_samples=num_samples,
        ),

        
    ]

    #Data augmentation
    if flip_vertical:
        transforms_list.append( 
        RandFlipd(
            keys=["image", "label"],
            prob=flip_vertical_prob,
            spatial_axis=0,
        ))

    if flip_horizontal:
        transforms_list.append(
        RandFlipd(
            keys=["image", "label"],
            prob=flip_horizontal_prob,
            spatial_axis=1,
        ))
    if rotate90:
        transforms_list.append( 
            RandRotate90d(
                keys=["image", "label"],
                prob=rotate90_prob,
                max_k=rotate90_max_k,
            ))
    if intensity_scale:
        transforms_list.append(
            RandScaleIntensityd(
                keys=["image"],
                factors=intensity_scale_factors,
                prob=intensity_scale_prob,
            ))
    if intensity_shift:
        transforms_list.append(
            RandShiftIntensityd(
                keys=["image"],
                offsets=intensity_shift_offsets,
                prob=intensity_shift_prob,
            ))

    train_transforms = Compose(transforms_list)


    # Transform for validation data. This does not have random crops. To account for the full image, we will use a sliding prediction
    val_transforms = Compose([
        LoadTiffd(keys=["image", "label", "annotation"]),
        
    

        FilterByAnnotationd(
            mask_key="label",
            annotation_key="annotation",
            annotation_indices=annotation_indices
        ),
        DeleteItemsd(

            keys=["annotation"]

        ),

        BinarizeMaskd(keys=["label"]),

        EnsureChannelFirstd(
            keys=["image", "label"],
            channel_dim="no_channel"
        ),

        ScaleIntensityd(keys=["image"]),

        EnsureTyped(
            keys=["image", "label"],
            dtype=torch.float32
        ),
    ])

    return train_transforms, val_transforms
