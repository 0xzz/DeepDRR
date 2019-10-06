import pydicom as dicom
import numpy as np
from skimage.transform import resize
import segmentation
import matplotlib.pyplot as plt

materials = {1: "air", 2: "soft tissue", 3: "cortical bone"}


def through_plane_location(dicom_file):
    """
    Gets spatial coordinate of image origin whose axis is perpendicular to image plane.

    :param dicom_file: a pydicom.dataset.FileDataset object or the full path to the DICOM file
    :return: through-plane location
    """
    if not isinstance(dicom_file, dicom.dataset.FileDataset):
        dicom_file = dicom.dcmread(dicom_file)

    assert dicom_file.Modality == "CT" or dicom_file.Modality == "MR", "Only CT and MR slices are supported."

    orientation = tuple((float(o) for o in dicom_file.ImageOrientationPatient))
    position = tuple((float(p) for p in dicom_file.ImagePositionPatient))
    rowvec, colvec = orientation[:3], orientation[3:]  # rowvec: X-axis / colvec: Y-axis
    normal_vector = np.cross(rowvec, colvec)
    slice_pos = np.dot(position, normal_vector)
    return slice_pos


def load_dicom(source_path=r"./*/*/", fixed_slice_thinckness=None, new_resolution=None, truncate=None, smooth_air=False,
               use_thresholding_segmentation=False):
    unordered_datasets = []
    through_plane_locations = []
    for file in source_path.glob('**/*'):
        try:
            one_slice = dicom.read_file(str(file))
            if one_slice.Modality == 'CT' or one_slice.Modality == 'MR':
                unordered_datasets.append(one_slice)
                through_plane_locations.append(through_plane_location(one_slice))
            else:
                print(
                    f"Skipped parsing 'FrameOfReferenceUID' for unsupported modality {one_slice.Modality} at {file}.")
        except:
            print(f'Skipped unparsable file {file}.')

    datasets = [ds for _, ds in sorted(zip(through_plane_locations, unordered_datasets))]

    # Get ref file
    refDs = datasets[0]

    # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
    volume_size = [int(refDs.Rows), int(refDs.Columns), len(datasets)]

    if not hasattr(refDs, "SliceThickness"):
        print('Volume has no attribute Slice Thickness, please provide it manually!')
        print('using fixed slice thickness of:', fixed_slice_thinckness)
        voxel_size = [float(refDs.PixelSpacing[1]), float(refDs.PixelSpacing[0]), fixed_slice_thinckness]
    else:
        voxel_size = [float(refDs.PixelSpacing[1]), float(refDs.PixelSpacing[0]), float(refDs.SliceThickness)]

    # The array is sized based on 'PixelDims'
    volume = np.zeros(volume_size, dtype=np.float64)

    # loop through all the DICOM files
    for i, ds in enumerate(datasets):
        # store the raw image data
        volume[:, :, i] = ds.pixel_array.astype(np.int32)

    #use intercept point
    if hasattr(refDs, "RescaleIntercept"):
        volume += int(refDs.RescaleIntercept)

    volume = np.moveaxis(volume, [0, 1, 2], [1, 0, 2]).copy()

    #truncate
    if truncate:
        volume = volume[truncate[0][0]:truncate[0][1], truncate[1][0]:truncate[1][1], truncate[2][0]:truncate[2][1]]

    # volume = np.flip(volume,2)
    #upsample Volume
    if new_resolution:
        volume, volume_size, voxel_size = upsample(volume, new_resolution, voxel_size)

    #convert hu_values to density
    densities = conv_hu_to_density(volume, smoothAir=smooth_air)

    #convert hu_values to materials
    if not use_thresholding_segmentation:
        materials = conv_hu_to_materials(volume)
    else:
        materials = conv_hu_to_materials_thresholding(volume)

    return densities.astype(np.float32), materials, np.array(voxel_size, dtype=np.float32)


def upsample(volume, newResolution, voxelSize):
    upsampled_voxel_size = list(np.array(voxelSize) * np.array(volume.shape) / newResolution)
    upsampled_volume = resize(volume, newResolution, order=1, cval=-1000)
    return upsampled_volume, upsampled_voxel_size, upsampled_voxel_size


def conv_hu_to_density(hu_values, smoothAir=False):
    #Use two linear interpolations from data: (HU,g/cm^3)
    # use for lower HU: density = 0.001029*HU + 1.03
    # use for upper HU: density = 0.0005886*HU + 1.03

    #set air densities
    if smoothAir:
        hu_values[hu_values <= -900] = -1000;
    #hu_values[hu_values > 600] = 5000;
    densities = np.maximum(np.minimum(0.001029 * hu_values + 1.030, 0.0005886 * hu_values + 1.03), 0);
    return densities


def conv_hu_to_materials_thresholding(hu_values):
    print("segmenting volume with thresholding")
    materials = {}
    # Air
    materials["air"] = hu_values <= -800
    # Soft Tissue
    materials["soft tissue"] = (-800 < hu_values) * (hu_values <= 350)
    # Bone
    materials["bone"] = (350 < hu_values)

    return materials


def conv_hu_to_materials(hu_values):
    print("segmenting volume with Vnet")
    segmentation_network = segmentation.SegmentationNet()
    materials = segmentation_network.segment(hu_values)

    return materials
