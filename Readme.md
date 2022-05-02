3D mapping based on the Hololens research mode StreamRecorderApp output.

# Prerequisites
The mapping code assumes Python >=3.7

1. The adapted version of the Hololens research mode sample repository at https://github.com/mgprt/HoloLens2ForCV ('main' branch)
1. The adapted version of the Hierarchical Localization repository at https://github.com/mgprt/Hierarchical-Localization ('fix_pairs_from_poses' branch). See the Readme file for instructions how to install the modules.
1. This repository

# Recording and data preparation (Hololens2forCV repo)
1. Deploy the Adapted `StreamRecorderApp` to hololens.
1. Record data and download it from hololens
1. Run `convert_images.py` to convert the RGB ('PV') images to actual image files
1. Run `undistort_images.py` to undistort the grayscale images based on the provided lookup tables.

# Sparse mapping
1. Run `create_colmap_reconstruction.py`. The `--recording_path` argument should point to the main folder. The created model will be written to `--output_model_path`. (Needs to be an existing directory path)

# Dense mapping
1. Run the `colmap image_undistorter`
1. Run the `colmap patch_match_stereo` (`--PatchMatchStereo.geom_consistency false` seems to work better)
1. Run `colmap stereo_fusion`


# Sparse mapping overview
The script `create_colmap_reconstruction.py` automatically creates a sparse COLMAP reconstruction based on the RGB and grayscale images recorded by hololens.
It uses the reported image poses as priors to initialize the reconstruction, but optimizes the poses in a BA step afterwards.
The script extracts Superpoint features, and determines image pairs to be matched with Superglue based on overlapping frustrums according to the prior poses (TODO: Either rotate all GS images upright and adapt the poses accordingly or switch to different, rotation invariant features -> far right/left cameras are not just rotated in 90deg steps, so rotation invariance might actually lead to more matches).
After triangulating points from the images with prior poses, we perform multiple iterations of bundle adjustment (while keeping the intrinsic camera parameters fixed), filtering, and retriangulation.