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
