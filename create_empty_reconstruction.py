import argparse
import numpy as np
import os
import pycolmap

import matplotlib.pyplot as plt

from colmap import database as colmap_db
from hl24cv.io import load_extrinsics, load_rig2world_transforms

from hloc import extract_features, match_features, visualization, pairs_from_exhaustive
from hloc import reconstruction as hloc_reconstruction
from pathlib import Path
from scipy.spatial.transform import Rotation as R

def camera_id_from_image(db_path, image_id):
    db = colmap_db.COLMAPDatabase.connect(db_path)
    cursor = db.execute("SELECT camera_id from images WHERE image_id = ?;", (image_id,))
    camera_id = cursor.fetchone()[0]
    db.close()
    return camera_id

def update_camera(db_path, camera_id, model, width, height, fx, fy, cx, cy):
    db = colmap_db.COLMAPDatabase.connect(db_path)
    db.execute(
        "UPDATE cameras "
        "SET "
        "model = ?, "
        "width = ?, "
        "height = ?, "
        "params = ? "
        "WHERE camera_id = ?;",
        (model, width, height, colmap_db.array_to_blob(np.array([fx, fy, cx, cy])), camera_id))
    db.commit()
    db.close()

def read_keypoints(db_path, image_id):
    db = colmap_db.COLMAPDatabase.connect(db_path)
    cursor = db.execute("SELECT * FROM keypoints WHERE image_id = ?;", (image_id,))

    row = cursor.fetchone()
    keypoints = colmap_db.blob_to_array(row[3], np.float32, (row[1], row[2]))
    db.close()
    return keypoints

def add_grayscale_images(reconstruction, camera_name, image_base_path, db_path, feature_conf, matcher_conf):
    # Make sure the folders are there
    image_data_path = image_base_path / camera_name
    undist_image_data_path = image_data_path / 'undist'
    if not undist_image_data_path.is_dir():
        print(f"Error - folder {undist_image_data_path} does not exist")
        return

    # We rely on some hloc functions that cannot work with spaces in the image name
    # We use a softlink to avoid the spaces in the camera names as workaround for now
    sanitized_camera_name = camera_name.replace(' ', '_')
    image_path = image_base_path / sanitized_camera_name

    if image_path.exists():
        assert(image_path.is_symlink())
        assert(image_path.resolve(strict=True) == image_data_path)
    else:
        image_path.symlink_to(image_data_path, target_is_directory=True)

    undist_image_path = image_path / 'undist'

    image_file_extension = 'pgm'
    image_paths = undist_image_path.glob('*.' + image_file_extension)
    image_name_list = [str(p.relative_to(image_base_path)) for p in image_paths]

    # TODO: Remove again - just for debugging
    # image_name_list = image_name_list[:10]

    image_ids = image_preprocessing(image_name_list, image_base_path, db_path, feature_conf, matcher_conf)

    # Read the K matrix
    K = np.loadtxt(undist_image_path / 'K.txt')
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]
    image_size = np.loadtxt(undist_image_path / 'image_size.txt')
    image_width, image_height = image_size

    # Read the camera extrinsics
    rig_to_cam_path = image_base_path / (camera_name + '_extrinsics.txt')
    rig_to_world_path = image_base_path / (camera_name + '_rig2world.txt')

    assert(rig_to_cam_path.exists())
    assert(rig_to_world_path.exists())

    # from camera to rig space transformation (fixed)
    rig2cam = load_extrinsics(rig_to_cam_path)

    # from rig to world transformations (one per frame)
    rig2world_transforms = load_rig2world_transforms(rig_to_world_path)

    
    for image_name, image_id in image_ids.items():

        # We need the image timestamps to match to the estimated poses
        # Timestamps are the image names without the camera folder path and file extension
        # TODO: Check how we can make this cross-platform ('/' vs. '\')
        ts_prefix = str(Path(sanitized_camera_name) / 'undist') + '/'
        ts_suffix = '.' + image_file_extension
        
        assert(image_name[:len(ts_prefix)] == ts_prefix)
        assert(image_name[-len(ts_suffix):] == ts_suffix)

        timestamp = float(image_name[len(ts_prefix):-len(ts_suffix)])

        assert(timestamp in rig2world_transforms)

        rig2world = rig2world_transforms[timestamp]

        cam2world = rig2world @ np.linalg.inv(rig2cam)

        # Hololens uses a different coordinate system than colmap.
        # The Z-axis is flipped, meaning forward direction is negative Z,
        # and Y is pointing upwards.
        image2cam = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

        image2world = cam2world @ image2cam

        register_reconstruction_image(image_id, image_name, reconstruction, db_path, image2world, int(image_width), int(image_height), fx, fy, cx, cy)

def image_preprocessing(image_name_list, image_base_path, db_path, feature_conf, matcher_conf):

    # TODO: This assumes a linux filesystem
    # There is a Python 'tempfile' module that could make it cross-platform
    tmp_output_dir = Path('/tmp/hololens_mapping')
    tmp_output_dir.mkdir(exist_ok=True)
    tmp_sfm_pairs_path = tmp_output_dir / 'pairs-sfm.txt'
    tmp_features_path = tmp_output_dir / 'features.h5'
    tmp_matches_path = tmp_output_dir / 'matches.h5'

    pycolmap.import_images(db_path, image_base_path, pycolmap.CameraMode.PER_IMAGE, 'PINHOLE', image_name_list)

    extract_features.main(feature_conf, image_base_path, image_list=image_name_list, feature_path=tmp_features_path)
    pairs_from_exhaustive.main(tmp_sfm_pairs_path, image_list=image_name_list)
    match_features.main(matcher_conf, tmp_sfm_pairs_path, features=tmp_features_path, matches=tmp_matches_path)

    image_ids = hloc_reconstruction.get_image_ids(db_path)

    # Make sure to only import the new data
    new_image_ids = {name:id for name, id in image_ids.items() if name in image_name_list}

    hloc_reconstruction.import_features(new_image_ids, db_path, tmp_features_path)
    hloc_reconstruction.import_matches(new_image_ids, db_path, tmp_sfm_pairs_path, tmp_matches_path, min_match_score=None, skip_geometric_verification=False)
    hloc_reconstruction.geometric_verification(db_path, tmp_sfm_pairs_path)

    # Remove all the files for now to awoid potential bugs
    tmp_sfm_pairs_path.unlink()
    tmp_features_path.unlink()
    tmp_matches_path.unlink()
    # tmp_output_dir.rmdir()

    return new_image_ids

def register_reconstruction_image(image_id, image_name, reconstruction, db_path, image2world, im_width, im_height, fx, fy, cx, cy):
    camera_id = camera_id_from_image(db_path, image_id)
    # use PINHOLE camera model
    camera_model = 1
    update_camera(db_path, camera_id, camera_model, im_width, im_height, fx, fy, cx, cy)

    # TODO: Read the camera from the database, then we can update the DB camera in a separate function (and save all the cam param arguments)
    camera = pycolmap.Camera('PINHOLE', im_width, im_height, [fx, fy, cx, cy], camera_id)

    # TODO: Read from database?
    image = pycolmap.Image(name = image_name, camera_id = camera_id, id = image_id)
    keypoints = read_keypoints(db_path, image_id)
    kp_list = np.split(keypoints.astype(np.float64), keypoints.shape[0], axis=0)
    image.points2D = pycolmap.ListPoint2D([pycolmap.Point2D(np.transpose(p)) for p in kp_list])

    quat = R.from_matrix(image2world[:3,:3]).inv().as_quat()
    # Pycolmap expects the real part of the quaternion in the first position
    image.qvec = np.roll(quat, 1)
    image.tvec = -R.from_matrix(image2world[:3,:3]).inv().as_matrix() @ image2world[:3,3]

    reconstruction.add_camera(camera)
    reconstruction.add_image(image, check_not_registered=True)
    reconstruction.register_image(image_id)

def create_reconstruction(recording_path, model_path):

    # Make sure all the required files are there
    pv_path = recording_path / 'PV'
    if not pv_path.is_dir():
        print(f"Error - folder {pv_path} does not exist")
        return

    # Create model
    reconstruction = pycolmap.Reconstruction()

    # Create database
    db_path = model_path / 'db.db'
    db = colmap_db.COLMAPDatabase(db_path)
    db.create_tables()
    db.commit()
    db.close()

    if reconstruction.num_images() != 0:
        print(f"Error - model already has {reconstruction.num_images()} images - aborting")
        return

    feature_conf = extract_features.confs['superpoint_inloc']
    matcher_conf = match_features.confs['superglue-fast']

    for cam_name in ['VLC LL', 'VLC LF', 'VLC RF', 'VLC RR']:
            add_grayscale_images(reconstruction, cam_name, recording_path, db_path, feature_conf, matcher_conf)

    map_images = pv_path.glob('*.png')
    
    image_path_list = [str(p.relative_to(recording_path)) for p in map_images]

    # TODO: Remove again - just for simpler debugging
    # image_path_list = image_path_list[:5]

    image_ids = image_preprocessing(image_path_list, recording_path, db_path, feature_conf=feature_conf, matcher_conf=matcher_conf)

    # Read the image poses and focal lengths
    pv_file_path = list(Path(recording_path).glob('*pv.txt'))
    assert len(list(pv_file_path)) == 1
    pv_file_path = pv_file_path[0]

    with open(pv_file_path) as f:
        lines = f.readlines()
        static_params = lines[0].split(",")
        cx, cy = [float(x) for x in static_params[:2]]
        width, height = [int(x) for x in static_params[2:]]

        image_positions = np.zeros((len(lines)-1, 3), dtype=np.float32)
        

        for line_idx, line in enumerate(lines[1:]):
            vals = line.split(",")
            # TODO: Do we need to handle other image formats as well?
            image_name = 'PV/' + vals[0] + ".png"

            # TODO Remove again
            if not image_name in image_ids:
                continue

            fx, fy = [float(x) for x in vals[1:3]]
            cam2world = np.array([float(x) for x in vals[3:]])
            cam2world = cam2world.reshape((4,4))

            # Hololens uses a different coordinate system than colmap for the PV camera.
            # The Z-axis is flipped, meaning forward direction is negative Z,
            # and Y is pointing upwards.
            image2cam = np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ])

            image2world = cam2world @ image2cam

            im_id = image_ids[image_name]

            register_reconstruction_image(im_id, image_name, reconstruction, db_path, image2world, width, height, fx, fy, cx, cy)

    print(f"Num images: {reconstruction.num_images()}")

    pycolmap.triangulate_points(reconstruction, db_path, recording_path, str(model_path))
    
    reconstruction.write_text(str(model_path))
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert images')
    parser.add_argument("--recording_path", required=True,
                        help="Path to recording folder")
    parser.add_argument("--output_model_path", required=True,
                        help="Where to store the created model")
    args = parser.parse_args()
    
    create_reconstruction(Path(args.recording_path), Path(args.output_model_path))
