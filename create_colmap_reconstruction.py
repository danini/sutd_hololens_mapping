import argparse
import numpy as np
import pycolmap

from colmap import database as colmap_db
from hl24cv.io import load_extrinsics, load_rig2world_transforms

from hloc import extract_features, match_features, pairs_from_exhaustive, pairs_from_poses
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

def read_db_camera(db_path, camera_id):
    db = colmap_db.COLMAPDatabase.connect(db_path)
    cursor = db.execute(
        "SELECT * "
        "FROM cameras "
        "WHERE camera_id = ?;",
        (camera_id,))
    
    row = cursor.fetchone()
    db_cam_id, model, width, height, params, prior = row
    params = colmap_db.blob_to_array(params, np.float64)

    assert(db_cam_id == camera_id)

    # We only handle the pinhole model right now
    assert(model == 1)

    db.close()
    fx = params[0]
    fy = params[1]
    cx = params[2]
    cy = params[3]
    camera = pycolmap.Camera('PINHOLE', width, height, [fx, fy, cx, cy], camera_id)

    return camera

def read_db_image(db_path, image_id):
    db = colmap_db.COLMAPDatabase.connect(db_path)
    cursor = db.execute(
        "SELECT image_id, name, camera_id "
        "FROM images "
        "WHERE image_id = ?;",
        (image_id,))
    
    row = cursor.fetchone()
    db_image_id, image_name, camera_id = row
    db.close()

    assert(db_image_id == image_id)

    image = pycolmap.Image(name = image_name, camera_id = camera_id, id = image_id)

    return image

def read_keypoints(db_path, image_id):
    db = colmap_db.COLMAPDatabase.connect(db_path)
    cursor = db.execute("SELECT * FROM keypoints WHERE image_id = ?;", (image_id,))

    row = cursor.fetchone()
    keypoints = colmap_db.blob_to_array(row[3], np.float32, (row[1], row[2]))
    db.close()
    return keypoints

def split_image_calib(image_calib):
    image_names = []
    image_poses = []
    image_calibs = []

    for image_name, data in image_calib.items():
        image2world, image_width, image_height, fx, fy, cx, cy = data

        image_names.append(image_name)
        image_poses.append(image2world)
        image_calibs.append((image_width, image_height, fx, fy, cx, cy))

    return image_names, image_poses, image_calibs


def get_gs_cam_images(camera_name, image_base_path):
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

    # TODO: This actually returns false if the symlink exists but points to an invalid path
    if image_path.exists():
        assert(image_path.is_symlink())
        # Make sure we also capture symlinks in image_data_path
        assert(image_path.resolve(strict=True) == image_data_path.resolve(strict=True))
    else:
        image_path.symlink_to(image_data_path, target_is_directory=True)

    undist_image_path = image_path / 'undist'

    image_file_extension = 'pgm'
    image_paths = undist_image_path.glob('*.' + image_file_extension)
    image_name_list = [str(p.relative_to(image_base_path)) for p in image_paths]

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

    image_calib = {}
    for image_name in image_name_list:

        # We need the image timestamps to match to the estimated poses
        # Timestamps are the image names without the camera folder path and file extension
        # TODO: Check how we can make this cross-platform ('/' vs. '\')
        ts_prefix = str(Path(sanitized_camera_name) / 'undist') + '/'
        ts_suffix = '.' + image_file_extension
        
        assert(image_name[:len(ts_prefix)] == ts_prefix)
        assert(image_name[-len(ts_suffix):] == ts_suffix)

        timestamp = float(image_name[len(ts_prefix):-len(ts_suffix)])

        # Apparently this happens (rarely)
        if not timestamp in rig2world_transforms:
            print(f"Error - could not find timestamp {timestamp} in rig poses (rig2world).")
            continue

        rig2world = rig2world_transforms[timestamp]

        cam2world = rig2world @ np.linalg.inv(rig2cam)

        # For the grayscale cameras the coordinate system is already as we need it
        image2cam = np.eye(4)

        image2world = cam2world @ image2cam

        image_calib[image_name] = (image2world, image_width, image_height, fx, fy, cx, cy)

    # We want to make sure that the 3 lists (image name, pose, calibration) are consistent
    # Therefore we recreate the image list from the dict
    image_names, image_poses, image_calibs = split_image_calib(image_calib)

    return image_names, image_poses, image_calibs

def get_pv_cam_images(pv_file_path, image_base_path):

    image_calib = {}
    with open(pv_file_path) as f:
        lines = f.readlines()
        static_params = lines[0].split(",")
        cx, cy = [float(x) for x in static_params[:2]]
        width, height = [int(x) for x in static_params[2:]]

        for line_idx, line in enumerate(lines[1:]):
            vals = line.split(",")
            # TODO: Do we need to handle other image formats as well?
            image_name = 'PV/' + vals[0] + ".png"

            # Only use images that are actually there
            if not (image_base_path / image_name).exists():
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

            image_calib[image_name] = (image2world, width, height, fx, fy, cx, cy)

    # We want to make sure that the 3 lists (image name, pose, calibration) are consistent
    # Therefore we recreate the image list from the dict
    image_names, image_poses, image_calibs = split_image_calib(image_calib)

    return image_names, image_poses, image_calibs


def register_image(image_id, reconstruction, db_path, image2world):
    camera_id = camera_id_from_image(db_path, image_id)

    camera = read_db_camera(db_path, camera_id)
    image = read_db_image(db_path, image_id)
    keypoints = read_keypoints(db_path, image_id)
    kp_list = np.split(keypoints.astype(np.float64), keypoints.shape[0], axis=0)
    image.points2D = pycolmap.ListPoint2D([pycolmap.Point2D(np.transpose(p)) for p in kp_list])
    print(f"Added {len(kp_list)} keypoints to image {image_id}")
    assert(len(kp_list) > 0)

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

    feature_conf = extract_features.confs['superpoint_aachen']
    matcher_conf = match_features.confs['superglue-fast']

    cam_images = {}
    for cam_name in ['VLC LL', 'VLC LF', 'VLC RF', 'VLC RR']:
        image_names, image_poses, image_calibs = get_gs_cam_images(cam_name, recording_path)
        cam_images[cam_name] = (image_names, image_poses, image_calibs)

    # Read the file with all extra PV image data
    pv_file_path = list(Path(recording_path).glob('*pv.txt'))
    assert len(list(pv_file_path)) == 1
    pv_file_path = pv_file_path[0]

    image_names, image_poses, image_calibs = get_pv_cam_images(pv_file_path, recording_path)
    cam_images['PV'] = (image_names, image_poses, image_calibs)

    all_image_names = []
    all_cam_calibs = []
    for _, image_data in cam_images.items():
        # # TODO: Only use a few images per cam for debugging, remove again
        # num_cam_imgs = 50
        # all_image_names += image_data[0][:num_cam_imgs]
        # all_cam_calibs += image_data[2][:num_cam_imgs]
        all_image_names += image_data[0]
        all_cam_calibs += image_data[2]

    pycolmap.import_images(db_path, recording_path, pycolmap.CameraMode.PER_IMAGE, 'PINHOLE', all_image_names)
    image_ids = hloc_reconstruction.get_image_ids(db_path)

    # Update the cameras in the database with the known calibrations
    assert(len(all_image_names) == len(all_cam_calibs))
    for i in range(len(all_image_names)):
        image_name = all_image_names[i]
        image_id = image_ids[image_name]
        camera_id = camera_id_from_image(db_path, image_id)

        im_width, im_height, fx, fy, cx, cy = all_cam_calibs[i]

        # use PINHOLE camera model
        camera_model = 1
        update_camera(db_path, camera_id, camera_model, im_width, im_height, fx, fy, cx, cy)

    # TODO: This assumes a linux filesystem
    # There is a Python 'tempfile' module that could make it cross-platform
    tmp_output_dir = Path('/tmp/hololens_mapping')
    tmp_output_dir.mkdir(exist_ok=True)
    tmp_sfm_pairs_path = tmp_output_dir / 'pairs-sfm.txt'
    tmp_features_path = tmp_output_dir / 'features.h5'
    tmp_matches_path = tmp_output_dir / 'matches.h5'

    # Extract features
    image_name_list = image_ids.keys()
    extract_features.main(feature_conf, recording_path, image_list=image_name_list, feature_path=tmp_features_path)
    hloc_reconstruction.import_features(image_ids, db_path, tmp_features_path)

    for cam_name in ['VLC LL', 'VLC LF', 'VLC RF', 'VLC RR', 'PV']:
        cam_image_names, cam_image_poses, _ = cam_images[cam_name]
        # cam_images[cam_name] = (image_names, image_poses, image_calibs)
        for i in range(len(cam_image_names)):
            
            image_name = cam_image_names[i]

            # When we only use a subset of images, make sure to only add those.
            if not image_name in image_ids:
                continue

            image2world = cam_image_poses[i]
            image_id = image_ids[image_name]
            register_image(image_id, reconstruction, db_path, image2world)


    empty_model_path = model_path / 'empty'
    empty_model_path.mkdir(exist_ok=True)
    # Pairs from poses expects a binary model
    reconstruction.write(str(empty_model_path))

    pairs_from_poses.main(empty_model_path, tmp_sfm_pairs_path, min(10, len(image_ids)), rotation_threshold=45)

    match_features.main(matcher_conf, tmp_sfm_pairs_path, features=tmp_features_path, matches=tmp_matches_path)

    hloc_reconstruction.import_matches(image_ids, db_path, tmp_sfm_pairs_path, tmp_matches_path, min_match_score=None, skip_geometric_verification=False)
    hloc_reconstruction.geometric_verification(db_path, tmp_sfm_pairs_path)

    # Remove all the files for now to avoid bugs when changing the input
    tmp_sfm_pairs_path.unlink()
    tmp_features_path.unlink()
    tmp_matches_path.unlink()

    reconstruction = pycolmap.triangulate_points(reconstruction, db_path, recording_path, str(model_path))
    print(reconstruction.summary())
    
    debug_final_model_path = model_path / 'final'
    debug_final_model_path.mkdir(exist_ok=True)
    reconstruction.write_text(str(debug_final_model_path))
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert images')
    parser.add_argument("--recording_path", required=True,
                        help="Path to recording folder")
    parser.add_argument("--output_model_path", required=True,
                        help="Where to store the created model")
    args = parser.parse_args()
    
    create_reconstruction(Path(args.recording_path), Path(args.output_model_path))
