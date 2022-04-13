import argparse
import numpy as np
import os
import pycolmap

from colmap import database as colmap_db
from hloc import extract_features, match_features, visualization, pairs_from_exhaustive
from hloc import reconstruction as hloc_reconstruction
from hloc.visualization import plot_images, read_image
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
        #     keypoints = np.asarray(keypoints, np.float32)
        # self.execute(
        #     "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
        #     (image_id,) + keypoints.shape + (array_to_blob(keypoints),))

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

    # Extract image features and match
    images = pv_path

    outputs = Path('/tmp/outputs/demo/')
    sfm_pairs = outputs / 'pairs-sfm.txt'
    loc_pairs = outputs / 'pairs-loc.txt'
    sfm_dir = outputs / 'sfm'
    features = outputs / 'features.h5'
    matches = outputs / 'matches.h5'

    feature_conf = extract_features.confs['superpoint_aachen']
    matcher_conf = match_features.confs['superglue']

    map_images = images.glob('*.png')
    
    image_path_list = [str(p.relative_to(images)) for p in map_images]

    # TODO: Remove again - just for simpler debugging
    image_path_list = image_path_list[:100]
    
    ri = [read_image(images / Path(r)) for r in image_path_list[:5]]
    plot_images(ri, dpi=25)

    pycolmap.import_images(db_path, pv_path, pycolmap.CameraMode.PER_IMAGE, 'PINHOLE', image_path_list)

    extract_features.main(feature_conf, images, image_list=image_path_list, feature_path=features)
    pairs_from_exhaustive.main(sfm_pairs, image_list=image_path_list)
    match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)

    image_ids = hloc_reconstruction.get_image_ids(db_path)
    hloc_reconstruction.import_features(image_ids, db_path, features)
    hloc_reconstruction.import_matches(image_ids, db_path, sfm_pairs, matches, min_match_score=None, skip_geometric_verification=False)

    # Read the image poses and focal lengths
    pv_file_path = list(Path(recording_path).glob('*pv.txt'))
    assert len(list(pv_file_path)) == 1
    pv_file_path = pv_file_path[0]

    with open(pv_file_path) as f:
        lines = f.readlines()
        static_params = lines[0].split(",")
        cx, cy = [float(x) for x in static_params[:2]]
        width, height = [int(x) for x in static_params[2:]]
        

        for line_idx, line in enumerate(lines[1:]):
            vals = line.split(",")
            # TODO: Do we need to handle other image formats as well?
            image_name = vals[0] + ".png"

            # TODO Remove again
            if not image_name in image_ids:
                continue

            fx, fy = [float(x) for x in vals[1:3]]
            pose = np.array([float(x) for x in vals[3:]])
            pose = pose.reshape((4,4))
            # image_name, fx, fy, pose = line.split(",")
            print(f"Image name: {image_name}")
            print(f"Image pose: {pose}")



            im_id = image_ids[image_name]

            camera_id = camera_id_from_image(db_path, im_id)
            # use PINHOLE camera model
            camera_model = 1
            update_camera(db_path, camera_id, camera_model, width, height, fx, fy, cx, cy)

            camera = pycolmap.Camera('PINHOLE', width, height, [fx, fy, cx, cy], camera_id)

            # image = pycolmap.Image(os.path.join(recording_path, 'PV', image_name + '.png'), [], R.from_matrix(pose[:3,:3]).as_quat(), pose[:3,3], im_id)
            image = pycolmap.Image(name = image_name, camera_id = camera_id, id = im_id)
            keypoints = read_keypoints(db_path, im_id)
            kp_list = np.split(keypoints.astype(np.float64), keypoints.shape[0], axis=0)
            image.points2D = pycolmap.ListPoint2D([pycolmap.Point2D(np.transpose(p)) for p in kp_list])
            # image.name = os.path.join(recording_path, 'PV', image_name + '.png')
            # image.camera_id = im_id
            image.qvec = R.from_matrix(pose[:3,:3]).inv().as_quat()
            image.tvec = -R.from_matrix(pose[:3,:3]).inv().as_matrix() @ pose[:3,3]


            reconstruction.add_camera(camera)
            reconstruction.add_image(image, check_not_registered=True)
            reconstruction.register_image(im_id)

            print(f"Num images: {reconstruction.num_images()}")



    
    reconstruction.write_text(str(model_path))
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert images')
    parser.add_argument("--recording_path", required=True,
                        help="Path to recording folder")
    parser.add_argument("--output_model_path", required=True,
                        help="Where to store the created model")
    args = parser.parse_args()
    
    create_reconstruction(Path(args.recording_path), Path(args.output_model_path))
