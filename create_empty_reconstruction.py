import argparse
import numpy as np
import os
import pycolmap

from pathlib import Path
from scipy.spatial.transform import Rotation as R

def create_reconstruction(recording_path, model_path):
    # Make sure all the required files are there
    pv_path = os.path.join(recording_path, 'PV')
    if not os.path.isdir(pv_path):
        print(f"Error - folder {pv_path} does not exist")
        return

    # Create model
    reconstruction = pycolmap.Reconstruction()

    if reconstruction.num_images() != 0:
        print(f"Error - model already has {reconstruction.num_images()} images - aborting")
        return

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
            image_name = vals[0]
            fx, fy = [float(x) for x in vals[1:3]]
            pose = np.array([float(x) for x in vals[3:]])
            pose = pose.reshape((4,4))
            # image_name, fx, fy, pose = line.split(",")
            print(f"Image name: {image_name}")
            print(f"Image pose: {pose}")
            # Lets start the image IDs with 1 to stick to the COLMAP convention
            im_id = line_idx+1

            camera = pycolmap.Camera('PINHOLE', width, height, [fx, fy, cx, cy], im_id)

            # image = pycolmap.Image(os.path.join(recording_path, 'PV', image_name + '.png'), [], R.from_matrix(pose[:3,:3]).as_quat(), pose[:3,3], im_id)
            image = pycolmap.Image(name = os.path.join(recording_path, 'PV', image_name + '.png'), camera_id = im_id, id = im_id)
            # image.name = os.path.join(recording_path, 'PV', image_name + '.png')
            # image.camera_id = im_id
            image.qvec = R.from_matrix(pose[:3,:3]).inv().as_quat()
            image.tvec = -R.from_matrix(pose[:3,:3]).inv().as_matrix() @ pose[:3,3]


            reconstruction.add_camera(camera)
            reconstruction.add_image(image, check_not_registered=True)
            reconstruction.register_image(im_id)

            print(f"Num images: {reconstruction.num_images()}")

    
    reconstruction.write_text(model_path)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert images')
    parser.add_argument("--recording_path", required=True,
                        help="Path to recording folder")
    parser.add_argument("--output_model_path", required=True,
                        help="Where to store the created model")
    args = parser.parse_args()
    
    create_reconstruction(args.recording_path, args.output_model_path)
