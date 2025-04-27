import os
import tarfile

# Define the path to the train directory
train_dir = 'imagenet/train'

# Check if the train directory exists
if not os.path.exists(train_dir):
    print(f"Error: The {train_dir} directory does not exist.")
else:
    # Get all tar files in the train directory
    tar_files = [f for f in os.listdir(train_dir) if f.endswith('.tar')]

    # Iterate through each tar file
    for tar_file in tar_files:
        # Construct the full path of the tar file
        tar_path = os.path.join(train_dir, tar_file)
        # Extract the folder name (remove the .tar suffix)
        folder_name = os.path.splitext(tar_file)[0]
        # Construct the full path of the target folder
        target_folder = os.path.join(train_dir, folder_name)

        # Check if the target folder already exists
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        try:
            # Open the tar file
            with tarfile.open(tar_path, 'r') as tar:
                # Extract the tar file to the target folder
                tar.extractall(path=target_folder)
            print(f"Successfully extracted {tar_file} to {target_folder}")
            # Delete the original tar file after successful extraction
            os.remove(tar_path)
            print(f"Successfully deleted {tar_file}")
        except Exception as e:
            print(f"Error occurred while extracting {tar_file}: {e}")
    