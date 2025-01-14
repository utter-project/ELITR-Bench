import os
import shutil

def copy_and_rename_transcripts(source_dir, target_dir, required_splits):
    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Walk through the directory tree
    for root, dirs, files in os.walk(source_dir):
        # Skip directories not in required_splits
        if not any(split in root for split in required_splits):
            continue

        for file in files:
            # Check if the file name starts with 'transcript_MAN'
            if file.startswith('transcript_MAN'):
                # Extract the name of the parent folder
                parent_folder = os.path.basename(root)

                # Construct new file name
                new_file_name = f"{parent_folder}.txt"

                # Construct full paths
                source_path = os.path.join(root, file)
                target_path = os.path.join(target_dir, new_file_name)

                # Copy and rename the file
                shutil.copyfile(source_path, target_path)
                print(f"Copied: {source_path} -> {target_path}")

# Define source and target directories
elitr_directory = './ELITR-minuting-corpus/elitr-minuting-corpus-en'
data_directory = './data'
required_splits = ['dev', 'test2']

# Execute the function
copy_and_rename_transcripts(elitr_directory, data_directory, required_splits)