import os
import zipfile

def zip_project():
    # Define output filenames
    seg_zip_name = 'segmentation_task.zip'
    ssl_zip_name = 'ssl_learning_cluster.zip'

    # Common directories and files to include in both
    common_dirs = ['Dataset', 'Todo_List']
    common_files = ['README.md', 'requirements.txt', 'weights_finder.py']

    # Lists to hold file paths for each zip
    seg_files = []
    ssl_files = []

    # Walk through the directory
    for root, dirs, files in os.walk('.'):
        # Exclude hidden directories and Results
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'Results' and d != '__pycache__']
        
        for file in files:
            if file.startswith('.') or file.endswith('.pyc') or file.endswith('.zip') or file == os.path.basename(__file__):
                continue

            file_path = os.path.join(root, file)
            # Remove ./ prefix for cleaner paths in zip
            if file_path.startswith('./'):
                arcname = file_path[2:]
            else:
                arcname = file_path

            # Check if file is in a common directory
            is_common_dir = any(arcname.startswith(d + os.sep) for d in common_dirs)
            
            if is_common_dir or arcname in common_files:
                seg_files.append(arcname)
                ssl_files.append(arcname)
                continue

            # Handle specific files in root
            if arcname == 'main.py':
                seg_files.append(arcname)
            elif arcname == 'main_proxy.py' or arcname == 'clustering_analysis.py':
                ssl_files.append(arcname)
            
            # Handle Networks folder
            elif arcname.startswith('Networks' + os.sep):
                # Check if it's in Architectures
                if 'Architectures' in arcname:
                    if 'Proxy' in arcname:
                        ssl_files.append(arcname)
                    else:
                        # Assume non-proxy architectures belong to segmentation
                        # Note: __init__.py would fall here, which is fine to include in segmentation
                        # But we might need __init__.py in SSL too if it exists.
                        # Let's check for __init__.py specifically
                        if file == '__init__.py':
                            seg_files.append(arcname)
                            ssl_files.append(arcname)
                        else:
                            seg_files.append(arcname)
                else:
                    # Files directly in Networks/
                    if file == 'model.py':
                        seg_files.append(arcname)
                    elif file == 'model_proxy.py':
                        ssl_files.append(arcname)
                    elif file == '__init__.py':
                        seg_files.append(arcname)
                        ssl_files.append(arcname)
                    # Any other file in Networks/ not explicitly handled?
                    # ssl_clustering.py was deleted.
                    
    # Function to create zip
    def create_archive(zip_name, file_list):
        print(f"Creating {zip_name}...")
        with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in file_list:
                if os.path.exists(file):
                    print(f"  Adding {file}")
                    zipf.write(file)
                else:
                    print(f"  Warning: {file} not found, skipping.")
        print(f"Finished {zip_name}\n")

    create_archive(seg_zip_name, seg_files)
    create_archive(ssl_zip_name, ssl_files)

if __name__ == "__main__":
    zip_project()
