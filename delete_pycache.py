import os
import shutil

def delete_pycache(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            if dir_name == "__pycache__":
                pycache_path = os.path.join(root, dir_name)
                print(f"Deleting: {pycache_path}")
                shutil.rmtree(pycache_path)
    print("All __pycache__ directories have been deleted.")

# Replace with the root folder path where you want to delete __pycache__ folders
root_folder = "/Users/dorijandonajmagasic/Documents/Uni modules/Individual Project/Prompt_optimisation_project"  # Update this path to your project directory

if __name__ == "__main__":
    delete_pycache(root_folder)
