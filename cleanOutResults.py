import shutil
import os

def clean_directory(path):
    if os.path.exists(path):
        print(f"Removing directory: {path}")
        shutil.rmtree(path)
    else:
        print(f"Directory not found (already clean): {path}")

def main():
    # Base directories to clean
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Clean the entire results/ directory since it holds local outputs
    results_dir = os.path.join(base_dir, "results")
    clean_directory(results_dir)
    
    # Also clean the generated datasets to ensure a completely fresh start
    gandk_datasets = os.path.join(base_dir, "datasets", "gandk")
    normal_datasets = os.path.join(base_dir, "datasets", "normal")
    
    clean_directory(gandk_datasets)
    clean_directory(normal_datasets)
    
    print("Cleanup complete. Workspace is ready for fresh experiments.")

if __name__ == "__main__":
    main()
