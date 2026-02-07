import platform
import subprocess
import sys
import os
from scripts import initial_runtime

def check_python_version():
    print("Python version information:")
    print(f"Python Version: {platform.python_version()}")

def check_pytorch_and_cuda():
    try:
        import torch
        print("PyTorch is installed.")
        version = torch.__version__
        print(f"PyTorch Version: {version}")

        # Check if PyTorch version is >= 1.11
        if tuple(map(int, version.split('.')[:2])) >= (1, 11):
            print("PyTorch version is greater than or equal to 1.11.")
        else:
            print("WARNING: PyTorch version is less than 1.11. Please update PyTorch.")
        
        if torch.cuda.is_available():
            print("CUDA is available and PyTorch can utilize it.")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"Number of CUDA devices: {torch.cuda.device_count()}")
            print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        else:
            print("ERROR: CUDA is not available or PyTorch is not compiled with CUDA support.")
    except ImportError:
        print("ERROR: PyTorch is not installed.")
    except Exception as e:
        print(f"An error occurred while checking PyTorch or CUDA: {e}")

def check_requirements_installed(requirements_file='requirements.txt'):
    print("\nChecking if all packages in requirements.txt are installed...")
    try:
        with open(requirements_file, 'r') as file:
            requirements = file.readlines()
            for req in requirements:
                req = req.strip()
                if req:
                    try:
                        subprocess.run([sys.executable, '-m', 'pip', 'show', req.split('==')[0]], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        print(f"{req} is installed.")
                    except subprocess.CalledProcessError:
                        print(f"ERROR: {req} is NOT installed. Please install it using `pip install {req}`.")
    except FileNotFoundError:
        print(f"ERROR:{requirements_file} not found.")
    except Exception as e:
        print(f"An error occurred while checking requirements: {e}")

def check_files_exist(file_paths):
    print("\nChecking if all supplementary files are at the correct location...")
    all_files_exist = True
    for file_path in file_paths:
        if os.path.exists(file_path):
            print(f"File found: {file_path}")
        else:
            print(f"File NOT found: {file_path}")
            all_files_exist = False
    return all_files_exist

def main():
    print("Running Basic Test...\n")
    check_python_version()
    check_pytorch_and_cuda()
    check_requirements_installed('requirements.txt')
    print("\nEnvironment check complete.")
    print("Running functionality check...\n")
    initial_runtime.main(True)
    print("\nBasic Test completed. Please check if there is any error message.\n")


if __name__ == "__main__":
    main()
