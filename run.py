import subprocess

subprocess.run(["xvfb-run python3 TrainTask.py --resume_intrinsic CubeSimpleInt_2023_12_17_15_08_12"], shell=True)