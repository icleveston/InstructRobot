import subprocess

subprocess.run(["xvfb-run -a python3 TrainExtrinsic.py"], shell=True)