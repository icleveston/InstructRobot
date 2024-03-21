import subprocess

subprocess.run(["xvfb-run -a python3 TrainIntrinsic.py"], shell=True)