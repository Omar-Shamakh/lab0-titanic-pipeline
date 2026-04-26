import subprocess

subprocess.run(["uv", "run", "python", "main.py", "model=random_forest"])
subprocess.run(["uv", "run", "python", "main.py", "model=gradient_boosting"])