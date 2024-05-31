import subprocess

# List of packages with specified versions
packages = [
    "pandas==2.0.3",
    "numpy==1.25.2",
    "scikit-learn==1.2.2",
    "tensorflow==2.15.0"
]

# Install each package
for package in packages:
    subprocess.run(["pip", "install", package])

print("All specified packages have been installed.")