# Name of the virtual environment directory
$VENV_DIR = "venv"

# Create the virtual environment if it does not exist
if (-not (Test-Path $VENV_DIR)) {
    Write-Output "Creating virtual environment..."
    python -m venv $VENV_DIR
} else {
    Write-Output "The virtual environment already exists."
}

# Activate the virtual environment
& "$VENV_DIR\Scripts\Activate.ps1"

# Install dependencies from requirements.txt
Write-Output "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

Write-Output "The script has completed successfully."
