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

# Function to install or update a package
function Install-Or-Update-Package {
    param (
        [string]$PackageName
    )
    
    try {
        # Check if the package is installed and get its version
        $installedPackages = pip list --format=json | ConvertFrom-Json
        $package = $installedPackages | Where-Object { $_.name -eq $PackageName }
        
        if ($null -eq $package) {
            Write-Output "Installing $PackageName..."
            pip install $PackageName
            if ($LASTEXITCODE -ne 0) {
                Write-Error "Error installing $PackageName."
                return $false
            }
            Write-Output "$PackageName installed successfully."
        } else {
            Write-Output "$PackageName is already installed (version $($package.version))."
            
            # Check if a newer version is available
            Write-Output "Checking for updates for $PackageName..."
            pip list --outdated --format=json | ConvertFrom-Json | ForEach-Object {
                if ($_.name -eq $PackageName) {
                    Write-Output "Updating $PackageName from version $($package.version) to $($_.latest_version)..."
                    pip install --upgrade $PackageName
                    if ($LASTEXITCODE -ne 0) {
                        Write-Error "Error updating $PackageName."
                        return $false
                    }
                    Write-Output "$PackageName updated successfully to version $($_.latest_version)."
                }
            }
        }
        return $true
    }
    catch {
        Write-Error ("Error processing package " + $PackageName + ": " + $_)
        return $false
    }
}

# Install or update pipreqs
if (Install-Or-Update-Package -PackageName "pipreqs") {
    # Generate the requirements.txt file
    Write-Output "Generating requirements.txt with pipreqs..."
    pipreqs . --force

    # Install dependencies from requirements.txt
    Write-Output "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt

    Write-Output "The script has completed successfully."
} else {
    Write-Error "Failed to set up pipreqs. The script cannot continue."
    exit 1
}
