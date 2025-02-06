ENV_NAME="decictor"

# exit if conda is not installed
if ! command -v conda &> /dev/null
then
    echo "Conda is not installed. Exiting..."
    exit 1
fi

BASE_DIR=$(conda info --base)
DECICTOR_PIP="$BASE_DIR/envs/$ENV_NAME/bin/pip"

# create conda env decictor if not exist
if ! conda info --envs | grep -q "decictor"
then
    echo "Creating conda enviroment $ENV_NAME"
    conda create -n decictor python=3.7.16 -y
else
    echo "Conda envviroment $ENV_NAME exists"
fi

# install decictor dependencies
echo "Installing dependencies"
$DECICTOR_PIP install -r requirements.txt

# install pytorch
echo "Installing pytorch"
$DECICTOR_PIP install torch==1.13.1+cu117 \
    torchvision==0.14.1+cu117 \
    torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu117

echo -e "\033[0;32mDecictor is now ready to use with conda environment $ENV_NAME\033[0m"