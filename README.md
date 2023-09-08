# Master Thesis

## Environement
Depending on your operating system and graphic card, you will be able to use option 1 or 2 to install your environment. In all following commands we use ```mamba``` to install packages as it can do it much faster than ```conda```. To install ```mamba``` you have to run the command ```conda install mamba```. However, if you wish to use ```conda``` instead, you can substitute ```mamba``` with ```conda``` in all following commands.

### Option 1
To create an environment from file open a miniconda prompt, navigate to where you cloned this repo and run the following command: ```mamba env create -f environment.yaml```.
If there the installation of some packages failed, you can try the option 2.

### Option 2
Firstly, create an empty conda environment using ```mamba create -n master python```.

Then, run this command ```mamba install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia``` to install pytorch that is compatible with the cuda version installed. Because we are using an older graphic card, it does not receive cuda support in most recent versions 12.*, but only till the version 11.7. Therefore, we had to explicity specify the ```pytorch-cuda``` version.

Afterward, all remaining packages can be installed. ```conda``` will sometimes pick the older version of a package to comply with already installed packages. Therefore, it is important that you firstly install pytorch and then the rest of the packages. To do this, run the following command: ```mamba install ipykernel pandas numpy scipy opencv evaluate tqdm matplotlib datasets pillow pip```. Then, as some packages can only be installed with pip, run ```pip install xformers accelerate```.

## Data
You can download the required WE3DS dataset [here](https://zenodo.org/record/7457983). After download, you will have to extract all files in the master folder.