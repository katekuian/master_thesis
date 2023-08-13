# Master Thesis

To execute the jupyter notebooks in this repositority, you will first need to create an environment and install all the required packages. To do this, open miniconda prompt, navigate where you cloned this repo and run the following command:  
```conda env create -f environment.yaml``` or ```mamba env create -f environment.yaml```.

After creating the environment, you have to run following command ```conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia``` to install pytorch with cuda support. Unfortunately it does not work if you simply include this packages in the environment.yaml

You can download the required WE3DS dataset [here](https://zenodo.org/record/7457983). After download, you will have to extract all files in the master folder.