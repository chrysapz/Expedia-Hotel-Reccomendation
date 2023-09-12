# Expedia-Hotel-Reccomendation
This is our attempt to create accurate hotel recommendations for the Expedia website.
For details and motivation behind our implementation refer to the file Report.pdf.

## Solutions of actual Kaggle winners:

https://github.com/yangminglintw/Personalize-Expedia-Hotel-Searches---ICDM-2013

## Other potentially useful references

https://towardsdatascience.com/time-series-of-price-anomaly-detection-13586cd5ff46

## Running the code

```
# clone the repo, cd into the new dir
# first install mamba, a better version of conda (assuming you have anaconda or miniconda installed)
conda install mamba

# set up the environment
mamba env create -f environment.yml

# activate the new environment
conda activate dmt_expedia
```
 
Any new packages should be added to the environment.yml file then run the following command:
```
mamba env update -n dmt_expedia --file environment.yml  --prune
```
