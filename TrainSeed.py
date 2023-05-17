#!/usr/bin/env python
# coding: utf-8

# # Training File for Seed Dataset

# This is a training file for the Seed dataset (cluster number 43291981) located at "/global/cfs/cdirs/m3712/Mu2e/TrkAna/43291981". Running the following cells will call and run TrainBkg.ipynb. The AUC curve generated will be saved as "TrainBkgSeed.pdf" in the "training_plots" directory, and the .h5 file training model generated will be saved as "TrainBkgSeed.h5" in the "models" file

# In[1]:


suffix = "Seed"
treename = "TAKK"
#file_list = "/global/cfs/cdirs/m3712/Mu2e/TrkAna/43291981/files.txt"
file_list = "/Users/brownd/data/43291981/files.txt"
print("Using files in " + file_list)

get_ipython().run_line_magic('run', './TrainBkg.ipynb')

