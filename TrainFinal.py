#!/usr/bin/env python
# coding: utf-8

# # Training File for Final Dataset

# This is a training file for the Final dataset (cluster number 65717793) located at "/global/cfs/cdirs/m3712/Mu2e/TrkAna/65717793". Running the following cells will call and run TrainBkg.ipynb. The AUC curve generated will be saved as "TrainBkgFinal.pdf" in the "training_plots" directory, and the .h5 file training model generated will be saved as "TrainBkgFinal.h5" in the "models" file

# In[1]:


suffix = "Final"
treename = "TAKK"
file_list = "/global/cfs/cdirs/m3712/Mu2e/TrkAna/65717793/files.txt"
print("Using files in " + file_list)

with open("TrainBkg_Outdated.py") as f:
    exec(f.read())

