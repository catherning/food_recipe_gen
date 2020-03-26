@echo off

FOR %%b IN (True False) DO (
    FOR %%c IN  (True False) DO (
        python cuisine_classification/NN_classification.py  --balanced %%b --clustering %%c --nb-epochs 150 --fuse False  --file-type "cluster_centroid10000" --data-folder "F:\user\Google Drive\Catherning Folder\THU\Thesis\Work\Recipe datasets" 
    )
) 
