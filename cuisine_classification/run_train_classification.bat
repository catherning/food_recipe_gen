set bool = True False
set under_met = full cluster_centroid 

FOR %%balanced IN (%bool%) do(
    FOR %%clustering IN  (%bool%) do (
        FOR %%undersampling in (%under_met%) do(
            python cuisine_classification/NN_classification.py  --embed-dim1 1024  --balanced %%balanced --clustering %%clustering --nb-epochs 300 --fuse False  --file-type %%undersampling --data-folder "F:\user\Google Drive\Catherning Folder\THU\Thesis\Work\Recipe datasets" 
        )
    )
) 