FOR %%b IN (True False) DO (
    FOR %%c IN  (True False) DO (
        FOR %%u in (full cluster_centroid ) DO (
            python cuisine_classification/NN_classification.py  --embed-dim1 1024  --balanced %%b --clustering %%c --nb-epochs 300 --fuse False  --file-type %%u --data-folder "F:\user\Google Drive\Catherning Folder\THU\Thesis\Work\Recipe datasets" 
        )
    )
) 