@echo off

REM FOR %%b IN (True False) DO (
REM     FOR %%c IN  (True False) DO (
REM         python cuisine_classification/NN_classification.py --proba-threshold 0.95  --print-step 500 --balanced %%b --clustering %%c --nb-epochs 300  --file-type "full" --data-folder "F:\user\Google Drive\Catherning Folder\THU\Thesis\Work\Recipe datasets\cuisine_classification" 
REM     )
REM ) 

FOR %%b IN (True False) DO (
    FOR %%c IN  (True False) DO (
        python cuisine_classification/NN_classification.py --embedding-layer --embed-dim1 1024 --print-step 500  --proba-threshold 0.95  --balanced %%b --clustering %%c --nb-epochs 300  --file-type "full" --data-folder "F:\user\Google Drive\Catherning Folder\THU\Thesis\Work\Recipe datasets\cuisine_classification" 
    )
) 
