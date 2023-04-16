# forest-for-flood

1. Download the data from [FEMA's National Flood Insurance Policy Database](https://www.kaggle.com/datasets/lynma01/femas-national-flood-insurance-policy-database) and put it under `data/01_raw`
2. Create a directory `date/02_intermediate`
3. Run the code in the notebook under `notebooks/20230416-jds-intermediate-data.ipynb`. This will create an "intermediate" dataset in parquet form for easy loading. 

**NOTE:** *All the paths in the notebooks are relative, it's recommended to run them in the `notebooks` directory.*
