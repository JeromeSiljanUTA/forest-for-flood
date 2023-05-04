# forest-for-flood

1. Run either `git clone` or `git pull` so this repo is up to date on your system.
2. Run this command at the base of the repo (`forest-for-fire`): 
```
mkdir -p data/{01_raw,02_intermediate,04_models,06_reporting,}
```
2. Download the data from [FEMA's National Flood Insurance Policy Database](https://www.kaggle.com/datasets/lynma01/femas-national-flood-insurance-policy-database) and put it under `data/01_raw`
3. Go to the file `src/__init__.py` and take a look at the flags for `TRAIN_MODEL` and `LOAD_MODEL`. For the first run, `LOAD_MODEL` should be `False`
4. When you're ready to run the code, run with `python3 src/__init__.py`
5. I'd suggest putting your code under the if statements for the `TRAIN_MODEL` and `LOAD_MODEL` flags, but whatever works for you is best. 

**NOTE:** It would probably be a good idea to go under the `src/` directory and see what all of the methods do because that would help you understand how to move things around and write code.

Also, if you get more errors, you can change the line in the `__init__.py` that says `.config("spark.driver.memory", "10g")` and add more RAM for the model.

