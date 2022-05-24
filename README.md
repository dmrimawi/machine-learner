# machine-learner
This repository contains the code for the machine learner and the required packages

## Description
This repository has three different directories, as follows:

__Data file__: This directory has a CSV file for the data used for the training process. The data basically is a list of different ice cream recipe rates, each recipe consists of a list of ingredients marked by 1 if that specific ingredient is used in the recipe, and zero otherwise. 

__Output model__: This is the destination directory in which, the learner dumps the trained model to be used to classify future recipes.

__Python scrip learner__: Inside this directory is the machine learning algorithm, that will use both previous directories for the training process. 

The python script uses scikit-learn framework to create a classifying model using the kth nearest neighbor algorithm, to be able to classify new recipes.

To be able to run the learning script, you need to install the required pip packages, and then run the commands line script.

```
>> pip install -r requirements.txt

>> python ice_creame_learner.py
```

For more information about the project please refer to [ice cream web app](https://github.com/dmrimawi/ice_cream_web_app) repository.
