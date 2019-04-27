# fundamentalsofml
Homework assignments and projects from Fundamental of Machine Learning 

## Project00

Please refer to the information below for running the code. 

**Please note, one will not be able to run this code as the one of the necessary input data was too large to upload to Git.**

### Download the file or clone the repository to a folder on your desktop. 
### If using Anaconda...
  * Open Spyder (Python versions 3.6+)
  * Open the run.py file 
  * Run the file - this will call the functions for training and testing 
  * The training function will return...
    * the locations of the pixels of the ground truth for red vehicles in the training data
    * the KNN model, which is passed to the test function
    * the predictions of the validation set 
  * Inside the training function...
    * the overall accuracy will be printed 
    * the confusion matrix will be printed for the validation set
    * the accuracy for only the ground truth (red vehicles) will be printed)
  * The test function will return...
    * the locations of the pixels chosen as red vehicles
    * the predictions of the test set 
    * the labels of the predictions (whether or not it's a red vehicle)
### If running code using a script, use run.py to execute the program. 
### Files/Parameters used in run.py
  * If you would like to change the images for the training set, change the filename for the data_train variable. 
  * If you would like to change the images for the test set, change the filename for the data_test variable.
  * If you have a different ground truth for the training set, change the filename for the ground_truth variable.
  * If you would to change the k paramter for the KNN classifier, change the value of k variable.
  * Please do NOT change the variable names.
  
  
## Project01 

**This project was completed by Kiana Alikhademi, Diandra Prioleau, and Armisha Roberts**

**Dependencies**
Install Opencv software in the environment. This is necessary to convert the binary images into contour images. The current implementation is tested under Opencv 3.4.2 which could be downloaded through the following link:
https://opencv.org/releases.html



There are six main files associated with this project: classification.py, feature_extraction.py, findContours.py, 
normalization.py, train.py, and test.py. The only file that requires running is test.py, all other desired functionality from 
the previous files mentioned will be imported and used appropriately. While running this code it is essential to include a 
folder named "Traning_Images" without the quotation marks to store all of the images for testing after normalization. This 
folder is used to prevent clutter within the directory, but this can also be changed within the normalization.py file. However,
if the directory is changed it will also need to be changed in normalization.py.

This group is participating in the extra credit opportunity; therefore, to run the program the command line will require 4 
arguments instead of the initial 3 described through class communication. To run the model for the extra credit you must type,
**python test.py ClassData_2.npy out extra**. The ClassData_2.npy argument is the testing data file to be passed, out is the desired file name for the prediction lables, and extra will denote the use of the extra credit model to use. To run the model to classify a's and b's 
please type **python test.py extra_data_1.npy out ab**. All of the argument meanings are the same except for the last argument ab, which denotes the use of the model for a and b testing.


**Note:** The A-Z Handwritten Alphabets dataset, which consisted of all uppercase letters A-Z, was used in the training process for classifying unknown characters/images. This dataset was obtained through Kaggle (link: https://www.kaggle.com/sachinpatel21/csv-to-images/data). The code provided was used to convert the csv file to png images. In addition, the researchers also handwrote the lowercase letters e-g and l-z to use in the training process for classifying unknown characters/images. These datasets can be accessed in the repository - extra_data.npy & extra_labels.npy for the Kaggle dataset and extra_data_1.npy & extra_labels_1.npy for the lowercase letters e-g and l-z.

