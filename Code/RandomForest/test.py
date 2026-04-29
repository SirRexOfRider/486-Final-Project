#Braxton Rider
#4/26/26
#Random forest learning trying to classify images from 0-9
#I had to look up a bunch of stuff regarding how to actually load the data

#Had a lot of trouble actually getting the mnist to link
#So I had to download it locally ;/
from mnist import MNIST
import numpy as np

import os
#print(os.listdir('C:/Users/rider/OneDrive/Documents/GitHub/486-Final-Project/Code/RandomForest/Mnist'))

from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import f1_score

#================= CODE =====================
def main():
    
    #Having a bunch of issues actually loading this
    #Which is why the data is now stored locally and I'm having to call it like this
    mndata = MNIST('C:/Users/rider/OneDrive/Documents/GitHub/486-Final-Project/Code/RandomForest/Mnist')
    X_train, Y_train = mndata.load_training()
    X_test, Y_test = mndata.load_testing()
    #print("Files found:", os.listdir('Nums'))

    #Make them numpy arrays
    #You have to divide the X values by 255 so that it can be zero or one
    X_train = np.array(X_train) / 255.0
    Y_train = np.array(Y_train)

    X_test = np.array(X_test) / 255.0
    Y_test = np.array(Y_test)

    #Make forest
    #Tried tweaking values like max_depth, min_leafs, etc but it didn't help it
    #n_jobs = -1 made it run a bit faster at least
    forest = rfc(n_estimators = 200, random_state = 2, n_jobs = -1)

    #Tried cross validating, didn't really change anything for accuracy
    #Y_train_after_folds = cross_val_predict(forest, X_train, Y_train, cv=5, n_jobs=-1)
    #f1_cv = f1_score(Y_train, Y_train_after_folds, average='macro')
    
    #Fit it do the training data
    forest.fit(X_train, Y_train)
    
    #Get accuracy
    y_pred_forest = forest.predict(X_test)
    acc_forest_training = accuracy_score(Y_test, y_pred_forest)
    

    #======================= THIS IS WHERE YOU WOULD CHANGE THE DATA =============================
    
    #Change this file path to go through different numbers
    pictures_to_compare = "Nums"
    
    #=============================================================================================
    
    #This function reads the images from the folder in github and makes a new x and y set
    #To see how well the model does
    #Also returns names to print out to user to see what got missed
    X_new, Y_new, names = load_custom_images(pictures_to_compare)
    X_new = np.array(X_new)
    Y_new = np.array(Y_new)

    #Predict
    predictions = forest.predict(X_new)
    for name, pred in zip(names, predictions):
        print(f"Actual [{name[0]}] -> : {pred}")
        
    #Get accuracy
    y_pred_forest = forest.predict(X_new)
    acc_forest_custom = accuracy_score(Y_new, y_pred_forest)
    print(f"The accuracy after training forest: {acc_forest_training * 100:,.2f}%")
    print(f"The accuracy for predicting numbers: {acc_forest_custom * 100:,.2f}%")
    #print(f"Cross-validated f1 (from predictions): {f1_cv:,.2f}")

#Function that handles loading the custom images
#Also looked this up, thanks OS!
def load_custom_images(file_name):
    
    #Make parallel arrays for the data and lables    
    x_new = []
    y_new = []
    names = []
    
    for file in os.listdir(file_name):

        img_path = os.path.join(file_name, file)
        img = Image.open(img_path).convert('L')

        img = img.resize((28,28))
        img_array = np.array(img)

        # Normalize
        img_array = img_array / 255.0

        #Flatten
        img_array = img_array.flatten()

        x_new.append(img_array)
        y_new.append(int(file[0]))
        names.append(file)

    return x_new, y_new, names

if __name__ == "__main__":
    main()