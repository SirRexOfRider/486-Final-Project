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

from sklearn.datasets import fetch_openml
from sklearn.tree import DecisionTreeClassifier as dtc
from PIL import Image
#from sklearn.tree import decisiontreeregression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier as rfc

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
    forest = rfc(n_estimators=100, random_state = 40)

    #Fit it do the training data
    forest.fit(X_train, Y_train)
    
    #Get accuracy
    y_pred_forest = forest.predict(X_test)
    acc_forest = accuracy_score(Y_test, y_pred_forest)
    print(f"The accuray of the forest is {acc_forest}")

    #======================= THIS IS WHERE YOU WOULD CHANGE THE DATA =============================
    
    #Change this file path to go through different numbers
    pictures_to_compare = "Nums"
    
    #=============================================================================================
    
    #This is being called from a different function
    X_new, new_Y_test, names = load_custom_images(pictures_to_compare)
    X_new = np.array(X_new)
    new_Y_test = np.array(new_Y_test)

    #Predict
    predictions = forest.predict(X_new)
    for name, pred in zip(names, predictions):
        print(f"Actual [{name[0]}] -> : {pred}")
        
    #Get accuracy
    y_pred_forest = forest.predict(X_new)
    acc_forest = accuracy_score(new_Y_test, y_pred_forest)
    print(f"The accuray for the custom numbers is: {acc_forest}")

def load_custom_images(folder_path):
    
    #Make parallel arrays for the data and labels
    data = []
    filenames = []
    labels = []

    #For each file
    for file in os.listdir(folder_path):
     
        
        
        img_path = os.path.join(folder_path, file)
        img = Image.open(img_path).convert('L')
        img = img.resize((28, 28))

        img_array = np.array(img)

        # get values between 0 and 1
        img_array = img_array / 255.0

        img_array = img_array.flatten()

        #Append the formatted data
        data.append(img_array)
        
        #Append the file
        filenames.append(file)
        
        #Get the label of it (the first index)
        labels.append(int(file[0]))

    #Return everything
    return data, labels, filenames

if __name__ == "__main__":
    main()