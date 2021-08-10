import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import metrics
import pickle

def split(data,ratio):
    np.random.seed(42) #freeze random value
    shuffled = np.random.permutation(len(data)) #randon shuffle num generate krbe
    test_set_size = int(len(data)*ratio)
    test_indicies = shuffled[:test_set_size] #create value according to testsize row and colum
    train_indicies = shuffled[test_set_size:]  
    return data.iloc[train_indicies], data.iloc[test_indicies]


if __name__ == "__main__":
        #read data
        data = pd.read_csv("data.csv")
        train , test = split(data,0.2)
        X_train = train[['fiver','age','headach','runnyNose','persistent Cough','sore throatr','diffBreath','loss of smell']].to_numpy()
        X_test = test[['fiver','age','headach','runnyNose','persistent Cough','sore throatr','diffBreath','loss of smell']].to_numpy()
       
        Y_train = train[['innfectionProb']].to_numpy().reshape(868,)
        Y_test = test[['innfectionProb']].to_numpy().reshape(217,)

        clf = LogisticRegression()
        dfmodel =clf.fit(X_train,Y_train) #best form e rakhte hobe

         #code for interferech

        input_user = [99.179315,77, 1, 0, 0, 0, 0, 1]
        inf_prob = clf.predict_proba([input_user])[0][1]
        pred = clf.predict([input_user])
        if pred == 1:
            print("paitent is affected")
        else:
           print("paitent is not affected")
        file = open('Corona_vir_project.pkl','wb')
        pickle.dump(dfmodel,file)

      