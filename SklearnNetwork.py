from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


class SkLearnNetwork():

    def __init__(self, predict_col, exclude_col=None, title_row=True):
        self.RawData = 'New_Data.csv'
        self.full_data, self.complete_data, self.incomplete_data = self.split_data(self.RawData, predict_col, title_row=True)
        self.com_inputs, self.com_outputs, self.incom_inputs, self.incom_outputs = self.read_data(predict_col, exclude_col)
        self.pred_list = self.best_n_neighbours()
        self.create_excel_sheet(predict_col)


    def split_data(self, filename, predict_col, title_row=True):
        if title_row == False:
            df = pd.read_csv(self.RawData, header=None)
        else:
            df = pd.read_csv(self.RawData)

        complete_data = df[df.iloc[:,predict_col-1].notna()]
        incomplete_data = df[df.iloc[:,predict_col-1].isna()]
        print("INCOMPLETE")
        print(incomplete_data)
        print('COMPLETE')
        print(complete_data)
        return df, complete_data, incomplete_data



    def read_data(self, predict_col, exclude_col):
        df = self.complete_data
        com_inputs = df.iloc[:, :]
        com_outputs = df.iloc[:, predict_col - 1]
        remove_list = []
        for x in exclude_col:
            remove_list.append(x-1)
        remove_list.append(predict_col-1)
        com_inputs.drop(com_inputs.columns[remove_list], inplace=True, axis = 1)
        print(com_inputs)
        print(com_outputs)

        df = self.incomplete_data
        incom_inputs = df.iloc[:, :]
        incom_outputs = df.iloc[:, predict_col - 1]
        remove_list = []
        for x in exclude_col:
            remove_list.append(x - 1)
        remove_list.append(predict_col - 1)
        incom_inputs.drop(incom_inputs.columns[remove_list], inplace=True, axis=1)
        print(incom_inputs)
        print(incom_outputs)

        return com_inputs, com_outputs, incom_inputs, incom_outputs

    def best_n_neighbours(self):

        X_train, X_test, y_train, y_test = train_test_split(self.com_inputs, self.com_outputs, test_size=0.2, random_state=5, stratify=self.com_outputs)
        best_n_neighbours = 0
        highest_score = 0
        for x in range(1, 50):
            first = KNeighborsClassifier(n_neighbors=x)
            first.fit(X_train, y_train)
            print(first.score(X_test, y_test), x)
            if (first.score(X_test, y_test)) > highest_score:
                highest_score = first.score(X_test, y_test)
                best_n_neighbours = x
        print('THE BEST SCORE IS ' + str(highest_score) + ' WITH ' + str(best_n_neighbours) + ' NEIGHBOURS')

        second = KNeighborsClassifier(n_neighbors=best_n_neighbours)
        second.fit(self.com_inputs, self.com_outputs)
        predicted_list = []
        for value in self.incom_inputs.values.tolist():
            print(value)
            predicted = second.predict([value])
            print(predicted)
            predicted_list.append(predicted)
        return predicted_list

    def create_excel_sheet(self, predict_col):
        print(self.full_data.iloc[:,predict_col-1])
        i=0
        for x in self.full_data.iloc[:,predict_col-1]:
            print(x)
            print(type(x))
            #if x.isempty() == True:
             #   x.replace(self.pred_list[i])
              #  i =+ 1
               # print('ITS WORKING')

        print(self.full_data)




if __name__ == "__main__":

    A = int(input('Predict Column (USE BASE 1 INDEX): '))
    print('Any Columns to Exclude? If so, type number then enter.(When done or if none, type "done")')
    try:
        exclude_list = []

        while True:
            exclude_list.append(int(input()))
    except:
        pass
    B=exclude_list
    print('Excluding'+str(B))
    C = str(input('There a title row. (True or False): '))


    print("Data input:")
    print('Predicting Column '+str(A))
    print('Excluding Column(s) '+str(B))
    if C == "True" or "true":
        print('There is a title row')
        C=True
    else:
        print('There is not a title row')
        C=False

    Network = SkLearnNetwork(A,B, title_row=C)




