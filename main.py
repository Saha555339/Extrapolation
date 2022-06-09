import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

class A:
    def __init__(self):
        private: self.x=10
        self.y=20
    def sum(self):
        self.x+self.y #не работает


def main():
    dataset = pd.read_csv('C:\lab\lab9\gdp_csv.csv')
    print(dataset.shape)
    print(dataset.head())
    print(dataset.describe())
    print('''Введите код:
    1. Arab World - ARB
    2. Caribbean small states - CSS
    3. Central Europe and the Baltics - CEB
    4. Early-demographic dividend - EAR
    5. East Asia & Pacifics - EAS
    6. Euro area - EMU
    7. Europe & Central Asia - ECS
    8. European Union - EUU
    9. Fragile and conflict affected situation - FCS''')
    cc=input()
    df_filter = dataset['Country Code'].isin([str(cc)])
    dp = dataset[df_filter]
    dp.plot(x='Year', y='Value', style='o')
    x = dp.iloc[:, 2:3].values
    y = dp.iloc[:, 3].values
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.0001)
    regressor = LinearRegression().fit(x, y)
    regressor.fit(X_train, y_train)
    #print(regressor.intercept_)
    #print(regressor.coef_)
    y_pred = regressor.predict(X_train)
    #print(X_train)
    df = pd.DataFrame({'Actual': y_train, 'Predicted': y_pred})
    #print(df)
    plt.plot(X_train, y_pred, '.')
    plt.show()

if __name__=="__main__":
    main()
    a=A()
    print(a.sum())
