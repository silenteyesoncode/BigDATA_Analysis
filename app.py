from flask import Flask, make_response, request, render_template
import io
import os
import csv
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

app = Flask(__name__)
model = pickle.load(open(r'model.pkl', 'rb'))

# Base Route
@app.route('/')
def hello():
    return render_template('bigmart.html')


# Prediction for dataset
@app.route('/predict_for_set', methods=['POST'])
def predict_for_set():
    file = request.files.get('file')
    df = pd.read_csv(file)
    # check for categorical attributes
    cat_col = []
    for x in df.dtypes.index:
        if df.dtypes[x] == 'object':
            cat_col.append(x)
    cat_col.remove('Item_Identifier')
    cat_col.remove('Outlet_Identifier')
    item_weight_mean = df.pivot_table( values="Item_Weight", index='Item_Identifier')
    miss_bool = df['Item_Weight'].isnull()
    for i, item in enumerate(df['Item_Identifier']):
        if miss_bool[i]:
            if item in item_weight_mean:
                df['Item_Weight'][i] = item_weight_mean.loc[item]['Item_Weight']
            else:
                df['Item_Weight'][i] = np.mean(df['Item_Weight'])
    outlet_size_mode = df.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))
    miss_bool = df['Outlet_Size'].isnull()
    df.loc[miss_bool, 'Outlet_Size'] = df.loc[miss_bool,'Outlet_Type'].apply(lambda x: outlet_size_mode[x])
    # replace zeros with mean
    df.loc[:, 'Item_Visibility'].replace(
        [0], [df['Item_Visibility'].mean()], inplace=True)
    # combine item fat content
    df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'LF': 'Low Fat', 'reg': 'Regular', 'low fat': 'Low Fat'})
    df['Item_Fat_Content'].value_counts()
    # Creation of New Attributes
    df['New_Item_Type'] = df['Item_Identifier'].apply(lambda x: x[:2])
    df['New_Item_Type'] = df['New_Item_Type'].map({'FD': 'Food', 'NC': 'Non-Consumable', 'DR': 'Drinks'})
    df.loc[df['New_Item_Type'] == 'Non-Consumable','Item_Fat_Content'] = 'Non-Edible'
    # create small values for establishment year
    df['Outlet_Years'] = 2013 - df['Outlet_Establishment_Year']
    le = LabelEncoder()
    df['Outlet'] = le.fit_transform(df['Outlet_Identifier'])
    cat_col = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size','Outlet_Location_Type', 'Outlet_Type', 'New_Item_Type']
    for col in cat_col:
        df[col] = le.fit_transform(df[col])
    # Input Split
    X = df.drop(columns=['Outlet_Establishment_Year', 'Item_Identifier', 'Outlet_Identifier'])
    print(X)
    print(X.dtypes)
    # Prediction
    output = model.predict(X).tolist()
    sales = sum(output)
    df['Item_Outlet_Sales'] = output
    df['Revenue'] = df['Item_Outlet_Sales']*df['Item_MRP']
    revenue = sum(df['Revenue'])
    print(revenue)
    revenue /= 1000000
    print(revenue)
    return render_template('bigmart.html', pred1="The {} is the overall number of items that are expected to be sold from bigmart stores.".format(sales), pred2="The total revenue that should be generate is ${} million.".format(revenue))

# Prediction for single product
@app.route('/predict_for_one', methods=['POST'])
def predict_for_one():
    d = None
    d = request.form.to_dict()
    df = pd.DataFrame([d.values()], columns=d.keys())
    df = df.infer_objects()
    df[['Item_Weight','Item_Visibility','Item_MRP','Outlet_Establishment_Year']] = df[['Item_Weight','Item_Visibility','Item_MRP','Outlet_Establishment_Year']].apply(pd.to_numeric)
    # Process dataframe as required
    # check for categorical attributes
    cat_col = []
    for x in df.dtypes.index:
        if df.dtypes[x] == 'object':
            cat_col.append(x)
    cat_col.remove('Item_Identifier')
    cat_col.remove('Outlet_Identifier')
    item_weight_mean = df.pivot_table( values="Item_Weight", index='Item_Identifier')
    miss_bool = df['Item_Weight'].isnull()
    for i, item in enumerate(df['Item_Identifier']):
        if miss_bool[i]:
            if item in item_weight_mean:
                df['Item_Weight'][i] = item_weight_mean.loc[item]['Item_Weight']
            else:
                df['Item_Weight'][i] = np.mean(df['Item_Weight'])
    outlet_size_mode = df.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))
    miss_bool = df['Outlet_Size'].isnull()
    df.loc[miss_bool, 'Outlet_Size'] = df.loc[miss_bool,'Outlet_Type'].apply(lambda x: outlet_size_mode[x])
    # replace zeros with mean
    df.loc[:, 'Item_Visibility'].replace(
        [0], [df['Item_Visibility'].mean()], inplace=True)
    # combine item fat content
    df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'LF': 'Low Fat', 'reg': 'Regular', 'low fat': 'Low Fat'})
    df['Item_Fat_Content'].value_counts()
    # Creation of New Attributes
    df['New_Item_Type'] = df['Item_Identifier'].apply(lambda x: x[:2])
    df['New_Item_Type'] = df['New_Item_Type'].map({'FD': 'Food', 'NC': 'Non-Consumable', 'DR': 'Drinks'})
    df.loc[df['New_Item_Type'] == 'Non-Consumable','Item_Fat_Content'] = 'Non-Edible'
    # create small values for establishment year
    df['Outlet_Years'] = 2013 - df['Outlet_Establishment_Year']
    le = LabelEncoder()
    df['Outlet'] = le.fit_transform(df['Outlet_Identifier'])
    cat_col = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size','Outlet_Location_Type', 'Outlet_Type', 'New_Item_Type']
    for col in cat_col:
        df[col] = le.fit_transform(df[col])
    # Input Split
    X = df.drop(columns=['Outlet_Establishment_Year', 'Item_Identifier', 'Outlet_Identifier'])
    print(X)
    print(X.dtypes)
    # Prediction
    output = model.predict(X).tolist()
    sales = sum(output)
    df['Item_Outlet_Sales'] = output
    df['Revenue'] = df['Item_Outlet_Sales']*df['Item_MRP']
    revenue = sum(df['Revenue'])
    return render_template('bigmart.html', pred1='The {} units of this product are expected to be sold.'.format(sales), pred2='${} is the revenue that should be generated by selling this much units of selected product.'.format(revenue))

if __name__ == "__main__":
    app.run(debug = True)
