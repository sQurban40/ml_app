
import numpy as np
import pandas as pd
import streamlit as st 
import pickle
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

def plot_data(x,y,clr):
    fig, ax = plt.subplots()
    plt.scatter(x, y, color=clr,label=y.name)
    plt.title(f'Age vs {y.name}')
    plt.xlabel('Age')
    plt.ylabel(y.name)
    st.pyplot(fig)
def transform_data(data):
    nan = np.nan
    imputer = KNNImputer(n_neighbors=3, weights="uniform")
    transformed_data=imputer.fit_transform(data)
    transformed_data=pd.DataFrame(transformed_data)
    return transformed_data
def model_implementation(data):
    target=data.iloc[:,0:1].values
    features=data.iloc[:,1:]
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.30, random_state=42)
    lreg_model = linear_model.LinearRegression()
    lreg_model.fit(X_train,y_train)
    y_pred_train=lreg_model.predict(X_train)
    y_pred_test=lreg_model.predict(X_test)
    return lreg_model
def main(): 
    model = pickle.load(open('linear_reg_model.pkl', 'rb'))
    #st.title("Patient Age Predictor")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Patient Age Predictor App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    
    st.header("Upload Protein Data")
    Protien1_file = st.file_uploader("Upload Protien1 excel file", accept_multiple_files=False)
    if Protien1_file:
        protein1 = pd.read_excel(Protien1_file)
        st.write(protein1.head())
    Protien2_file = st.file_uploader("Upload Protien2 excel file", type="xlsx", accept_multiple_files=False)
    if Protien2_file:
        protein2 = pd.read_excel(Protien2_file)
        st.write(protein2.head())
    Protien3_file = st.file_uploader("Upload Protien3 excel file", type="xlsx", accept_multiple_files=False)
    if Protien3_file:
        protein3 = pd.read_excel(Protien3_file)
        st.write(protein3.head())
    if st.button("Analyze Data"): 
        if Protien1_file and Protien2_file and Protien3_file:
            st.header("Data Visualization")
            protein1.rename(columns={'Methylation (%)':'Methylation_prot1'},inplace=True)
            protein1.Age=protein1.Age.round(3)
            plot_data(protein1['Age'],protein1['Methylation_prot1'],"lightblue")
            
            protein2.rename(columns={'Methylation (%)':'Methylation_prot2'},inplace=True)
            protein2.Age=protein2.Age.round(3)
            plot_data(protein2['Age'],protein2['Methylation_prot2'],"c")
            
            protein3.rename(columns={'Methylation (%)':'Methylation_prot3'},inplace=True)
            protein3.Age=protein3.Age.round(3)
            plot_data(protein3['Age'],protein3['Methylation_prot3'],"lightgreen")
            
            all_data=pd.concat([protein1, protein2,protein3])
            all_data.sort_values(by='Age',inplace=True)
            st.markdown("Combining 3 protein Data", unsafe_allow_html = True)
            transformed_data=transform_data(all_data)
            st.write(transformed_data.head())
            lreg_model=model_implementation(transformed_data)
            
        else:
            st.warning("Please upload a all three Protein data files.")
    st.header("Predictions on test data")
    protein1 = st.text_input("% methylation of protein 1","0") 
    protein2 = st.text_input("% methylation of protein 2","0") 
    protein3 = st.text_input("% methylation of protein 3","0") 
    if st.button("Predict Age"): 
        st.write(protein1,protein2,protein3)
        test_sample=[int(protein1),int(protein1),int(protein1)]
        #test_sample_actual_ages=[[52],[44.7],[61.9],[32.3]]
        if(len(test_sample)==1):
            test_sample=test_sample.reshape(-1, 3)
        predicted_ages=lreg_model.predict(test_sample)
        st.write("Predicted Age for Given Sample is\n",np.round(predicted_ages,2))
    #age = st.text_input("Age","0") 
    #workclass = st.selectbox("Working Class", ["Federal-gov","Local-gov","Never-worked","Private","Self-emp-inc","Self-emp-not-inc","State-gov","Without-pay"]) 
    #hours_per_week = st.text_input("Hours per week","0") 
    #nativecountry = st.selectbox("Native Country",["Cambodia","Canada","China","Columbia","Cuba","Dominican Republic","Ecuador","El Salvadorr","England","France","Germany","Greece","Guatemala","Haiti","Netherlands","Honduras","HongKong","Hungary","India","Iran","Ireland","Italy","Jamaica","Japan","Laos","Mexico","Nicaragua","Outlying-US(Guam-USVI-etc)","Peru","Philippines","Poland","Portugal","Puerto-Rico","Scotland","South","Taiwan","Thailand","Trinadad&Tobago","United States","Vietnam","Yugoslavia"]) 
    
    #if st.button("Predict"): 
    #   features = [[age,workclass,education,marital_status,occupation,relationship,race,gender,capital_gain,capital_loss,hours_per_week,nativecountry]]
    #     data = {'age': int(age), 'workclass': workclass, 'education': education, 'maritalstatus': marital_status, 'occupation': occupation, 'relationship': relationship, 'race': race, 'gender': gender, 'capitalgain': int(capital_gain), 'capitalloss': int(capital_loss), 'hoursperweek': int(hours_per_week), 'nativecountry': nativecountry}
    #    print(data)
    #    df=pd.DataFrame([list(data.values())], columns=['age','workclass','education','maritalstatus','occupation','relationship','race','gender','capitalgain','capitalloss','hoursperweek','nativecountry'])
                
      
if __name__=='__main__': 
    main()
