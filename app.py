
import numpy as np
import pandas as pd
import streamlit as st 
import sklearn
import pickle
def plot_data(x,y,clr):
    #data.plot(x=index,y='Methylation_prot1',kind='scatter')
    #st.header("Data Visualization")
    st.write(f"Protein {i+1} Data Distribution")
    fig, ax = plt.subplots()
    plt.scatter(x, y, color=clr,label=y.name,ax=ax)
    plt.title(f'Age vs {y.name}')
    plt.xlabel('Age')
    plt.ylabel(y.name)
    st.pyplot(fig)
def main(): 
    model = pickle.load(open('linear_reg_model.pkl', 'rb'))
    st.title("Patient Age Predictor")
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
        plot_data(protein1['Age'],protein1['Methylation_prot1'],"lightblue")
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
            protein1.rename(columns={'Methylation (%)':'Methylation_prot1'},inplace=True)
            protein1.Age=protein1.Age.round(3)
            protein2.rename(columns={'Methylation (%)':'Methylation_prot2'},inplace=True)
            protein2.Age=protein2.Age.round(3)
            protein3.rename(columns={'Methylation (%)':'Methylation_prot3'},inplace=True)
            protein3.Age=protein3.Age.round(3)
            all_data=pd.concat([protein1, protein2,protein3])
            all_data.sort_values(by='Age',inplace=True)
            st.write(all_data.head())
            #plot_data(protein1['Age'],protein1['Methylation_prot1'],"lightblue")



    age = st.text_input("Age","0") 
    workclass = st.selectbox("Working Class", ["Federal-gov","Local-gov","Never-worked","Private","Self-emp-inc","Self-emp-not-inc","State-gov","Without-pay"]) 
    
    hours_per_week = st.text_input("Hours per week","0") 
    nativecountry = st.selectbox("Native Country",["Cambodia","Canada","China","Columbia","Cuba","Dominican Republic","Ecuador","El Salvadorr","England","France","Germany","Greece","Guatemala","Haiti","Netherlands","Honduras","HongKong","Hungary","India","Iran","Ireland","Italy","Jamaica","Japan","Laos","Mexico","Nicaragua","Outlying-US(Guam-USVI-etc)","Peru","Philippines","Poland","Portugal","Puerto-Rico","Scotland","South","Taiwan","Thailand","Trinadad&Tobago","United States","Vietnam","Yugoslavia"]) 
    
    if st.button("Predict"): 
        features = [[age,workclass,education,marital_status,occupation,relationship,race,gender,capital_gain,capital_loss,hours_per_week,nativecountry]]
        data = {'age': int(age), 'workclass': workclass, 'education': education, 'maritalstatus': marital_status, 'occupation': occupation, 'relationship': relationship, 'race': race, 'gender': gender, 'capitalgain': int(capital_gain), 'capitalloss': int(capital_loss), 'hoursperweek': int(hours_per_week), 'nativecountry': nativecountry}
        print(data)
        df=pd.DataFrame([list(data.values())], columns=['age','workclass','education','maritalstatus','occupation','relationship','race','gender','capitalgain','capitalloss','hoursperweek','nativecountry'])
                
      
if __name__=='__main__': 
    main()
