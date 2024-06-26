import streamlit as st
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline


st.title ("Student Performance Prediction")

st.write("### Student's Profile")
col1, col2 , col3 = st.columns(3)
gender = col1.selectbox("Enter you Gender", ("male","female"))
race_ethnicity = col2.selectbox("Enter you ethicity", ("group A","group B","group C","group D","group E"))
parental_level_of_education = col3.selectbox("Select Parent Education",("associate's degree","bachelor's degree","high school","master's degree","some college","some high school"))
lunch = col1.selectbox("Lunch Type", ("free/reduced","standard"))
test_preparation_course = col2.selectbox("Enter you Course",("none","completed"))
reading_score = col1.number_input("Enter your Reading score", min_value=0, max_value=100, value=50)
writing_score = col2.number_input("Enter your Writing score", min_value=0, max_value=100, value=50)

if st.button("Predict"):
## Construct df for prediction
    data = CustomData(gender = gender,
                          race_ethnicity = race_ethnicity,
                          parental_level_of_education = parental_level_of_education,
                          lunch = lunch,
                          test_preparation_course = test_preparation_course,
                          reading_score = reading_score,
                          writing_score = writing_score)
    pred_df = data.get_data_as_data_frame()
    pred_pipeline = PredictPipeline()
    predict_result = pred_pipeline.predict(pred_df)
    st.write(f"The math score (predict) is: {int(predict_result[0])}")