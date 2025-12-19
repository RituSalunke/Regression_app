import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

FEATURE_ORDER = ["age", "bmi", "children", "sex", "smoker", "region"]


st.set_page_config(page_title="Insurance Test Data", layout="centered")
st.title("📝 Insurance Test Data Creator")

@st.cache_data
def convert_for_download(df):
    return df.to_csv(index=False).encode("utf-8")

@st.cache_data
def get_model():
    sex_encoder = joblib.load("lcoder_sex.joblib")
    smoker_encoder  = joblib.load("lcoder_smoker.joblib")
    region_encoder  = joblib.load("lcoder_region.joblib")

    scaler = joblib.load("standard_scaler.joblib")
    model = joblib.load("trained_liner_model.joblib")
    return sex_encoder, smoker_encoder, region_encoder , scaler, model
sex_encoder, smoker_encoder, region_encoder, scaler, model = get_model()

st.write("Sex encoder classes:", sex_encoder.classes_)


# Create tabs
tab1, tab2 = st.tabs(["Manual Entry", "Upload CSV"])

# =====================================================
# TAB 1: MANUAL ENTRY
# =====================================================
with tab1:
    st.subheader("Manual Data Entry")

    with st.form("insurance_form"):
        age = st.text_input("Age")
        bmi = st.text_input("BMI")
        children = st.text_input("Children")

        sex = st.text_input("Sex (male / female)")
        smoker = st.text_input("Smoker (yes / no)")
        region = st.text_input("Region (northwest / southeast / etc)")

        submit = st.form_submit_button("Test Prediction")

    if submit:
        if None in (age, bmi, children) or not sex or not smoker or not region:
            st.error("Please fill all fields")
        else:
            manual_df = pd.DataFrame({
                "age": [age],
                "bmi": [bmi],
                "children": [children],
                "sex": [sex.lower().strip()],
                "smoker": [smoker.lower().strip()],
                "region": [region.lower().strip()]
            })

            # Encode categorical columns
            manual_df["sex_label"] = sex_encoder.transform(manual_df["sex"])
            manual_df["smoker_label"] = smoker_encoder.transform(manual_df["smoker"])
            manual_df["region_label"] = region_encoder.transform(manual_df["region"])

            # Drop original categorical columns
            manual_df = manual_df.drop(columns=["sex", "smoker", "region"])

            # FORCE EXACT FEATURE SET
            # -------------------------------
            manual_df = manual_df.reindex(columns=list(scaler.feature_names_in_))

            # 🔍 DEBUG (TEMPORARY)
            st.write("Scaler expects columns:", list(scaler.feature_names_in_))
            st.write("Manual DF columns:", list(manual_df.columns))

            # Scale
            scaled_data = scaler.transform(manual_df)

            # Predict
            prediction = model.predict(scaled_data)

            st.success("Prediction Successful ✅")
            st.write("### Predicted Insurance Charges:")
            st.metric(label="Charges", value=f"{prediction[0]:,.2f}")

            predictions = model.predict(scaled_data)
            manual_df["Predicted_Charges"] = predictions

            st.write("### Prediction Results")
            st.dataframe(manual_df)

            if st.checkbox("Download Result CSV"):
                csv = convert_for_download(manual_df)
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv,
                    file_name="ins_predictions.csv",
                    mime="text/csv"
                )
            #st.success("Manual test data created ✅")
            #st.dataframe(manual_df)

            #manual_df.to_csv("insurance_test_manual.csv", index=False)
            #st.info("Saved as insurance_test_manual.csv")

# =====================================================
# TAB 2: CSV UPLOAD
# =====================================================
with tab2:
    st.subheader("Upload Test CSV File")

    uploaded_file = st.file_uploader(
        "Upload CSV with columns: age, bmi, children, sex, smoker, region",
        type=["csv"]
    )

    if uploaded_file is not None:
        upload_df = pd.read_csv(uploaded_file)

        st.success("File uploaded successfully ✅")
        st.write("### Preview of Uploaded Data")
        st.dataframe(upload_df)

        #upload_df.to_csv("insurance_test_uploaded.csv", index=False)
        #st.info("Saved as insurance_test_uploaded.csv")

        # Encode
        upload_df["sex_label"] = sex_encoder.transform(upload_df["sex"].str.lower())
        upload_df["smoker_label"] = smoker_encoder.transform(upload_df["smoker"].str.lower())
        upload_df["region_label"] = region_encoder.transform(upload_df["region"].str.lower())

        # DROP ORIGINAL STRING COLUMNS
        upload_df = upload_df.drop(columns=["sex", "smoker", "region"])

        # FORCE EXACT TRAINING FEATURES
        # -------------------------------
        upload_df = upload_df.reindex(columns=list(scaler.feature_names_in_))

        # Scale
        scaled_data = scaler.transform(upload_df)

        # Predict
        predictions = model.predict(scaled_data)
        upload_df["Predicted_Charges"] = predictions

        st.write("### Prediction Results")
        st.dataframe(upload_df)

        if st.checkbox("Download Result CSV"):
            csv = convert_for_download(upload_df)
            st.download_button(
                label="⬇️ Download CSV",
                data=csv,
                file_name="insurance_predictions.csv",
                mime="text/csv"
            )

    #df2 = scaler1.transform(upload_df)
    #prediction = model1.predict(df2)
    #upload_df['Pred. value'] = prediction
    #st.dataframe(upload_df)

    #f1 = st.checkbox("Downlod")
    #if f1:
     #   csv = convert_for_download(upload_df)
      #  st.download_button(
       #     label="Download CSV",
        #    data=csv,
         #   file_name="data.csv",
          #  mime="text/csv",
           # icon=":material/download:",
        #)
    #    upload_df["Predicted_" + Y] = prediction
#
#    st.dataframe(upload_df)



