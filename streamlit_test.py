import streamlit_test as st

st.title("OpenCV Streamlit Practice")
st.header("Header")

file = st.file_uploader("Upload a file")

st.text("Hello World")

value = st.selectbox("Select", ['None', 'Choice 1', 'Choice 2'])

st.text("Text: " + value)
st.write("Write: " + value)

checkboxValue = st.checkbox("Apply Filter")
st.write(checkboxValue)