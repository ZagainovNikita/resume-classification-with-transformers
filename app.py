import streamlit as st

from model import create_classifier, preprocess_text, make_prediction

model = create_classifier(25)


def main():
    st.title("Resume scanner")
    file = st.file_uploader("Upload your resume", type=["txt", "docx", "pdf"])

    if file is not None:
        try:
            resume_bytes = file.read()
            resume_text = resume_bytes.decode("utf-8")
            resume_text = preprocess_text(resume_text)
            prediction = make_prediction(model, resume_text)
            st.text(f"This resume is application for the {prediction} position")

        except UnicodeDecodeError:
            st.text("Failed to decode your file. Make sure it is valid")


if __name__ == "__main__":
    main()
