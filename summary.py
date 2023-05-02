import streamlit as st
from transformers import pipeline

# Set up the app title and description
st.title('Text Summarizer')
st.write('Enter the text to summarize and select the desired length of the summary.')

@st.cache_data # Cache the pipeline object to speed up processing
def summarizer():
    # Load the pre-trained summarization pipeline
    return pipeline('summarization', model='t5-small', tokenizer='t5-small')

def generate_summary(text, summary_length):
    # Remove any leading/trailing white space from the input text
    text = text.strip()

    # Use the summarization pipeline to generate the summary
    summarization = summarizer()
    summary = summarization(text, max_length=summary_length, min_length=int(summary_length*0.6), do_sample=False)
    summary = ''.join(summary[0]['summary_text'])

    return summary

def main():
    # Create a text input for the user to enter the text to summarize
    text = st.text_area('Enter the text to summarize:', height=200)

    # Create a slider for the user to select the length of the summary
    summary_length = st.slider('Select the length of the summary:', 30, 500, 100, step=10)
    # Create a button to generate the summary
    if st.button('Generate Summary'):
        # Check if the user has entered any text
        if not text:
            st.error('Please enter some text to summarize.')
        else:
            # Generate the summary
            summary = generate_summary(text, summary_length)
            
            # Display the summary
            st.write('Summary:')
            st.write(summary)

if __name__ == '__main__':
    main()
    st.write('**developed by Er Ashish KC Khatri**')
    st.text('Website: www.ashishkhatri.com.np')
    st.text('Contact: +977-9846262393')
    
