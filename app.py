import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import random
from datetime import datetime

# Load the emotion detection model
pipe_lr = joblib.load(open("model/text_emotion.pkl", "rb"))

# Dictionary mapping emotions to emoji icons
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱",
    "joy": "😊", "neutral": "😐", "sadness": "😔",
    "shame": "😳", "surprise": "😮"
}
# Dictionary mapping emotions to prompts
emotion_prompts_dict = {
    "joy": "I'm glad you are happy. Want to speak more about it?",
    "sadness": "I'm here to listen. Would you like to share more?",
    "anger": "I understand. Do you want to talk more about it?",
    "fear": "It's okay to feel scared sometimes. Do you want to share more?",
    "disgust": "That sounds tough. Do you want to talk more about it?",
    "surprise": "Wow, that's interesting! Do you want to tell me more?",
    "shame": "It's okay to feel ashamed. Do you want to talk about it?",
    "neutral": "I see. Do you want to talk more about it?"
}

# Path to the dataset containing suggestions for each emotion
suggestions_dataset_path = "/Users/samd26013/PycharmProjects/Text Emotion Detection/suggestions.csv"


def predict_emotions(docx):
    # Predict the emotion for the given text
    results = pipe_lr.predict([docx])
    return results[0]

def get_random_suggestions(emotion):
    # Get four random suggestions for the given emotion from the dataset
    suggestions_df = pd.read_csv(suggestions_dataset_path)
    emotion_suggestions = suggestions_df[suggestions_df['Emotion'] == emotion]['Suggestion']
    random_suggestions = random.sample(emotion_suggestions.tolist(), min(len(emotion_suggestions), 4))
    return random_suggestions

def get_prediction_proba(docx):
    # Get the probability distribution over emotions for the given text
    results = pipe_lr.predict_proba([docx])
    return results


def main():
    # Create session state for user login and history visibility
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'show_history' not in st.session_state:
        st.session_state.show_history = False
    if 'display_feeling_input' not in st.session_state:
        st.session_state.display_feeling_input = False

    # Read the existing dataset
    existing_dataset_path = "/Users/samd26013/PycharmProjects/Text Emotion Detection/history.csv"
    history_df = pd.read_csv(existing_dataset_path)

    st.set_page_config(
        page_title="Therapy Moo",
        page_icon=":heart:",
        layout="wide"
    )

    st.markdown("<h1><span style='text-decoration: underline;'>Therapy Moo</span> 🐮</h1>", unsafe_allow_html=True)
    st.markdown(" ")

    # User login
    st.sidebar.title("User Login")
    user_id = st.sidebar.text_input("Enter User ID")

    if st.sidebar.button("Login"):
        st.session_state.user_id = user_id

    if st.session_state.user_id:
        st.write(f"Logged in as: {st.session_state.user_id}")

        reset_button = st.button("Reset Input")
        if reset_button:
            st.session_state.display_feeling_input = False

        with st.form(key='my_form'):
            st.info("Hello there! How can I help you today?")
            raw_text = st.text_area("Type here:")
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            st.session_state.display_feeling_input = True

    if st.session_state.display_feeling_input:
        with st.form(key='feeling_form'):
            emotion_first_input = predict_emotions(raw_text)
            prompt = emotion_prompts_dict[emotion_first_input]
            st.info(prompt)
            feeling_input = st.text_area("Type here:", key="feeling_input")
            feeling_submit = st.form_submit_button(label='Submit')

        if feeling_submit:
            # Predict emotion and get probability distribution for the initial input
            raw_prediction = predict_emotions(raw_text)
            raw_probability = get_prediction_proba(raw_text)

            # Predict emotion and get probability distribution for the feeling input
            combined_text = raw_text + " " + feeling_input
            feeling_prediction = predict_emotions(combined_text)
            feeling_probability = get_prediction_proba(combined_text)

            # Compare confidence scores for both predictions and choose the higher confidence as the resultant emotion
            raw_confidence = np.max(raw_probability)
            feeling_confidence = np.max(feeling_probability)

            if raw_confidence > feeling_confidence:
                resultant_emotion = raw_prediction
                resultant_probability = raw_probability
            else:
                resultant_emotion = feeling_prediction
                resultant_probability = feeling_probability

            col1, col2, col3 = st.columns([1, 3, 2])

            with col1:
                # Display emotion prediction
                emoji_icon = emotions_emoji_dict[resultant_emotion]
                st.success("Predicted Emotion")
                st.write(f"{resultant_emotion.capitalize()} {emoji_icon}")

                # Display confidence score
                st.write(f"Confidence: {np.max(resultant_probability):.2f}")

            with col2:
                # Display personalized suggestions based on the resultant emotion
                st.success("Personalized Suggestions")
                random_suggestions = get_random_suggestions(resultant_emotion)
                for suggestion in random_suggestions:
                    st.write(f"- {suggestion}")

            with col3:
                # Display probability distribution chart
                st.success("Prediction Probability")
                proba_df = pd.DataFrame(resultant_probability, columns=pipe_lr.classes_)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions", "probability"]

                fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
                st.altair_chart(fig, use_container_width=True)

            # Get the current date and time
            current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            new_row = pd.DataFrame([[current_date, raw_text, feeling_input, resultant_emotion,
                                     ", ".join(random_suggestions), st.session_state.user_id]],
                                   columns=history_df.columns)
            history_df = pd.concat([new_row, history_df], ignore_index=True)

            # Save the updated dataset
            history_df.to_csv(existing_dataset_path, index=False)

    # Display history dataframe for the logged-in user when the button is clicked
    if st.session_state.user_id:
        show_history = st.button("Show History")
        if show_history:
            st.session_state.show_history = not st.session_state.show_history

        if st.session_state.show_history:
            st.success("History")
            user_history_df = history_df[history_df['User_ID'] == st.session_state.user_id]
            for index, row in user_history_df.iterrows():
                st.write("---")
                st.write(f"Entry Date: {row['Date']}")
                st.write(f"Input 1: {row['Input']}")
                st.write(f"Input 2: {row['Feeling']}")
                st.write(f"Predicted Emotion: {row['Predicted Emotion']}")
                st.write(f"Suggested Actions: {row['Suggested Actions']}")
                st.write("---")
        elif not st.session_state.show_history:
            st.warning("History is hidden. Click 'Show History' to view.")
    elif not st.session_state.user_id:
        st.markdown("""
            <div style='background-color: rgba(184, 0, 42, 0.4); padding: 20px; text-align: center; border-radius: 15px;'>
            <h3>Welcome to Therapy Moo, your personal therapy buddy!</h3> 

            <p>This interactive platform utilizes advanced text emotion detection technology to provide personalized support and suggestions based on the emotions conveyed in your text. Simply log in, share your thoughts, and receive predictions about your emotions, along with tailored suggestions to help you navigate your feelings.</p> 

            <p>Whether you're seeking comfort, inspiration, or guidance, Therapy Moo is here to lend a listening ear and offer valuable insights to support your emotional well-being. Start your journey towards greater self-awareness and emotional wellness today with Therapy Moo!</p>
            </div>
        """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
