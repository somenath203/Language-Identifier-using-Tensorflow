import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle


word_tokenizer = pickle.load(open('word_tokenizer.pkl', 'rb'))

model = load_model('textCNN_model.h5')


class_names = ['Arabic', 'Chinese', 'Dutch', 'English', 'Estonian', 'French', 'Hindi', 'Indonesian', 'Japanese', 'Korean', 'Latin', 'Persian', 'Portugese', 'Pushto', 'Romanian', 'Russian', 'Spanish', 'Swedish', 'Tamil', 'Thai', 'Turkish', 'Urdu']


def greetMe(languageTextInput):

    if languageTextInput == "":

        return "Please enter something in the input field for prediction."
    
    else:

        sample_data_sequence = word_tokenizer.texts_to_sequences([languageTextInput])

        sample_data_padded = pad_sequences(sample_data_sequence, padding='post', maxlen=100)

        prediction = model.predict(sample_data_padded)

        predicted_class = np.argmax(prediction)

        pred_final_result = class_names[predicted_class]

        return f'The predicted language is {pred_final_result}'


custom_css = """
    .desc { text-align: center; }
"""

myapp = gr.Interface(fn=greetMe,
                     description="<p class='desc'>" + "Please note that the current language prediction model supports a total of 22 languages, which include: Arabic, Chinese, Dutch, English, Estonian, French, Hindi, Indonesian, Japanese, Korean, Latin, Persian, Portuguese, Pashto, Romanian, Russian, Spanish, Swedish, Tamil, Thai, Turkish, and Urdu. For optimal accuracy in language prediction, it is recommended to provide input consisting of at least a 40-word paragraph. Inputs shorter than this may result in incorrect language predictions." + "</p>",
                     css=custom_css,
                     inputs=[
                         gr.Textbox(lines=20, placeholder="enter the original text whose language is to be predicted",
                                    label="Text Input Field", interactive=True)
                     ],
                     outputs=gr.Textbox(
                         lines=20, label="Predicted Language Output", show_copy_button=True),
                     allow_flagging='never')


if __name__ == "__main__":
    myapp.launch(show_api=False)
