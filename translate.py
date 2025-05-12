from googletrans import Translator, LANGUAGES
from gtts import gTTS
import os

# Initialize the translator
translator = Translator()

# Function to translate text and save audio in all languages
def translate_and_save_audio_in_all_languages(text, audio_folder='Useful_scripts/audio'):
    try:
        # Create audio folder if it doesn't exist
        if not os.path.exists(audio_folder):
            os.makedirs(audio_folder)
        
        # Loop over all available languages
        for lang_code, lang_name in LANGUAGES.items():
            print(f"Translating to: {lang_name} ({lang_code})")
            try:
                # Translate the text
                translated = translator.translate(text, dest=lang_code)
                translated_text = translated.text
                print(f"Translated text: {translated_text}")

                # Generate the audio file
                try:
                    tts = gTTS(translated_text, lang=lang_code)
                    safe_lang_name = lang_name.replace(" ", "_")
                    audio_path = os.path.join(audio_folder, f"{safe_lang_name}.mp3")
                    tts.save(audio_path)
                    print(f"Audio file saved: {audio_path}")
                except ValueError:
                    print(f"Error: The language '{lang_name}' ({lang_code}) is not supported by gTTS.")
            
            except Exception as e:
                print(f"An error occurred while translating to {lang_name} ({lang_code}): {e}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Sample input text
text_to_translate = r"The cat sat quietly on the mat, watching the world with sleepy eyes as the quick brown fox leapt gracefully over the lazy dog, its fur glistening in the morning sun. Curious about the commotion, she wondered aloud, Why did the man close the door? As if answering her question, the rain finally stopped, prompting the children to rush outside, their laughter echoing through the air. Despite searching everywhere, she couldn't find the book on the table, and with a sigh of resignation, she turned away, leaving the room to settle into silence once more."

# Call the function to translate and save in all languages
translate_and_save_audio_in_all_languages(text_to_translate)