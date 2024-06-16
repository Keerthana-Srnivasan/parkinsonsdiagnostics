#author: keerthana srinivasan
#date: 6/16/2024
#description: streamlit application for parkinson's diagnosis. input features are extracted from audio and fed into surrogate model


#import libraries and dependencies
import numpy as np #numpy is a library for numerical analysis and calculations
import streamlit as st #streamlit to run streamlit app
from st_audiorec import st_audiorec #st_audio rec is a sub-component of streamlit to capture audio as wav. file
import pandas as pd #to format the input features into a dataframe
import parselmouth #to calculate needed input features
import joblib #to upload surrogate model finalized_model.sav
from parselmouth.praat import call #to calculate needed input features
from sklearn.decomposition import PCA #principal component analysis
from sklearn.preprocessing import StandardScaler #preprocess and normalize data beforehand
import librosa #load signal (y) value and sr from wav. file

#update UI of streamlit
st.markdown("""
<style>
body {
    background: linear-gradient(to right, yellow, red);
}
</style>
""", unsafe_allow_html=True)

#basic description of application that would appear on a sidebar
with st.sidebar:
    st.write("This tool leverages voice-based biomarkers to predict the probability of Parkinson's Disease. The model used to process your data has been proven to carry near 100 percent accuracy.")
    st.write("Please do not play with the model by using it for any other purpose other than as a diagnostics tool, as this will cause hallucination.")
    wav_audio_data = st_audiorec()

#titles
st.title("Diagnostics Tool for Parkinson's Disease")
st.write("Results will show here once you record.")
view_results_button = st.button("View Results")

#function to find the period wave of the audio. the frequency is calculated using the to_pitch() function from parselmouth
def calculate_periods(file_path):
    snd = parselmouth.Sound(file_path)
    pitch = snd.to_pitch()
    frequencies = pitch.selected_array['frequency']
    periods = [1 / f for f in frequencies if f > 0] #period is inverse of frequency
    return periods, frequencies

#function to load model and output results
def model_load(feature_set):
    try:
        loaded_model = joblib.load("finalized_model.sav")
        y_data = loaded_model.predict(feature_set)
        st.write("Prediction: ", y_data)
    except Exception as e:
        st.error(f"An error occurred while making the prediction: {e}")

#if audio has been recorded, open the most recent wav. file, extrant the minimum and maximum frequencies using the calculate_periods() function
#find y and sr using librosa
#set values of harmonics_count and epsilon        
if wav_audio_data is not None:
    with open("recorded_audio.wav", "wb") as f:
        f.write(wav_audio_data)
        file_path = "recorded_audio.wav"
        periods, frequencies = calculate_periods(file_path)
        freq_max = max(frequencies)
        freq_min = max(min(frequencies), 2.5)
        y, sr = librosa.load(file_path, sr=11025) 
        harmonics_count = 10 
        epsilon = 0.05


def extract_features(voiceID, f0min, f0max, unit):
    sound = parselmouth.Sound(voiceID) # read the sound
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max) #create a praat pitch object
    meanF0 = call(pitch, "Get mean", 0, 0, unit) # get mean pitch
    stdevF0 = call(pitch, "Get standard deviation", 0 ,0, unit) # get standard deviation
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0) #get harmonicity to calculate harmonic to noise ratio (hnr)
    hnr = call(harmonicity, "Get mean", 0, 0) #get hnr
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) #jitter (%)
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3) #jitter (abs)
    rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3) #jitter (rap)
    ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3) #jitter (ppq)
    ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3) #jitter (ddp)
    localShimmer =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6) #shimmer (%)
    localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6) #shimmer (dB)
    apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6) #apq3
    aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6) #apq5
    apq11Shimmer =  call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6) #apq
    ddaShimmer = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6) #dda

    #format all needed features intoa  numpy array
    features = np.array([
            localJitter, 
            localabsoluteJitter,
            rapJitter,
            ppq5Jitter, 
            ddpJitter, 
            localShimmer,
            localdbShimmer, 
            apq3Shimmer, 
            aqpq5Shimmer, 
            apq11Shimmer,
            ddaShimmer
        ]).reshape(1, -1)

    #format numpy array into pandas dataframe
    audio_df = pd.DataFrame(features, columns=["MDVP:Jitter", "MDVP:Jitter.1", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", 
                                                "MDVP:Shimmer", "MDVP:Shimmer.1", "Shimmer:APQ3", "Shimmer:APQ5", 
                                                "MDVP:APQ", "Shimmer:DDA"])
    csv_data = audio_df.to_csv('recorded_data.csv', index=True)
    model_load(audio_df) #load dataframe into model for prediction

#extract_features() function will only run if the "view results" button is clicked
if view_results_button:
    features = extract_features(voiceID=file_path, f0min=freq_min, f0max=freq_max, unit="Hertz")