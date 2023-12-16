# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 15:30:22 2023

@author: vmurc
"""

import streamlit as st
import joblib
import numpy as np
import random 
import pickle

# Load your trained and calibrated model
# To load the model in Streamlit
def main():
    st.title('Predict Squirrel Approach')

    # load model
    with open('cal_Squirrel_RF2.pk2', 'rb') as f:
        squirrel_model = pickle.load(f)

    #with open('squirrel_kmeans.pkl', 'rb') as f:
    #    kmeans = pickle.load(f)
    # Load your trained and calibrated model
    #squirrel_model = pickle.load(open('cal_Squirrel_RF.pkl','rb'))#joblib.load('cal_Squirrel_RF.pkl')
    kmeans =         joblib.load('squirrel_kmeans.pkl')  # Adjust the file path as needed

    with st.expander("Location and Time"):
        col1, col2, col3 = st.columns(3)
        with col1:
            X = st.number_input('X Coordinate', format="%.2f")
        with col2:
            Y = st.number_input('Y Coordinate', format="%.2f")
            cluster = kmeans.predict(np.array([[X, Y]]))[0]
        with col3:
            time_of_day = st.selectbox('Time of Day', options=['AM', 'PM'])
            Daytime = 1 if time_of_day == 'AM' else 0

    with st.expander("Squirrel Characteristics"):
        col1, col2, col3 = st.columns(3)
        with col1:
            squirrel_age = st.selectbox('Squirrel Age', options=['Adult', 'Juvenile', 'Unknown'])
            Age_Adult, Age_Juvenile, Age_Unknown = 0, 0, 0
            if squirrel_age == 'Adult':
                Age_Adult = 1
            elif squirrel_age == 'Juvenile':
                Age_Juvenile = 1
            elif squirrel_age == 'Unknown':
                Age_Unknown = 1
            
        with col2:
            fur_color = st.selectbox('Primary Fur Color', options=['Black', 'Cinnamon', 'Gray', 'Unknown'])
            PFC_Black, PFC_Cinnamon, PFC_Gray, PFC_Unknown = 0, 0, 0, 0
            if fur_color == 'Black':
                PFC_Black = 1
            elif fur_color == 'Cinnamon':
                PFC_Cinnamon = 1
            elif fur_color == 'Gray':
                PFC_Gray = 1
            elif fur_color == 'Unknown':
                PFC_Unknown = 1
        with col3:   
            highlight_color = st.selectbox('Highlight Fur Color', options=['Black', 'Cinnamon', 'Gray', 'Mixed', 'Unknown', 'White'])
            HFC_Black, HFC_Cinnamon, HFC_Gray, HFC_Mixed, HFC_Unknown, HFC_White = 0, 0, 0, 0, 0, 0
            if highlight_color == 'Black':
                HFC_Black = 1
            elif highlight_color == 'Cinnamon':
                HFC_Cinnamon = 1
            elif highlight_color == 'Gray':
                HFC_Gray = 1
            elif highlight_color == 'Mixed':
                HFC_Mixed = 1
            elif highlight_color == 'Unknown':
                HFC_Unknown = 1
            elif highlight_color == 'White':
                HFC_White = 1

    with st.expander("Squirrel Behaviors"):
        col1, col2,col3 = st.columns(3)
        with col1:
            Located_on_Ground = st.checkbox('Located on Ground')
            Running = st.checkbox('Running')
            Chasing = st.checkbox('Chasing')
            Climbing = st.checkbox('Climbing')
            Eating = st.checkbox('Eating')
        with col2:
            Foraging = st.checkbox('Foraging')
            Kuks = st.checkbox('Kuks')
            Quaas = st.checkbox('Quaas')
            Moans = st.checkbox('Moans')
            Tail_flags = st.checkbox('Tail flags')
        with col3:
            Tail_twitches = st.checkbox('Tail twitches')
            Indifferent = st.checkbox('Indifferent')
            Runs_from = st.checkbox('Runs from')
            Weekday = st.checkbox('Weekday')
            
    with st.expander("Seen In"):
        col1, col2 = st.columns(2)
        with col1:
            Seen_in_Tree = st.checkbox('Seen in Tree')
            Seen_in_Shrubbery = st.checkbox('Seen in Shrubbery')
            Seen_in_Rock = st.checkbox('Seen in Rock')
            Seen_in_Grassland = st.checkbox('Seen in Grassland')
        with col2:
            Seen_in_Path = st.checkbox('Seen in Path')
            Seen_in_Structure = st.checkbox('Seen in Structure')
            Seen_in_Water = st.checkbox('Seen in Water')

    with st.expander("Other Activities"):
        col1, col2, col3 = st.columns(3)
        with col1:
            Playful = st.checkbox('Playful')
        with col2:
            Digging = st.checkbox('Digging')
        with col3:
            Relaxing = st.checkbox('Relaxing')

    if st.button('Predict'):
        
        features = [X, Y, Daytime, Located_on_Ground, Running, Chasing,Climbing,Climbing,Eating,Foraging,
                    Kuks,Quaas,Moans,Tail_flags,Tail_twitches,Indifferent,Runs_from,Weekday,
                    Age_Adult, Age_Juvenile, Age_Unknown, 
                    PFC_Black, PFC_Cinnamon, PFC_Gray, PFC_Unknown, 
                    HFC_Black, HFC_Cinnamon, HFC_Gray, HFC_Mixed, HFC_Unknown, HFC_White,
                    Seen_in_Tree, Seen_in_Shrubbery, Seen_in_Rock, Seen_in_Grassland, Seen_in_Path, Seen_in_Structure, Seen_in_Water,
                    Playful,Digging,Relaxing,cluster
                    ]
        features = [float(features[0]), float(features[1])] + [int(val) for val in features[2:]]
        prediction = squirrel_model.predict_proba([features])[0, 1] * 100
        st.write(f'Squirrel approach probability: {prediction:.3f}%')
        # Display image based on prediction
        if prediction < 0.5:
            image_path = random.choice(['sad_squirrel1.jpg', 'sad_squirrel2.jpg'])
            st.image(image_path, caption='Sad Squirrel :(')
        else:
            image_path = random.choice(['happy_squirrel1.jpg', 'happy_squirrel2.jpg'])
            st.image(image_path, caption='Happy Squirrel :)')

if __name__ == '__main__':
    main()
