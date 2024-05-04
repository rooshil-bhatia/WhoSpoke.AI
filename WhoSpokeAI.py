import streamlit as st
from faster_whisper import WhisperModel
import datetime
import pandas as pd
import re
import time
import os
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment
import speechbrain
import torch
import librosa
import io
import traceback
from pyannote.audio import Model
from pyannote.audio import Inference
from pyannote.core import Segment as Seg
import sqlite3
import pickle

#This is Our Embedding Model.
embedding_model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

#To get Start and End Time Stamps
def convert_time(secs):
    return datetime.timedelta(seconds=round(secs))

#Main Code to convert speech to text and perform Diarization 
def speech_to_text(audio_file, selected_source_lang, whisper_model, num_speakers):
    #Compute Type Int8
    model = WhisperModel(whisper_model, compute_type="int8")
    time_start = time.time()

    try:
        # Get duration
        audio_content = audio_file.read()
        r=io.BytesIO(audio_content)
        audio_data, sample_rate = librosa.load(io.BytesIO(audio_content), sr=16000)#Sample Rate 16000
        duration = len(audio_data) / sample_rate 
        print(duration, 'this is the duration of my file')

        # Transcribe audio
        options = dict(language=selected_source_lang, beam_size=5, best_of=5)
        transcribe_options = dict(task="transcribe", **options)
        segments_raw, info = model.transcribe(io.BytesIO(audio_content), **transcribe_options)

        #Segments containing chunks of segments raw with Start, End and Text
        segments = []
        for segment_chunk in segments_raw:
            chunk = {}
            chunk["start"] = segment_chunk.start
            chunk["end"] = segment_chunk.end
            chunk["text"] = segment_chunk.text
            segments.append(chunk)

    except Exception as e:
        raise RuntimeError("Error reading audio")

    try:
        # Create embedding Function
        def segment_embedding(segment):
          try:
              audio = Audio()
              start = segment["start"]
              end = min(duration, segment["end"])
              clip = Segment(start, end)
              waveform, sample_rate = audio.crop(r, clip)
              embeddings = embedding_model(waveform[None])

              # Print embeddings information for debugging
              print("Embeddings shape:", embeddings.shape)

              return embeddings

          except Exception as e:
              traceback.print_exc()
              raise RuntimeError("Error during segment embedding", e)


        # Create embeddings
        embeddings = np.zeros(shape=(len(segments), 192))
        for i, segment in enumerate(segments):
            embeddings[i] = segment_embedding(segment)
        embeddings = np.nan_to_num(embeddings)

        if num_speakers == 0:
            
            score_num_speakers = {}

            for num_speakers in range(2, 5+1):
                clustering1 = AgglomerativeClustering(num_speakers, linkage='ward', metric='euclidean').fit(embeddings)
                score1 = silhouette_score(embeddings, clustering1.labels_, metric='euclidean')
                clustering2 = AgglomerativeClustering(num_speakers, linkage='complete', metric='l1').fit(embeddings)
                score2 = silhouette_score(embeddings, clustering2.labels_, metric='l1')
                clustering3 = AgglomerativeClustering(num_speakers, linkage='average', metric='l2').fit(embeddings)
                score3 = silhouette_score(embeddings, clustering3.labels_, metric='l2')
                clustering4 = AgglomerativeClustering(num_speakers, linkage='single', metric='l1').fit(embeddings)
                score4 = silhouette_score(embeddings, clustering4.labels_, metric='l1')
                clustering5 = AgglomerativeClustering(num_speakers, linkage='complete', metric='manhattan').fit(embeddings)
                score5 = silhouette_score(embeddings, clustering5.labels_, metric='manhattan')
                clustering6 = AgglomerativeClustering(num_speakers, linkage='complete', metric='cosine').fit(embeddings)
                score6 = silhouette_score(embeddings, clustering6.labels_, metric='cosine')
                clustering7 = AgglomerativeClustering(num_speakers, linkage='single', metric='cosine').fit(embeddings)
                score7 = silhouette_score(embeddings, clustering7.labels_, metric='cosine')
                clustering8 = AgglomerativeClustering(num_speakers, linkage='complete', metric='euclidean').fit(embeddings)
                score8 = silhouette_score(embeddings, clustering8.labels_, metric='euclidean')
                
                score_num_speakers[num_speakers]=[[score1, 'ward', 'euclidean'],
                                                  [score2, 'complete', 'l1'],
                                                  [score3, 'average', 'l2'],
                                                  [score4, 'single', 'l1'],
                                                  [score5, 'complete', 'manhattan'],
                                                  [score6, 'complete', 'cosine'],
                                                  [score7, 'single', 'cosine'],
                                                  [score8, 'complete', 'euclidean']]
            print(score_num_speakers)
            max_score=-2
            link=''
            met='' 
            best_num_speaker=-1                           
            for i in score_num_speakers:
              for j in score_num_speakers[i]:
                t=j[0]
                if t>max_score:
                  max_score=t
                  link=j[1]
                  met=j[2]
                  best_num_speaker=i
            print('best number of speakers',best_num_speaker)
            print('linkage',link)
            print('metric',met)
            print('max_score', max_score)
                

        else:
            best_num_speaker = num_speakers

        # Assign speaker label
        print('before last clustering', best_num_speaker)
        clustering = AgglomerativeClustering(best_num_speaker).fit(embeddings)
        labels = clustering.labels_
        print(labels)
        for i in range(len(segments)):
            segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

        # Make output
        objects = {
            'Start' : [],
            'End': [],
            'Speaker': [],
            'Text': []
        }
        text = ''
        for (i, segment) in enumerate(segments):
            if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
                objects['Start'].append(str(convert_time(segment["start"])))
                objects['Speaker'].append(segment["speaker"])
                if i != 0:
                    objects['End'].append(str(convert_time(segments[i - 1]["end"])))
                    objects['Text'].append(text)
                    text = ''
            text += segment["text"] + ' '
        objects['End'].append(str(convert_time(segments[i - 1]["end"])))
        objects['Text'].append(text)
        for key, value in objects.items():
          print(f"{key}: {value}")


        #for embeddings:
        def time_to_seconds(time_str):
          hours, minutes, seconds = map(int, time_str.split(':'))
          total_seconds = hours * 3600 + minutes * 60 + seconds
          return total_seconds
        


        #to extract exactly 1 voice embedding of all the unique speakers in the audio file.

        embed_dict={} #Dictionary containing key as a Unique Speaker Label and Value as it's Voice Embedding.
        df_results = pd.DataFrame(objects)
        start_list=df_results['Start'].apply(time_to_seconds)
        end_list=df_results['End'].apply(time_to_seconds)
        speaker_list=list(df_results['Speaker'])
        
        model_emb = Model.from_pretrained("pyannote/embedding", 
                              use_auth_token="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")
        
        inference = Inference(model_emb, window="whole")#Chose window = "whole" over "sliding" so that we can extract embeddings between our custom duration using total seconds.

        for i in range(len(start_list)):
            t=end_list[i]-start_list[i]
            if t>0 and speaker_list[i] not in embed_dict: #Time Stamp Duration should be atleast greater than 0 and speaker should not be there in embed_dict
              excerpt = Seg(start_list[i], end_list[i])
              embedding = inference.crop(r, excerpt)
              embed_dict[speaker_list[i]]=list(embedding)
        print(embed_dict)
              
        #Path of our Database 
        database_name = "/content/embeddings.db"

        # A function to insert Voice Embeddings to our Database 
        def insert_embeddings(database_name, embed_dict):
            conn = sqlite3.connect(database_name)
            cursor = conn.cursor()

            for label, embedding in embed_dict.items():
                # Convert the embedding to bytes using pickle
                embedding_bytes = pickle.dumps(embedding)
                try:
                    # Insert the label and embedding into the database
                    cursor.execute("INSERT INTO embeddings (Label, Embedding) VALUES (?, ?)", (label, embedding_bytes))
                except sqlite3.IntegrityError:
                    # If the label already exists, update the embedding
                    cursor.execute("UPDATE embeddings SET Embedding = ? WHERE Label = ?", (embedding_bytes, label))

            conn.commit()
            conn.close()
        


      
        # A function to check if our database is empty.
        def is_database_empty(database_name):
            conn = sqlite3.connect(database_name)
            cursor = conn.cursor()

            # Execute a SELECT query to check if there are any rows in the table
            cursor.execute("SELECT COUNT(*) FROM embeddings")
            count = cursor.fetchone()[0]

            # Close the connection
            conn.close()

            # If count is 0, the table is empty; otherwise, it's not empty
            return count == 0

        #This is a function which returns a list of Voice Embeddings stored in our database.     
        def view_embeddings(database_name):
            conn = sqlite3.connect(database_name)
            cursor = conn.cursor()

            # Select all rows from the embeddings table
            cursor.execute("SELECT * FROM embeddings")
            rows = cursor.fetchall()

            # Print each row (label and corresponding embedding)
            embedding_list=[]
            for row in rows:
                label, n, embedding_bytes = row
                embedding = pickle.loads(embedding_bytes)
                print("Label:", label)
                print("Embedding:", embedding)
                embedding_list.append(embedding)
                print()

            conn.close()
            return embedding_list

        # If Database is Empty then all the Voice Embeddings of unique speakers in embed_dict will be added to.
        if is_database_empty(database_name):
          insert_embeddings(database_name, embed_dict)
          print("Inside empty database Embeddings inserted successfully.")
        else:
          selected_dict={}    
          to_check=view_embeddings(database_name) #Retrieve all Voice Embeddings from the Database.

          from scipy.spatial.distance import cdist
          for i,j in embed_dict.items(): #a tuple containing (Speaker Label, Voice Embedding)
            count=0 
            for z in to_check:
              distance = cdist(np.array(j).reshape(1,512), np.array(z).reshape(1,512), metric="cosine")[0,0]
              print(distance)
              if distance>0.779: # Our Threshold.
                count+=1
            if count==len(to_check): #if count variable is equal to the lenght of the database this means the current Voice Embedding is not in database hence we will add it in select_dict.
              selected_dict[i]=j
          insert_embeddings(database_name, selected_dict)#Insterting only selected Voice Embeddings.
          print('Selected Embeddings have been added to the Database')

        #This is to create Voice Embeddings of All the Speakers with all their utterences.
        all_embed_list=[]
        df_results_2 = pd.DataFrame(objects)
        start_list_2=df_results_2['Start'].apply(time_to_seconds)
        end_list_2=df_results_2['End'].apply(time_to_seconds)
        all_speaker_list=list(df_results_2['Speaker'])
        
        model_emb = Model.from_pretrained("pyannote/embedding", 
                              use_auth_token="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")
        
        inference = Inference(model_emb, window="whole")
        for i in range(len(start_list_2)):
            t=end_list_2[i]-start_list_2[i]
            if t>0:
              excerpt = Seg(start_list_2[i], end_list_2[i])
              embedding = inference.crop(r, excerpt)
              all_embed_list.append(list(embedding))
            else:
                all_embed_list.append('Invalid')#If Start Time and End Time was same to t=0 hence Embedding model was giving error.


        #Same function as View Embeddings this one just returns a list of lists which is in the form[[name, embedding]]
        def view_embeddings_names(database_name):
            conn = sqlite3.connect(database_name)
            cursor = conn.cursor()

            # Select all rows from the embeddings table
            cursor.execute("SELECT * FROM embeddings")
            rows = cursor.fetchall()

            # Print each row (label and corresponding embedding)
            embedding_list=[]
            for row in rows:
                label, name, embedding_bytes = row
                embedding = pickle.loads(embedding_bytes)
                print("Label:", label)
                print("Embedding:", embedding)
                embedding_list.append([name,embedding])
                print()
            return embedding_list

            
        from scipy.spatial.distance import cdist
        names=[]#this is to add a names column in our Dataframe
        speaker_name=''
        to_check_all=view_embeddings_names(database_name)
        for i in all_embed_list:
          if i!='Invalid':
            mini=1000 #find the closet embedding in the database.
            for j in to_check_all: #j is [name, embedding]
                distance = cdist(np.array(i).reshape(1,512), np.array(j[1]).reshape(1,512), metric="cosine")[0,0]
                if distance<mini:
                  if j[0]:
                      speaker_name=j[0]
                  else:
                      speaker_name='Not in Database'
                  mini=distance
            if mini<=0.779: #closet embedding should be less than threshold value.
                names.append(speaker_name) #To append name as Speaker Name.
            else:
                names.append(speaker_name)#To append name as Not in Database.
          else:
            names.append('Time to short')#To append name as Time to Short.
        
        #Adding a Name Column to our DataFrame
        df_results["Name"]=list(names) 
        #Rearranging our Dataframe to show output.
        df_results = df_results[['Start', 'End', 'Speaker', 'Name', 'Text']] 
        
        #Calculating association of a Speaker Label with Speaker Name and taking max.
        most_frequent_n = df_results.groupby('Speaker')['Name'].agg(lambda x: x.value_counts().idxmax())
        #Populating the association in the whole Database for Each Speaker Label to correct any misidentification.
        df_results['Name'] = df_results.apply(lambda row: most_frequent_n[row['Speaker']] if row['Speaker'] in most_frequent_n.index else row['Name'], axis=1)
          

        return df_results

    except Exception as e:
        # Print exception for debugging
        print("Exception occurred:", e)
        raise RuntimeError("Error Running inference with local model", e)

st.title("Speech-to-Text App")
uploaded_file = st.file_uploader("Upload an audio file (WAV)", type=["wav","mp3"])
selected_option = st.selectbox("Select an option:", ["large-v2","large-v3"])
if uploaded_file:
    st.audio(uploaded_file, format="audio/wav", start_time=0)
    num_speakers = 0  # Set to 0 for automatic detection
    selected_source_lang = 'en'
    selected_whisper_model = selected_option

    if st.button("Process Audio"):
         df_results= speech_to_text(uploaded_file, selected_source_lang, selected_whisper_model, num_speakers)

         st.subheader("Transcript:")
         st.table(df_results)
