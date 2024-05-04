# WhoSpoke.AI


## Feature Extraction and Noise Reduction
 Pyannote and spkrec-ecapa-voxceleb: The system leverages Pyannote in conjunction with the spkrec-ecapa-voxceleb toolkit from SpeechBrain to extract robust audio features from recordings. This approach empowers accurate speaker differentiation even in challenging acoustic environments.
Improved Accuracy with Faster Whisper: Signal processing techniques are employed to refine the audio input, further bolstering transcription accuracy. By utilizing the Faster Whisper model, the system achieves a transcription fidelity of 95%.
## Automated Speaker Detection and Segmentation
Agglomerative Hierarchical Clustering (AHC): To streamline workflow efficiency and ensure precise speaker identification within transcripts, the system implements Agglomerative Hierarchical Clustering (AHC) for automatic speaker segmentation.
## Scalable Speaker Recognition with Embeddings
Speaker Embeddings for Future Identification: WhoSpoke.AI introduces a novel speaker diarization approach that utilizes voice embeddings. The system extracts speaker-specific features during the diarization process. These embeddings are subsequently stored in a dedicated database, facilitating efficient speaker recognition in future recordings. This approach lays the groundwork for real-time speaker identification in large audio datasets.
## Enhanced Usability with Speaker Name Integration
Speaker Name Integration: To further enhance the user experience of the transcribed conversations, WhoSpoke.AI integrates speaker names alongside the corresponding speech segments. This functionality retrieves speaker names from the database based on matching the incoming audio stream's voice embedding with entries within the database. This research paves the way for the development of speaker-centric applications with improved readability and usability.
