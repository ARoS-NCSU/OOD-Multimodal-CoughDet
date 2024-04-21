# Multimodal Cough Detection with Out-of-Distribution Detection

## Abstract
Cough detection is a crucial tool for long-term monitoring of respiratory illnesses. While clinical methods are accurate, they are not available in a home-based setting. In contrast,  earable devices offer a more accessible alternative, but face challenges in ensuring user speech privacy and detecting coughs accurately in real-world settings due to potential poor
audio quality and background noise. This study addresses these challenges by developing a small-size multimodal cough and speech detection system, enhanced with an Out-of- istribution (OOD) detection algorithm. Through our analyses, we demon- strate that combining transfer learning, a multimodal approach, and OOD detection techniques  ignificantly improves system performance. Without OOD inputs, the system shows high accuracies of 92.59% in the in-subject setting and 90.79% in the cross-subject setting.  ith  OD inputs, it still maintains overall accuracies of 91.97% and 90.31% in these respective settings by incorporating OOD detection, despite the number of OOD inputs being twice  hat of In-Distribution (ID) inputs. This research are promising towards a more efficient, user-friendly cough and speech detection method suitable for wearable devices.

![Overview.](/overview.png)
## Data Description
A total of 12 participants were involved in this study as approved by NC State University IRB Protocol 25003. The participants are student volunteers with similar age and health situation. Participants sat (∼ 2 min), walked (∼ 2 min), ran (∼ 2 min), walked (∼ 2 min), and sat (∼ 2 min) with 30-second resting intervals in each activation transition.

Audios were recorded by two chest-mounted microphones, one facing away from the participant (out-microphone) and one facing toward the participant (in-microphone). We  custom designed an enclosure and used microphones taken from commercially available Bluetooth earbuds (Tozo model T10 with the speaker circuit disconnected. The partici
pant’s movement was recorded with Mbientlab’s MetaMotionS r1 sensor mounted on chest capturing 9-axis IMU data. 

At the beginning of each recording, participants clapped three times and this procedure is used for data synchronization across different modalities. These three claps are distinctly observable in both the audio and IMU signals, producing accurate synchronization. The data was labeled using the open-source tool Audino. 

We categorize sounds into several classes: participant-generated sounds such as “Cough”, “Speech”, “Sneeze”, “Deep Breath”, “Groan”, and “Laugh”; “Speech (far)”, which represent speech from individuals around the subject; and “Other Sounds”, indicating unlabeled environmental noises; including periods of silence. 

### Data Pre-preocessing

Run `LabelConveter.ipynb` to convert audios and labels into `.mat` format. The processed data will be saved into ``processedData\v1`` folder. Then run `DataPreprocess.m` to sync with IMU signals. Sychronized files will be saved into ``processedData\v2`` folder.

- ``processedData\v1`` folder: Audio and annotation only

- ``processedData\v2`` folder: Audio, sychornized IMU, Annotations and corresponding timesamples and frequencys.

Annotation format: similar to one-hot encoding to save overlap annotation.
label index from 0 - 8 corresponding to ['Cough', 'Speech', 'Sneeze','Deep breath','Groan','Laugh', 'Speech (far)', 'Other Sound', 'Unkown']

## Multimodal Cough Detection model
The codes for models are in [src_models](https://github.com/ARoS-NCSU/OOD-Multimodal-CoughDet/tree/main/src_models)
