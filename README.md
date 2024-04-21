Run `LabelConveter.ipynb` to convert audios and labels into `.mat` format. The processed data will be saved into ``processedData\v1`` folder. Then run `DataPreprocess.m` to sync with IMU signals. Sychronized files will be saved into ``processedData\v2`` folder.

- ``processedData\v1`` folder: Audio and annotation only

- ``processedData\v2`` folder: Audio, sychornized IMU, Annotations and corresponding timesamples and frequencys.

Annotation format: similar to one-hot encoding to save overlap annotation.
label index from 0 - 8 corresponding to ['Cough', 'Speech', 'Sneeze','Deep breath','Groan','Laugh', 'Speech (far)', 'Other Sound', 'Unkown']

The codes for models are in [src_models](https://github.com/ARoS-NCSU/OOD-Multimodal-CoughDet/tree/main/src_models)
