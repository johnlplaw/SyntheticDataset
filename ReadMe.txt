This project will develop a classification model to classify a text into emotion classes (the same emotions supported by text2emotion library). The coding demonstrates the pipeline process for generating a synthetic dataset when the training dataset for multiple languages dataset is unavailable. The pre-trained language models (mBERT, XLM-R) are to be tuned.

The structure of the coding is shown below:

data: The folder to store the data
----------------------------------------------------------
- SourceDataSets.xlsx - the file keeps the source data from the public repository

Common: Library (Folder: commons)
----------------------------------------------------------
- lang: Libraries for common functions
-- CleanText.py
-- GenCodeMixedSwitched.py
-- LibLang.py
-- LibNLP.py

- MySQL: Library for accessing MySQL
-- mysqlHelper.py

Part 1: Process of the synthetic dataset (Folder: Proc_Synthetic_Dataset)
-----------------------------------------------------------
- Step1_InsertDataInoDB.py
- Step2_CleanTheTweets.py
- Step3_Translate.py
- Step4_CodeMixed.py
- Step5_CodeSwitched.py

Part 2: Process of the model development (Folder: Proc_Pretrained_Model)
-----------------------------------------------------------
* Library
- PTM_Lib.py
- Variable.py

* Prepare the training dataset (Set 1)
- Prepare1_Load_Data_label.py
- Prepare1_OverView_Dataset_lbl.py
- Prepare2_Re-Sampling_dataset_lbl.py
- Prepare3a_Prepare_Training_dataset_lbl.py

* Identifying the best learning rate
- Preproc_Training_FB_XLM-R_ori_batch.py
- Preproc_Training_mBERT_ori_batch.py

* Tuning process
- TrainingProc_FB_XLM-R_ori_lbl_batch_Create.py
- TrainingProc_mBERT_ori_lbl_batch_Create.py

Part 3: Validate the synthetic dataset (Folder: Proc_Sync_Validation)
-----------------------------------------------------------
* Library
- ValidationFunction.py
- ValidationObject.py

* Validate
- Validate_similarity_type1.py
