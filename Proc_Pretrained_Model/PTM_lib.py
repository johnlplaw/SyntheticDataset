import torch
from transformers import AdamW
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
import time
from torchviz import make_dot
import os
import matplotlib.pyplot as plt
import Variable as var
import pickle
import pandas as pd
import random
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import roc_curve, auc
import commons.mysql.mysqlHelper as sqlHelper
import mysql.connector

def get_full_dataSet(langType, samplingType):
    """
    Get the dataset in full
    :param langType: Langauge type
    :param samplingType: Sampling Type
    :return: The selected dataset in full
    """
    txt_file = ''
    label_file = ''
    if langType == var.LANG_TYPE_ENG and samplingType == var.SAMPLING_TYPE_OVER:
        txt_file = var.FILE_OVERSAMPLING_ENG_TEXT_DATASET_LBL
        label_file = var.FILE_OVERSAMPLING_ENG_LABEL_DATASET_LBL
    elif langType == var.LANG_TYPE_MULTI and samplingType == var.SAMPLING_TYPE_OVER:
        txt_file = var.FILE_OVERSAMPLING_MULTI_TEXT_DATASET_LBL
        label_file = var.FILE_OVERSAMPLING_MULTI_LABEL_DATASET_LBL
    elif langType == var.LANG_TYPE_COMBINED_MULTI and samplingType == var.SAMPLING_TYPE_OVER:
        txt_file = var.FILE_OVERSAMPLING_MULTICOMB_TEXT_DATASET_LBL
        label_file = var.FILE_OVERSAMPLING_MULTICOMB_LABEL_DATASET_LBL
    elif langType == var.LANG_TYPE_ENG and samplingType == var.SAMPLING_TYPE_UNDER:
        txt_file = var.FILE_UNDERSAMPLING_ENG_TEXT_DATASET_LBL
        label_file = var.FILE_UNDERSAMPLING_ENG_LABEL_DATASET_LBL
    elif langType == var.LANG_TYPE_MULTI and samplingType == var.SAMPLING_TYPE_UNDER:
        txt_file = var.FILE_UNDERSAMPLING_MULTI_TEXT_DATASET_LBL
        label_file = var.FILE_UNDERSAMPLING_MULTI_LABEL_DATASET_LBL
    elif langType == var.LANG_TYPE_COMBINED_MULTI and samplingType == var.SAMPLING_TYPE_UNDER:
        txt_file = var.FILE_UNDERSAMPLING_MULTICOMB_TEXT_DATASET_LBL
        label_file = var.FILE_UNDERSAMPLING_MULTICOMB_LABEL_DATASET_LBL

    fileObj = open(txt_file, 'rb')
    text = pickle.load(fileObj)
    fileObj.close()
    fileObj = open(label_file, 'rb')
    label = pickle.load(fileObj)
    fileObj.close()

    return text, label

def get_subset_combine_dataset(samplingType, sample_count = -1, random_seed = 42):
    txt_file = ''
    label_file = ''
    if samplingType == var.SAMPLING_TYPE_OVER:
        txt_file = var.FILE_OVERSAMPLING_MULTICOMB_TEXT_DATASET_LBL
        label_file = var.FILE_OVERSAMPLING_MULTICOMB_LABEL_DATASET_LBL
    elif samplingType == var.SAMPLING_TYPE_UNDER:
        txt_file = var.FILE_UNDERSAMPLING_MULTICOMB_TEXT_DATASET_LBL
        label_file = var.FILE_UNDERSAMPLING_MULTICOMB_LABEL_DATASET_LBL

    fileObj = open(txt_file, 'rb')
    multi_combined_db_under_txt_1 = pickle.load(fileObj)
    fileObj.close()
    fileObj = open(label_file, 'rb')
    multi_combined_db_under_label_1 = pickle.load(fileObj)
    fileObj.close()

    cleanedtxt_list = []
    translate_chn_list = []
    translate_my_list = []
    translate_tm_list = []
    cm_en_chn_list = []
    cm_en_my_list = []
    cm_en_tm_list = []
    cm_chn_en_list = []
    cm_chn_my_list = []
    cm_chn_tm_list = []
    cm_my_en_list = []
    cm_my_chn_list = []
    cm_my_tm_list = []
    cm_tm_en_list = []
    cm_tm_chn_list = []
    cm_tm_my_list = []
    cw_en_chn_list = []
    cw_en_my_list = []
    cw_en_tm_list = []
    cw_chn_en_list = []
    cw_chn_my_list = []
    cw_chn_tm_list = []
    cw_my_en_list = []
    cw_my_chn_list = []
    cw_my_tm_list = []
    cw_tm_en_list = []
    cw_tm_chn_list = []
    cw_tm_my_list = []

    for item in multi_combined_db_under_txt_1:
        print(len(item))
        cleanedtxt_list.append(item[0])
        translate_chn_list.append(item[1])
        translate_my_list.append(item[2])
        translate_tm_list.append(item[3])
        cm_en_chn_list.append(item[4])

        cm_en_my_list.append(item[5])
        cm_en_tm_list.append(item[6])
        cm_chn_en_list.append(item[7])
        cm_chn_my_list.append(item[8])
        cm_chn_tm_list.append(item[9])

        cm_my_en_list.append(item[10])
        cm_my_chn_list.append(item[11])
        cm_my_tm_list.append(item[12])
        cm_tm_en_list.append(item[13])
        cm_tm_chn_list.append(item[14])

        cm_tm_my_list.append(item[15])
        cw_en_chn_list.append(item[16])
        cw_en_my_list.append(item[17])
        cw_en_tm_list.append(item[18])
        cw_chn_en_list.append(item[19])

        cw_chn_my_list.append(item[20])
        cw_chn_tm_list.append(item[21])
        cw_my_en_list.append(item[22])
        cw_my_chn_list.append(item[23])
        cw_my_tm_list.append(item[24])

        cw_tm_en_list.append(item[25])
        cw_tm_chn_list.append(item[26])
        cw_tm_my_list.append(item[27])

    dict = {
        'std_label': multi_combined_db_under_label_1,
        'cleanedtxt':cleanedtxt_list,
        'translate_chn':translate_chn_list,
        'translate_my':translate_my_list,
        'translate_tm':translate_tm_list,
        'cm_en_chn':cm_en_chn_list,
        'cm_en_my':cm_en_my_list,
        'cm_en_tm':cm_en_tm_list,
        'cm_chn_en':cm_chn_en_list,
        'cm_chn_my':cm_chn_my_list,
        'cm_chn_tm':cm_chn_tm_list,
        'cm_my_en':cm_my_en_list,
        'cm_my_chn':cm_my_chn_list,
        'cm_my_tm':cm_my_tm_list,
        'cm_tm_en':cm_tm_en_list,
        'cm_tm_chn':cm_tm_chn_list,
        'cm_tm_my':cm_tm_my_list,
        'cw_en_chn':cw_en_chn_list,
        'cw_en_my':cw_en_my_list,
        'cw_en_tm':cw_en_tm_list,
        'cw_chn_en':cw_chn_en_list,
        'cw_chn_my':cw_chn_my_list,
        'cw_chn_tm':cw_chn_tm_list,
        'cw_my_en':cw_my_en_list,
        'cw_my_chn':cw_my_chn_list,
        'cw_my_tm':cw_my_tm_list,
        'cw_tm_en':cw_tm_en_list,
        'cw_tm_chn':cw_tm_chn_list,
        'cw_tm_my_list':cw_tm_my_list
    }

    df = pd.DataFrame(dict)
    if sample_count != -1 and sample_count <= len(df):
        random.seed(random_seed)
        result_df = df.sample(n=sample_count)
        return result_df
    else:
        return df




def Main_trainingModel(modelType, tokenizer, model, texts, labels, BATCH_SIZE, LEARNING_RATE, num_epochs):
    # For grahic
    os.environ["PATH"] += os.pathsep + '/opt/homebrew/bin/'

    # Print model architecture and details
    print("Model architecture:")
    print(model)

    # Print model configuration
    print("\nModel configuration:")
    print(model.config)

    # Tokenize input texts
    tokenized_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    # Build dataset and dataloader
    input_ids = tokenized_inputs["input_ids"]
    attention_mask = tokenized_inputs["attention_mask"]
    labels = torch.tensor(labels)

    dataset = TensorDataset(input_ids, attention_mask, labels)

    train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Build and train the model
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # Use CrossEntropyLoss for multi-class classification
    criterion = torch.nn.CrossEntropyLoss()

    # Move model to GPU if available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # data for monitoring of the training process
    epochsList = []
    train_accuracyList = []
    val_accuracyList = []
    train_lossList = []
    val_lossList = []

    for epoch in range(num_epochs):
        start_time = time.monotonic()  # Record the start time for the epoch
        model.train()
        total_correct = 0
        total_samples = 0
        total_loss = 0.0

        for batch in train_dataloader:
            batch = tuple(t.to(device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
            labels = batch[2].to(device)
            outputs = model(**inputs)
            logits = outputs.logits
            loss = criterion(logits, labels)
            total_loss += loss.item()

            _, predicted_labels = torch.max(logits, 1)
            total_correct += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

            optimizer.zero_grad()
            loss.backward()

            # Applying dropout manually
            #---------
            # Use the default dropout_rate = 0.1
            # dropout_rate = 0.2
            # for name, param in model.named_parameters():
            #     if 'dropout' in name:
            #         param.grad = param.grad * (1 - dropout_rate)
            # ---------
            optimizer.step()

        accuracy = total_correct / total_samples
        avg_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_loss:.4f}, Training Accuracy: {accuracy:.4f}')

        # Validation
        model.eval()
        total_correct_val = 0
        total_samples_val = 0
        total_loss_val = 0.0

        with torch.no_grad():
            for batch in val_dataloader:
                batch = tuple(t.to(device) for t in batch)
                inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
                labels = batch[2].to(device)
                outputs = model(**inputs)
                logits = outputs.logits
                loss = criterion(logits, labels)
                total_loss_val += loss.item()

                _, predicted_labels_val = torch.max(logits, 1)
                total_correct_val += (predicted_labels_val == labels).sum().item()
                total_samples_val += labels.size(0)

                batch_probs = F.softmax(logits, dim=1)


        accuracy_val = total_correct_val / total_samples_val
        avg_loss_val = total_loss_val / len(val_dataloader)

        # Calculate and print the time spent
        end_time = time.monotonic()
        epoch_time = end_time - start_time

        print(
            f'Epoch {epoch + 1}/{num_epochs}, Time: {epoch_time:.2f}s, Validation Loss: {avg_loss_val:.4f}, Validation Accuracy: {accuracy_val:.4f}')

        # Keeping training info
        epochsList.append(epoch)
        train_accuracyList.append(accuracy)
        val_accuracyList.append(accuracy_val)
        train_lossList.append(avg_loss)
        val_lossList.append(avg_loss_val)

        # save the info into database
        record_training(modelType, epoch, avg_loss, accuracy, avg_loss_val, accuracy_val, epoch_time)

    # Save the trained model if needed
    torch.save(model.state_dict(), var.DIR_OUTPUT + modelType + '_model.pth')

    # Create a graph of the computation
    inputs = None
    for batch in val_dataloader:
        batch = tuple(t.to(device) for t in batch)
        inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
        break
    outputs = model(**inputs)
    graph = make_dot(outputs.logits, params=dict(model.named_parameters()))

    # Save the graph as an image (PNG)
    graph.render(filename=var.DIR_OUTPUT + modelType + '_model', format='png', cleanup=True)

    # Save the trained model if needed
    torch.save(model.state_dict(), var.DIR_OUTPUT + modelType + '_model.pth')

    # Generate graph
    # Plotting accuracy
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochsList, train_accuracyList, label='Training Accuracy', marker='o')
    plt.plot(epochsList, val_accuracyList, label='Validation Accuracy', marker='o')
    plt.title('Epoch vs Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plotting average loss
    plt.subplot(1, 2, 2)
    plt.plot(epochsList, train_lossList, label='Training Loss', marker='o')
    plt.plot(epochsList, val_lossList, label='Validation Loss', marker='o')
    plt.title('Epoch vs Average Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.legend()

    plt.tight_layout()

    # Save the figure to an image file (e.g., PNG)
    plt.savefig(var.DIR_OUTPUT + modelType + '_training_graph.png')

    # Show the plot (optional)
    # plt.show()


def evaluation(combineDF, tokenizer, model, LABELS):

    model.eval()

    #for lang_type in var.language_type:
    for lang_type in ['translate_chn']:

        print("Working on " + lang_type + " ... start")
        training_txt = combineDF[lang_type].tolist()
        training_label = combineDF['std_label'].tolist()
        predicted_labelList = []
        all_probs = []
        all_labels = []
        for i in range(0, len(training_txt)):
            text = training_txt[i]
            label = training_label[i]

            # Tokenize input text
            inputs = tokenizer(text, return_tensors='pt')

            # Perform classification
            with torch.no_grad():
                outputs = model(**inputs)

            # Get predicted label (assuming binary classification)
            predicted_label = torch.argmax(outputs.logits, dim=1).item()
            predicted_labelList.append(predicted_label)

            batch_probs = F.softmax(outputs.logits, dim=1)
            all_probs.append(batch_probs)

        all_probs = torch.cat(all_probs, dim=0)

        training_label_str = [str(i) for i in training_label]
        predicted_labelList_str = [str(i) for i in predicted_labelList]

        conf_matrix = confusion_matrix(training_label_str, predicted_labelList_str, labels=LABELS)
        print("Confusion Matrix:")
        print(conf_matrix)
        ## the rows represent the true labels, and the columns represent the predicted labels.

        accuracy = accuracy_score(training_label_str, predicted_labelList_str)
        print("Accuracy:")
        print(accuracy)

        # Precision - How accuracy of positive prediction
        precision = precision_score(training_label_str, predicted_labelList_str, average=None, labels=LABELS)
        # Recall - ability of the classifier to find all positive instance.
        recall = recall_score(training_label_str, predicted_labelList_str, average=None, labels=LABELS)
        # f1 - high if both precision and recall are high
        f1 = f1_score(training_label_str, predicted_labelList_str, average=None, labels=LABELS)

        # Print precision, recall, and F1 score for each class
        for i, label in enumerate(LABELS):
            print(f" Class {label}:")
            print(f" Precision: {precision[i]}")
            print(f" Recall: {recall[i]}")
            print(f" F1 Score: {f1[i]}")
            print()

        print("Working on " + lang_type + " ... done")
        print("++++++++++++++++++++++++++")

        print(np.unique(training_label))

        print("++++++++++++++++++++++++++")

        # Compute ROC curve and AUC for each class
        plt.figure(figsize=(8, 6))

        training_label2 = [torch.tensor(int(elem)) for elem in training_label]
        training_label2_tensor = torch.tensor(training_label2)
        training_label2_reshaped = training_label2_tensor.reshape(-1, 1)
        for i in range(len(np.unique(training_label))):
            fpr, tpr, _ = roc_curve(training_label2_reshaped == i, all_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label='ROC curve (class %d) (AUC = %0.2f)' % (i, roc_auc))

        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve for Multi-class Classification')
        plt.legend(loc="lower right")
        #plt.show()
        plt.savefig(lang_type + '_roc.png')


def get_DF_LBL_file_path(lang, size):
    file_name = ""
    if lang == var.LANG_TYPE_ENG:
        file_name = var.FILE_TRAINING_ENG_NOS_LBL + str(size) + ".obj"
    elif lang == var.LANG_TYPE_MULTI:
        file_name = var.FILE_TRAINING_MUL_NOS_LBL + str(size) + ".obj"

    return file_name

def get_DF_PLBL_file_path(lang, size):
    file_name = ""
    if lang == var.LANG_TYPE_ENG:
        file_name = var.FILE_TRAINING_ENG_NOS_PLBL + str(size) + ".obj"
    elif lang == var.LANG_TYPE_MULTI:
        file_name = var.FILE_TRAINING_MUL_NOS_PLBL + str(size) + ".obj"

    return file_name

def get_DF_SLBL_file_path(lang, size):
    file_name = ""
    if lang == var.LANG_TYPE_ENG:
        file_name = var.FILE_TRAINING_ENG_NOS_SLBL + str(size) + ".obj"
    elif lang == var.LANG_TYPE_MULTI:
        file_name = var.FILE_TRAINING_MUL_NOS_SLBL + str(size) + ".obj"

    return file_name

def get_DF_gpt_LBL_file_path(lang, size, labelType):
    if lang == var.LANG_TYPE_ENG:
        file_name = ""
    elif lang == var.LANG_TYPE_MULTI:
        file_name = var.FILE_TRAINING_GPT_MUL_NOS_LBL + str(size) + "_" + labelType + ".obj"
    return file_name

def get_DF_column_name(lang):
    col_name = ""
    if lang == var.LANG_TYPE_ENG:
        col_name = var.COLUMN_NAME_ENG_TXT
    elif lang == var.LANG_TYPE_MULTI:
        col_name = var.COLUMN_NAME_MULTI_LANG_TXT
    return col_name

def record_training(model_id, epoch, training_loss, training_accuracy, val_loss, val_accuracy, duration):

    try:
        conn = sqlHelper.get_mysql_conn()
        mycursor = conn.cursor()
        Select_sql = """
                insert into model_training ( 
                model_name, epoch, elapsed, 
                training_loss, training_accuracy, val_loss, val_accuracy 
                ) values ( 
                %s, %s, %s, %s, %s, 
                %s, %s ) 
                ON DUPLICATE KEY UPDATE 
                    elapsed = %s, training_loss = %s, training_accuracy = %s, val_loss = %s, val_accuracy = %s
                """
        val = (model_id, epoch, duration, training_loss, training_accuracy, val_loss, val_accuracy, duration, training_loss, training_accuracy, val_loss, val_accuracy)
        mycursor.execute(Select_sql, val)
        conn.commit()

    except mysql.connector.Error as error:
        print("Failed to select record to database: {}".format(error))
    finally:
        if conn.is_connected():
            mycursor.close()
            conn.close()
            print("MySQL connection is closed")


def Main_trainingModel_batch(modelType, tokenizer, model, texts, labels, BATCH_SIZE, LEARNING_RATE, start_epochs, end_epoches):
    print(start_epochs)
    print(end_epoches)

    # For grahic
    os.environ["PATH"] += os.pathsep + '/opt/homebrew/bin/'

    # Print model architecture and details
    print("Model architecture:")
    print(model)

    # Print model configuration
    print("\nModel configuration:")
    print(model.config)

    # Tokenize input texts
    tokenized_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    # Build dataset and dataloader
    input_ids = tokenized_inputs["input_ids"]
    attention_mask = tokenized_inputs["attention_mask"]
    labels = torch.tensor(labels)

    dataset = TensorDataset(input_ids, attention_mask, labels)

    train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Build and train the model
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # Use CrossEntropyLoss for multi-class classification
    criterion = torch.nn.CrossEntropyLoss()

    # Move model to GPU if available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # data for monitoring of the training process
    epochsList = []
    train_accuracyList = []
    val_accuracyList = []
    train_lossList = []
    val_lossList = []

    # From check point
    prev_epoch = start_epochs -1
    start_state_file = var.DIR_OUTPUT + modelType + '_checkpoint_' + str(prev_epoch) + '_model.pth'
    if os.path.isfile(start_state_file):
        checkpoint = torch.load(start_state_file)

        epochsList = checkpoint['epochsList']
        train_accuracyList = checkpoint['train_accuracyList']
        val_accuracyList = checkpoint['val_accuracyList']
        train_lossList = checkpoint['train_lossList']
        val_lossList = checkpoint['val_lossList']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        avg_loss = checkpoint['loss']
        accuracy = checkpoint['accuracy']
        avg_loss_val = checkpoint['avg_loss_val']
        accuracy_val = checkpoint['accuracy_val']
        print('Done: loading check point')

    for epoch in range(start_epochs, end_epoches + 1):
        start_time = time.monotonic()  # Record the start time for the epoch
        model.train()
        total_correct = 0
        total_samples = 0
        total_loss = 0.0

        for batch in train_dataloader:
            batch = tuple(t.to(device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
            labels = batch[2].to(device)
            outputs = model(**inputs)
            logits = outputs.logits
            loss = criterion(logits, labels)
            total_loss += loss.item()

            _, predicted_labels = torch.max(logits, 1)
            total_correct += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

            optimizer.zero_grad()
            loss.backward()

            # Applying dropout manually
            # ---------
            # Use the default dropout_rate = 0.1
            # dropout_rate = 0.2
            # for name, param in model.named_parameters():
            #     if 'dropout' in name:
            #         param.grad = param.grad * (1 - dropout_rate)
            # ---------
            optimizer.step()

        accuracy = total_correct / total_samples
        avg_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch}/{end_epoches}, Training Loss: {avg_loss:.4f}, Training Accuracy: {accuracy:.4f}')

        # Validation
        model.eval()
        total_correct_val = 0
        total_samples_val = 0
        total_loss_val = 0.0

        with torch.no_grad():
            for batch in val_dataloader:
                batch = tuple(t.to(device) for t in batch)
                inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
                labels = batch[2].to(device)
                outputs = model(**inputs)
                logits = outputs.logits
                loss = criterion(logits, labels)
                total_loss_val += loss.item()

                _, predicted_labels_val = torch.max(logits, 1)
                total_correct_val += (predicted_labels_val == labels).sum().item()
                total_samples_val += labels.size(0)

                batch_probs = F.softmax(logits, dim=1)

        accuracy_val = total_correct_val / total_samples_val
        avg_loss_val = total_loss_val / len(val_dataloader)

        # Calculate and print the time spent
        end_time = time.monotonic()
        epoch_time = end_time - start_time

        print(
            f'Epoch {epoch}/{end_epoches}, Time: {epoch_time:.2f}s, Validation Loss: {avg_loss_val:.4f}, Validation Accuracy: {accuracy_val:.4f}')

        # Keeping training info
        epochsList.append(epoch)
        train_accuracyList.append(accuracy)
        val_accuracyList.append(accuracy_val)
        train_lossList.append(avg_loss)
        val_lossList.append(avg_loss_val)

        # save the info into database
        record_training(modelType, epoch, avg_loss, accuracy, avg_loss_val, accuracy_val, epoch_time)

        ## -----
        torch.save(
            {
                'epoch': epoch,

                'epochsList': epochsList,
                'train_accuracyList': train_accuracyList,
                'val_accuracyList': val_accuracyList,
                'train_lossList': train_lossList,
                'val_lossList': val_lossList,

                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),

                'loss': avg_loss,
                'accuracy': accuracy,
                'avg_loss_val': avg_loss_val,
                'accuracy_val': accuracy_val,
            },
            var.DIR_OUTPUT + modelType + '_checkpoint_' + str(epoch) + '_model.pth'
        )
        prev_file = var.DIR_OUTPUT + modelType + '_checkpoint_' + str(epoch - 1) + '_model.pth'
        if os.path.exists(prev_file):
            os.remove(prev_file)
            print(f"The file {prev_file} has been deleted.")

        ## -----
        print("Done: save check point")

    # Save the trained model if needed
    torch.save(model.state_dict(), var.DIR_OUTPUT + modelType + '_model.pth')

    # Create a graph of the computation
    inputs = None
    for batch in val_dataloader:
        batch = tuple(t.to(device) for t in batch)
        inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
        break
    outputs = model(**inputs)
    graph = make_dot(outputs.logits, params=dict(model.named_parameters()))

    # Save the graph as an image (PNG)
    graph.render(filename=var.DIR_OUTPUT + modelType + '_model', format='png', cleanup=True)

    # Save the trained model if needed
    torch.save(model.state_dict(), var.DIR_OUTPUT + modelType + '_model.pth')

    # Generate graph
    # Plotting accuracy
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochsList, train_accuracyList, label='Training Accuracy', marker='o')
    plt.plot(epochsList, val_accuracyList, label='Validation Accuracy', marker='o')
    plt.title('Epoch vs Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plotting average loss
    plt.subplot(1, 2, 2)
    plt.plot(epochsList, train_lossList, label='Training Loss', marker='o')
    plt.plot(epochsList, val_lossList, label='Validation Loss', marker='o')
    plt.title('Epoch vs Average Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.legend()

    plt.tight_layout()

    # Save the figure to an image file (e.g., PNG)
    plt.savefig(var.DIR_OUTPUT + modelType + '_training_graph.png')

    # Show the plot (optional)
    # plt.show()

