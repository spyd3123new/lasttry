import pandas as pd
import numpy as np
import torch
from tqdm.notebook import tqdm
import os
from transformers import BertTokenizer
from torch.utils.data import TensorDataset

from transformers import BertForSequenceClassification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
import random

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
import pickle






def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')

def accuracy_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return accuracy_score(labels_flat, preds_flat)

def precision_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return precision_score(labels_flat, preds_flat, average = 'weighted')

def recall_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return recall_score(labels_flat, preds_flat, average = 'weighted')

def accuracy_per_class(preds, labels):
    label_dict_inverse = {v: k for k, v in label_dict.items()}

    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')


def evaluate(dataloader_eval,model_name):

    model_name.eval()

    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in dataloader_eval:

        batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():
            outputs = train_model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total/len(dataloader_test)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return loss_val_avg, predictions, true_vals

dataset = 'Yoga_Dataset_Preprocessed.csv'

df_dataset = pd.read_csv(dataset)

unique_labels = df_dataset.Asan.unique()

label_dict = {}
for index, unique_label in enumerate(unique_labels):
    label_dict[unique_label] = index
label_dict
df_dataset['label'] = df_dataset.Asan.replace(label_dict)
df_dataset["text"] = df_dataset["preprocessed_benefit1"] + ' ' + df_dataset["preprocessed_benefit2"]
df_dataset.head()


train_set, test_set, train_labels, test_labels = train_test_split(df_dataset.index.values, df_dataset.label.values, test_size=0.15,                      random_state=42,
                                                  stratify=df_dataset.label.values)


df_dataset['data_type'] = ['not_set']*df_dataset.shape[0]

df_dataset.loc[train_set, 'data_type'] = 'train'
df_dataset.loc[test_set, 'data_type'] = 'test'
df_dataset.groupby(['Asan', 'label', 'data_type']).count()


# tokenise train and test data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

encoded_data_train = tokenizer.batch_encode_plus(
    df_dataset[df_dataset.data_type=='train'].text.values,
    add_special_tokens=True,
    return_attention_mask=True,
    pad_to_max_length=True,
    max_length=80,
    return_tensors='pt'
)

encoded_data_test = tokenizer.batch_encode_plus(
    df_dataset[df_dataset.data_type=='test'].text.values,
    add_special_tokens=True,
    return_attention_mask=True,
    pad_to_max_length=True,
    max_length=80,
    return_tensors='pt'
)


input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(df_dataset[df_dataset.data_type=='train'].label.values)

input_ids_test = encoded_data_test['input_ids']
attention_masks_test = encoded_data_test['attention_mask']
labels_test = torch.tensor(df_dataset[df_dataset.data_type=='test'].label.values)

dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)

train_model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)



batch_size = 4
epochs = 6
learning_rate = 2e-5

dataloader_train = DataLoader(dataset_train, sampler=RandomSampler(dataset_train), batch_size=batch_size)
dataloader_test = DataLoader(dataset_test, sampler=SequentialSampler(dataset_test), batch_size=batch_size)

optimizer = AdamW(train_model.parameters(),lr=learning_rate,eps=1e-8)

# Define the learning rate scheduler
num_training_steps = len(dataloader_train) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

#scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,num_training_steps=len(dataloader_train)*epochs)

#training_steps = range(len(learning_rate_values))

seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_model.to(device)


#print("starting loop")

# Create a list to store the learning rates at each step
save_directory2 = 'data_volume'
if not os.path.exists(save_directory2):
        
    learning_rates_values = []
    training_loss_values = []
    validation_loss_values = []
    train_acc_values = []
    test_acc_values = []

    for epoch in tqdm(range(1, epochs+1)):

        train_model.train()
        #print("model trained")
        loss_train_total = 0

        progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
        for batch in progress_bar:
            #print("in batch loop")
            train_model.zero_grad()

            batch = tuple(b.to(device) for b in batch)

            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'labels':         batch[2],
                     }

            outputs = train_model(**inputs)

            loss = outputs[0]
            loss_train_total += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(train_model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            #current_learning_rate = optimizer.learning_rate.numpy()  # Modify this based on your optimizer
            current_learning_rate = optimizer.param_groups[0]['lr']  # Get the current learning rate
            learning_rates_values.append(current_learning_rate)


            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})


        # Define the directory where you want to save the model
        save_directory = 'data_volume'

        # Create the directory if it doesn't exist
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save the model
        torch.save(train_model.state_dict(), f'{save_directory}/finetuned_BERT_epoch_{epoch}.model')
        #torch.save(train_model.state_dict(), f'data_volume/finetuned_BERT_epoch_{epoch}.model')
        #print("saved model")
        tqdm.write(f'\nEpoch {epoch}')

        loss_train_avg = loss_train_total/len(dataloader_train)
        tqdm.write(f'Training loss: {loss_train_avg}')

        val_loss, predictions, true_vals = evaluate(dataloader_test,train_model)
        #val_f1 = f1_score_func(predictions, true_vals)
        test_acc = accuracy_score_func(predictions, true_vals)
        tqdm.write(f'Validation loss: {val_loss}')

        print("Accuracy : ", accuracy_score_func(predictions, true_vals))
        print("F1 score :", f1_score_func(predictions, true_vals))
        print("Precision :", precision_score_func(predictions, true_vals))
        print("Recall score :", recall_score_func(predictions, true_vals))
        #print("specificity :", specificity_score_func(predictions, true_vals))
        #print("Accuracy per class : " , accuracy_per_class(predictions, true_vals))

        training_loss_values.append(loss_train_avg)
        validation_loss_values.append(val_loss)

        t_loss, predictions, true_vals = evaluate(dataloader_train,train_model)
        train_acc = accuracy_score_func(predictions, true_vals)

        train_acc_values.append(train_acc)
        test_acc_values.append(test_acc)




# Assuming true_vals is a 1D array or list of class labels
    true_vals = np.array(true_vals).reshape(-1, 1)  # Reshape to a 2D array

    # Create an instance of the OneHotEncoder
    encoder = OneHotEncoder(sparse=False)  # sparse=False to get a dense array

    # Fit and transform the encoder on the data
    y_true = encoder.fit_transform(true_vals)
    print(y_true)

    # Assuming y_true and y_scores are correctly formatted for multiclass classification
    #y_true = np.array(true_vals)  # Ground truth labels
    y_scores = np.array(predictions)  # Predicted class probabilities

    n_classes = len(label_dict)  # Number of classes based on the shape of the arrays

    # Initialize empty lists to store results for each class
    fpr = []
    tpr = []
    roc_auc = []

    # Calculate ROC curve and AUC for each class (OvR approach)
    for i in range(n_classes):
        fpr_i, tpr_i, _ = roc_curve(y_true[:, i], y_scores[:, i])
        roc_auc_i = roc_auc_score(y_true[:, i], y_scores[:, i])

        fpr.append(fpr_i)
        tpr.append(tpr_i)
        roc_auc.append(roc_auc_i)

    # Plot the ROC curves for each class
    plt.figure(figsize=(8, 6))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC Curve (One-vs-Rest)')
    plt.legend(loc='lower right')
    plt.show()


    plt.title('Learning rate')
    plt.plot(learning_rates_values)

    plt.title('Loss')
    plt.plot(training_loss_values, label='train')
    plt.plot(validation_loss_values, label='test')
    plt.legend()
    plt.show()

    plt.title('Accuracy')
    plt.plot(train_acc_values, label='train')
    plt.plot(test_acc_values, label='test')
    plt.legend()
    plt.show()




with open('label_dict.pkl', 'wb') as f:
    pickle.dump(label_dict, f)

predictions=[]
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
chk_model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)

#chk_model.to(device)
print()
print()
input_text = input("Please specifiy the medical condition for which yoga recommendation is being sought: ")
#print("You entered:", user_input)

#input_text ="diabetes"
chk_model.load_state_dict(torch.load('data_volume/finetuned_BERT_epoch_6.model', map_location=torch.device('cpu')))
input_ids = tokenizer.batch_encode_plus(
    input_text,
    add_special_tokens=True,
    return_attention_mask=True,
    pad_to_max_length=True,
    max_length=80,
    return_tensors='pt')

# Ensure input_ids is on the same device as the model (CPU or GPU)
input_ids = input_ids.to(chk_model.device)

chk_model.eval()

# Make the prediction
with torch.no_grad():
    outputs = chk_model(**input_ids)

logits = outputs.logits
logits = logits.detach().cpu().numpy()
predictions.append(logits)
#print(predictions)


preds_flat = np.argmax(predictions, axis=1).flatten()
#labels_flat = labels.flatten()
#print(preds_flat)

# Initialize a list to store keys that correspond to the search value
matching_keys = []
for  label in preds_flat:
  search_value = label
  #print("searching for:", label)
  # Iterate through the dictionary to find keys that match the search value
  for key, value in label_dict.items():
    if np.array_equal(value, search_value):
        if key not in matching_keys:
           matching_keys.append(key)






# Check if any keys were found
if matching_keys:
    print(f"For {input_text} recommended asans are : {', '.join(matching_keys)}")
else:
    print(f"No keys found for the value {search_value}")



