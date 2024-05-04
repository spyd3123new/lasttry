import torch
from transformers import BertForSequenceClassification, BertTokenizer
import numpy as np
import pickle


def predictionsz(inputstring):
  
  with open('label_dict.pkl', 'rb') as f:
      label_dict = pickle.load(f)

  predictions=[]
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
  chk_model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                        num_labels=len(label_dict),
                                                        output_attentions=False,
                                                        output_hidden_states=False)

  #chk_model.to(device)
  print()
  print()
  # input_text = input("Please specifiy the medical condition for which yoga recommendation is being sought: ")
  input_text=inputstring
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
  
  
  # print(matching_keys)

  # if matching_keys:
  #     print(f"For {input_text} recommended asans are : {', '.join(matching_keys)}")
  # else:
  #     print(f"No keys found for the value {search_value}")
  
  return matching_keys


# print(predictionsz("dengu"))