"""
Functions for testing Hugginface's GPT-2 trained model
"""
import json
from collections import defaultdict, Counter
from numpy import exp, log
import torch
import torch.nn.functional as F
from transformers import AutoConfig
from transformers import AutoTokenizer, GPT2Tokenizer
from transformers import AutoModelWithLMHead

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')

def load_model():
    """Creates a model and loads in weights for it."""
    config = AutoConfig.from_pretrained('gpt2-medium')

    model = AutoModelWithLMHead.from_pretrained(
        f'/content/drive/My Drive/irony_checkpoints/gpt2_medium_joke-3.pt',
        config=config
    )

    return model

model = load_model()


def predict(test_convs,
            tokenizer=tokenizer,
            model=model,
            device="cuda",
            num_top_softmax=1,
            json_file_out='add_stats_output.jsonl',
        ):
    """
    Function for model evaluation. saves and returns `Conversation` objects
    with model statistics (top SoftMax values for each conversation).
    """
    model.to(device)
    model.eval()
    answers = []
    prob_answers = []
    prompt = "\n--\nQ: Is this a joke or said ironically?\nA:"

    for test_conv in test_convs:
        tweet_template = test_conv+prompt
        input_ids = tokenizer.encode(tweet_template)
        input_tensor = torch.LongTensor(input_ids).to(device)
        with torch.no_grad():
            output = model(input_tensor,labels=input_tensor,return_dict=True)
        logits = output.logits
        logits = logits[...,-1,:]
        predicted_prob = F.softmax(logits, dim=-1)
        top_softmax = _top_softmax(predicted_prob, tokenizer, num_top_softmax)
        answers.append(top_softmax[0][0])
        prob_answers.append(top_softmax[0][1])
        
    with open(json_file_out, "w") as dump:
        json.dump({"text": text_convs, "model_responses": answers, "logits_answers": prob_answers}, dump, indent=4)
    return [test_convs, answers, prob_answers]

def _top_softmax(prob_dict, tokenizer, num_tokens):
    if num_tokens==1:
        max_index = torch.argmax(prob_dict)
        return [(tokenizer.decode([max_index]), str(prob_dict[max_index].item()))]
    num_tokens = min(len(prob_dict), num_tokens)
    _, sorted_indices = torch.sort(prob_dict[:], descending=True)
    sorted_indices = list(sorted_indices.cpu().numpy())
    return [(tokenizer.decode([index]), str(prob_dict[index].item())) for index in sorted_indices[:num_tokens]]

#log prob(token_j)  = logit_j - log(sum_k exp(logit_k))
def _log_prob(logits_tensor, labels_lst, num_tokens, tokenizer):
    _, sorted_indices = torch.sort(logits_tensor[:], descending=True)
    sorted_indices = list(sorted_indices.detach().numpy())
    logits = logits_tensor.detach.numpy()
    log_probs = list(logits - log(sum(exp(logits))))
    #converting logits into log probs shouldn't change indices...
    return [(tokenizer.decode([index]), str(log_probs[index].item())) for index in sorted_indices[:num_tokens]]
