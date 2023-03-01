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
        f'./models/gpt2_medium_joke_50bs-8.pt',
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
    Function for model evaluation. Returns text, the model's guess and the probability thereof.
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
        predicted_prob = _labels_only_logits(logits, [" Yes", " No"], tokenizer)
        top_softmax = _top_softmax(predicted_prob, tokenizer, num_top_softmax)
        answers.append(top_softmax[0][0])
        prob_answers.append(str(round(float(top_softmax[0][1]), 3)))


    return [test_convs, answers, prob_answers]

def _top_softmax(prob_dict, tokenizer, num_tokens):
    if num_tokens==1:
        max_index = torch.argmax(prob_dict)
        return [(tokenizer.decode([max_index]), str(prob_dict[max_index].item()))]
    num_tokens = min(len(prob_dict), num_tokens)
    _, sorted_indices = torch.sort(prob_dict[:], descending=True)
    sorted_indices = list(sorted_indices.cpu().numpy())
    return [(tokenizer.decode([index]), str(prob_dict[index].item())) for index in sorted_indices[:num_tokens]]

def _labels_only_logits(logits, labels, tokenizer):
    _logits = logits[:]
    tokenized_labels = [tokenizer.encode(label)[0] for label in labels]
    filter_out = [index for index in range(len(logits)) if index not in tokenized_labels]
    _logits[...,filter_out] = -10**8
    return F.softmax(_logits, dim=-1)

def _batched_logsoftmax(matrix: torch.Tensor):
    """Compute log(softmax(row)) for each row of the matrix.

    matrix: shape (batch, n)

    Return: (batch, n). For each i, out[i] should sum to 1.

    Do this without using PyTorch's logsoftmax function.
    For each row, subtract the maximum first to avoid overflow if the row contains large values.
    """
    maxes, _ = torch.max(matrix, dim=-1, keepdim=True)
    sum_exp = maxes + torch.log(torch.sum(torch.exp(matrix - maxes), dim=-1, keepdim=True))
    return matrix - sum_exp
