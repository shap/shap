import torch
import numpy as np
import scipy as sp

def get_encoder_outputs(input_ids,model,attention_mask):
    encoder = model.get_encoder()
    with torch.no_grad():
        encoder_outputs = encoder(input_ids, attention_mask=attention_mask, return_dict=True)
    del encoder
    return encoder_outputs

def get_conditional_logits(model, decoder_input, past, attention_mask, encoder_outputs):
    model_inputs = model.prepare_inputs_for_generation(
                    decoder_input, past=past, attention_mask=attention_mask, use_cache=True, encoder_outputs=encoder_outputs
                )
    with torch.no_grad():
        outputs = model(**model_inputs, return_dict=True)
    if "past_key_values" in outputs:
        past = outputs.past_key_values
    next_token_logits = outputs.logits[:, -1, :].detach().cpu()
    probs=next_token_logits[0].softmax(dim=0)
    del model_inputs, outputs
    logits = sp.special.logit(probs)
    return logits, past

def cal_conditional_logits(input_ids,model,tokenizer,decoder_inputs,encoder_outputs=None,attention_mask=None):
    conditional_logits=[]
    past=None
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input_ids=input_ids.to(device)
    if attention_mask is None:
        attention_mask = (input_ids!=tokenizer.pad_token_id).type(torch.int64)
    else:
        attention_mask = attention_mask.to(device)
    if encoder_outputs is None:
        encoder_outputs = get_encoder_outputs(input_ids,model,attention_mask)
    del input_ids
    for i in range(1,decoder_inputs.shape[1]):
        probs, past = get_conditional_logits(model, decoder_inputs[:,:i], past, attention_mask, encoder_outputs)
        conditional_logits.append(probs[decoder_inputs[0,i].item()])
    del past, attention_mask, encoder_outputs
    return np.array(conditional_logits)