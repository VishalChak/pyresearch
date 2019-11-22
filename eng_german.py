import torch
#torch.hub.list('pytorch/fairseq')
en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt16.en-de', tokenizer='moses', bpe='subword_nmt')
print(en2de.translate('Tejal is my best buddy'))
