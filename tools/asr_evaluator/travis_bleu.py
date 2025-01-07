import json
import editdistance
import string
import argparse
import time 
import re
import html



from pathlib import Path
from sacrebleu.metrics import BLEU

OUTPUT="scores.tsv"

PATH="/home/tbartley/Documents/speech_translation/output/evaluation/manifests/"
SUFFIX="_clean"

GOLD_TEXT_FIELD="answer"
PRED_TEXT_FIELD="prediction"
COMET_TEXT_FIELD="text"

COMET_MODEL="Unbabel/wmt22-comet-da"
BLEURT_MODEL="../bleurt/bleurt/BLEURT-20"

NUM=[str(n) for n in range(10)]


def clean(input_string, ngram=0):
    # # Removes beginng/end of sentence tokens
    output = re.sub(r'(<s>|</s>)', r'', input_string)
    # # Removes unknown characters
    output = re.sub(r'(<unk>)', r'⁇', output)
    ## Remove artifacts
    output = re.sub(r'\[filler\/\]', r'', output)
    output = re.sub(r'«', r'', output)
    output = re.sub(r'»', r'', output)
    output = re.sub(r'„', r'', output)
    output = re.sub(r'"', r'', output)
    output = re.sub(r'(\s+)', r' ', output)
    # # Removes leading spaces before punctuation
    output = re.sub(r"(\s+)([\.\,\?\-:])", r'\2', output)
    output = re.sub(r'\.+', ".", output)
    # # Replaces space at beginning and end of sentence
    output = re.sub(r'(^\s+|\s+$)', r'', output)

    # Add hyphen for standardization
    #output = output.replace("ß", "ss")
    nc = output.lower()
    return output



def clean_ngram(s, ngram=0):     
    s = s.split()
    #Initialize Start and Stop Pointers.
    i=0
    j=0
     
    #Initialize an empty string for new elements.
    new_elements=[]
     
    #Iterate String Using j pointer.
    while(j<len(s)):
         
    #If both elements are same then skip.
        if( s[i]==s[j] ):
            j+=1
             
    #If both elements are not same then append new element.
        elif((s[j]!=s[i]) or (j==len(s)-1)):
            new_elements.append(s[i])
             
    #After appending sliding over the window.
            i=j
            j+=1
             
    #Append the last string.
    new_elements.append(s[j-1])
    return " ".join(new_elements)

#assert False
def main(preds, dataset, tokenizer=None, source_language=None, target_language=None, comet="", bleurt=False, normalize=False):
    if comet:
        from comet import download_model, load_from_checkpoint
        model_path = download_model(COMET_MODEL)
        comet_model = load_from_checkpoint(model_path)
    if bleurt:
        from bleurt import score
        bleurt_model = score.BleurtScorer(BLEURT_MODEL)
    if normalize:
        normalizer = Normalizer("cased", lang=target_language[:2])
    manifest = dataset
    samples = {}
    
    num_gold, num_predict, num_norm = 0, 0, 0
    with open(manifest) as source:
        for line in source:
            gold = eval(line)
            #if (source_language is None or target_language is None) or (gold["source_lang"] == source_language and gold["target_lang"] == target_language):
            name = Path(gold["audio_filepath"]).stem
            samples[name] = [None, clean(gold[GOLD_TEXT_FIELD])]
            num_gold += 1
    with open(preds) as source:
        for line in source:
            pred = eval(line)
            name = Path(pred["audio_filepath"]).stem
            print(name)
            if name in samples:
                samples[name][0] = clean(pred[PRED_TEXT_FIELD])
                print(samples[name])
                num_predict += 1
                if normalize:
                    for n in range(10):
                        if str(n) in samples[name][0]:
                            samples[name][0] = normalizer.normalize(samples[name][0])
                            continue
    print(num_predict, num_gold)
    predictions, golds = zip(*samples.values())
    bleu = BLEU(tokenize=tokenizer)
    sb = bleu.corpus_score(predictions, [golds])

    wer_scores, wer_words = 0, 0
    for h, r in zip(predictions, golds):
        wer_words += len(r.split())
        wer_scores += editdistance.eval(h.split(), r.split())
    wer_score = 1.0 * wer_scores / wer_words

    if comet:
        with open(comet) as source:
            for line in source:
                ref = eval(line)
                name = Path(ref["audio_filepath"]).stem
                if name in samples:
                    samples[name].append(clean(ref[COMET_TEXT_FIELD]))
        
        data = []
        for smpl in samples.values():
            #print(smpl)
            data.append(
                {
                    "mt": smpl[0],
                    "ref": smpl[1],
                    "src": smpl[2],
                }
            )
        model_output = comet_model.predict(data, batch_size=128, gpus=1)
        with open(OUTPUT, "w+") as sink:
            for entry, score in zip(data, model_output["scores"]):
                sink.write(entry["ref"] + "\t" + entry["mt"] + "\t" + str(score) + "\n")
        print("Comet score: ", model_output.system_score)

                
    if bleurt:
        model_output = bleurt_model.score(references=golds, candidates=predictions)
        print("BLEURT Score: ", sum(model_output)/len(model_output))
    print(sb, "WER: ", wer_score)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                        prog='ProgramName',
                        description='What the program does',
                        epilog='Text at the bottom of help')
    parser.add_argument("predictions")
    parser.add_argument("dataset")
    parser.add_argument("--source_language")
    parser.add_argument("--target_language")
    parser.add_argument("--comet")
    parser.add_argument("--bleurt", action='store_true')
    parser.add_argument("--tokenizer", choices=BLEU.TOKENIZERS, default=None)
    parser.add_argument("--normalize", action='store_true')
    args = parser.parse_args()
    #assert False
    main(args.predictions, args.dataset, tokenizer=args.tokenizer, source_language=args.source_language, target_language=args.target_language, comet=args.comet, bleurt=args.bleurt, normalize=args.normalize)