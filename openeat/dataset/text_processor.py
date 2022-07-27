import re
def _tokenizer(text, sp, char_dict):
    tokens = []
    pattern = re.compile(r'([\u4e00-\u9fff])')
    # Example:
    #   txt   = "你好 ITS'S OKAY 的"
    #   chars = ["你", "好", " ITS'S OKAY ", "的"]
    chars = pattern.split(text.upper())
    mix_chars = [w for w in chars if len(w.strip()) > 0]
    for ch_or_w in mix_chars:
        # ch_or_w is a single CJK charater(i.e., "你"), do nothing.
        if pattern.fullmatch(ch_or_w) is not None:
            tokens.append(ch_or_w)
        # ch_or_w contains non-CJK charaters(i.e., " IT'S OKAY "),
        # encode ch_or_w using bpe_model.
        else:
            if sp:
                for p in sp.encode_as_pieces(ch_or_w):
                    tokens.append(p)
            else:
                tokens.append(ch_or_w)
    return tokens

def preprocess_Chinese(text):
    from zhon.hanzi import punctuation 
    text = re.sub(r'[{}]+'.format(punctuation),'',text)
    return text
def preprocess_English(text):
    from string import punctuation
    punctuation = punctuation.replace('\'','') # keep "I'm" invariant
    text = re.sub(r'[{}]+'.format(punctuation),'',text)
    return text
def _remove_punctuation(text):
    text = preprocess_Chinese(text)
    text = preprocess_English(text)
    text = text.replace('\\','')
    return text

