import re
def caption_preprocessing(text, remove_digits=True):
    # removw punctuation
    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern,'',text)

    # tokenize
    text=text.split()
    # convert to lower case
    text = [word.lower() for word in text]

    # remove tokens with numbers in them
    text = [word for word in text if word.isalpha()]
    # concat string
    text =  ' '.join(text)

    # insert 'startseq', 'endseq'
    text = 'startseq ' + text + ' endseq'
    return text