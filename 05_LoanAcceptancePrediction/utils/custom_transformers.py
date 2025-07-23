# utils.py

def categorize_pdays(value):
    if value == -1:
        return 'never'
    elif value <= 100:
        return 'recent'
    else:
        return 'old'
