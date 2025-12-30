import re


def korean_cleaners(text):
    text = re.sub(r'[^\w\s,.!?가-힣]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def collapse_whitespace(text):
    return re.sub(r'\s+', ' ', text)


def remove_special_characters(text):
    return re.sub(r'[^\w\s가-힣]', '', text)


def normalize_text(text, cleaners=["korean_cleaners"]):
    for cleaner_name in cleaners:
        if cleaner_name == "korean_cleaners":
            text = korean_cleaners(text)
        elif cleaner_name == "collapse_whitespace":
            text = collapse_whitespace(text)
        elif cleaner_name == "remove_special_characters":
            text = remove_special_characters(text)
    return text
