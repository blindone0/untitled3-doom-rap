# -*- coding: utf-8 -*-
"""Syllable splitting for the lyric tools (flow.py, static_lyrics.py, static_vocal.py): Ukrainian and English."""
import re

VOW_UK = set("аеєиіїоуюяАЕЄИІЇОУЮЯ")


def syllabify_uk(w):
    """char spans of syllables inside a word (CV-based, good enough for karaoke)"""
    idx = [i for i, c in enumerate(w) if c in VOW_UK]
    if not idx:
        return []
    spans, start = [], 0
    for k, vi in enumerate(idx):
        if k == len(idx) - 1:
            end = len(w)
        else:
            gap = idx[k + 1] - vi - 1
            end = vi + 1 if gap <= 1 else vi + 2
        spans.append((start, end))
        start = end
    return spans


def syllabify_en(w):
    """English: vowel groups are nuclei; silent final -e / -ed / -es dropped; consonants split between nuclei"""
    lw = w.lower()
    core = re.sub(r"[^a-z']+$", "", lw)           # strip trailing punctuation
    core = re.sub(r"'s$", "", core)
    n = len(core)
    is_v = [(c in "aeiou") or (c == "y" and i > 0) for i, c in enumerate(core)]
    nuclei, i = [], 0
    while i < n:
        if is_v[i]:
            j = i
            while j + 1 < n and is_v[j + 1]:
                j += 1
            nuclei.append([i, j])
            i = j + 1
        else:
            i += 1
    if len(nuclei) > 1:
        a, b = nuclei[-1]
        if a == b == n - 1 and core[-1] == "e" and not is_v[n - 2]:
            if not (core.endswith("le") and n >= 3 and not is_v[n - 3]):
                nuclei.pop()                            # silent -e (stone, home) but not -ble/-tle
        elif a == b == n - 2 and core.endswith("ed") and core[-3] not in "td" and not is_v[n - 3]:
            nuclei.pop()                                # stopped, buried
        elif a == b == n - 2 and core.endswith("es") and core[-3] not in "sxz" and not core.endswith(("shes", "ches")):
            nuclei.pop()                                # trees, clothes
    if not nuclei:
        return []
    spans, start = [], 0
    for k, (a, b) in enumerate(nuclei):
        if k == len(nuclei) - 1:
            end = len(w)
        else:
            cons = core[b + 1:nuclei[k + 1][0]]
            if len(cons) <= 1 or cons.startswith(("th", "ch", "sh", "ph", "wh", "tch")):
                end = b + 1                                 # no-thing, wea-ther, wa-tching
            else:
                end = b + 2                                 # spea-kers -> speak-ers style split after one consonant
        spans.append((start, end))
        start = end
    return spans


def syllabify_word(w, en=True):
    return syllabify_en(w) if en else syllabify_uk(w)


def word_re(en=True):
    return r"[^\s-]+" if en else r"\S+"          # English: hyphenated compounds count as two words


def syllables_of_line(text, en=True):
    """list of (char_start, char_end, word_index, first_in_word) for the sung part (stage directions in () skipped)"""
    out, wi, pending = [], 0, None  # pending = start of a vowel-less word (з, в, й ...) glued to the next syllable
    for m in re.finditer(word_re(en), text):
        tok = m.group(0)
        if tok.startswith("(") or tok.endswith(")"):
            continue
        spans = syllabify_word(tok, en)
        if not spans:
            if pending is None and any(c.isalpha() for c in tok):
                pending = m.start()
            continue
        for j, (a, b) in enumerate(spans):
            start = m.start() + a
            if j == 0 and pending is not None:
                start = pending
                pending = None
            out.append((start, m.start() + b, wi, j == 0))
        wi += 1
    return out


def count(text, en=True):
    return len(syllables_of_line(text, en))


def words_of_line(text, en=True):
    """[(word as it should be spoken, syllable count)], indexed like the word numbers in syllables_of_line"""
    out, pending = [], ""
    for m in re.finditer(word_re(en), text):
        tok = m.group(0)
        if tok.startswith("(") or tok.endswith(")"):
            continue
        spans = syllabify_word(tok, en)
        if not spans:
            if not pending and any(c.isalpha() for c in tok):
                pending = tok
            continue
        spoken = re.sub(r"^[^\w']+|[^\w']+$", "", (pending + " " + tok) if pending else tok)
        out.append((spoken, len(spans)))
        pending = ""
    return out
