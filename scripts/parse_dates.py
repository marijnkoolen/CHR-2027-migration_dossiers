import re
import difflib
from datetime import date

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
    # common non-English / OCR variants seen in the data
    "januar": 1, # German for January
    "januari": 1, # Dutch for January
    "februar": 2, # German for January
    "februari": 2, # Dutch for January
    "juni": 6,   # Dutch for June
    "juli": 7,   # Dutch for June
    
}

MONTH_NAMES = list(MONTHS.keys())

# born, then day, then a "word" that may contain internal whitespace
# (e.g. "Januar y"), then year. Whitespace around each part is optional
# to tolerate missing spaces like "born13 Juni1933".
PATTERN = re.compile(
    r"(born|bron|brorn|bor|bon|bnorn|bnor|bonr|bnonr|bronr)\s*(\d{1,2})\s*([A-Za-z](?:[A-Za-z\s]*[A-Za-z])?)\s*(\d{4})",
    re.IGNORECASE,
)


def resolve_month(raw_month: str):
    token = re.sub(r"\s+", "", raw_month).lower()  # collapse internal whitespace
    if token in MONTHS:
        return MONTHS[token]
    # fuzzy match against known month names to catch OCR typos
    match = difflib.get_close_matches(token, MONTH_NAMES, n=1, cutoff=0.6)
    if match:
        return MONTHS[match[0]]
    return None


def extract_dates(text: str):
    results = []
    for m in PATTERN.finditer(text):
        day_str, month_str, year_str = m.groups()
        month = resolve_month(month_str)
        day = int(day_str)
        year = int(year_str)
        if month is None:
            results.append((m.group(0), None, "unresolved month: %r" % month_str))
            continue
        try:
            d = date(year, month, day)
            results.append((m.group(0), d, None))
        except ValueError as e:
            results.append((m.group(0), None, str(e)))
    return results


if __name__ == "__main__":
    sample = (
        "born 29 july 1942; <NAME>; born 20 july 1943; <NAME>  born 13 JUly 1948; "
        "<NAME> born 16 july 1942; <NAME>  born 22 Janaury 1907; <NAME>  "
        "born 1 Septmeber 1939; <NAME>  born 3 Janaury 1945; <NAME>  "
        "born 8 Juen 1929; <NAME> born13 Juni1933; <NAME> born 19 Sepember 1928; "
        "<NAME>  born 5 Janaury 1950; <NAME>  born 3 Januar y 1949"
    )
    for raw, d, err in extract_dates(sample):
        print(f"{raw!r:45s} -> {d if d else 'FAILED: ' + err}")
