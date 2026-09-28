# mobicrowd/emoji_codec.py
from typing import Optional
from emoji import demojize, emojize

def encode_for_db(text: Optional[str]) -> str:
    if not text:
        return ""
    # turn emoji into :aliases: that are 7-bit safe for utf8 (3-byte)
    return demojize(text, language='alias')

def decode_for_api(text: Optional[str]) -> str:
    if not text:
        return ""
    # turn :aliases: back into real emoji for API/WS clients
    return emojize(text, language='alias')
