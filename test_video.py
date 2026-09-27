import numpy as np

from models.video import caption_segments, group_subtitles, render_caption

words = [
    {"word": "born", "start": 0.0, "end": 0.5},
    {"word": "in", "start": 0.6, "end": 0.7},
    {"word": "1964,", "score": 0.1},  # whisperx: unaligned, no timestamps
    {"word": "not", "start": 1.0, "end": 1.2},
    {"word": "42"},
]

groups = group_subtitles(words)
# untimed "1964," kept in text; trailing group with no timed word is dropped
assert [g["word"] for g in groups] == ["born in 1964, not"], groups
assert (groups[0]["start"], groups[0]["end"]) == (0.0, 1.2)
assert group_subtitles([]) == []

# each timed word lit until the next timed word starts; "1964," (idx 2) never lit
g = groups[0]
assert caption_segments(g["words"], g["end"]) == [
    (0, 0.0, 0.6),
    (1, 0.6, 1.0),
    (3, 1.0, 1.2),
], caption_segments(g["words"], g["end"])


def yellow_pixels(img):
    r, gr, b, a = img[..., 0], img[..., 1], img[..., 2], img[..., 3]
    return ((r > 200) & (gr > 180) & (b < 60) & (a == 255)).sum()


texts = ["Welcome", "to", "the", "channel,"]
lit = render_caption(texts, 3, 380)
assert lit.ndim == 3 and lit.shape[2] == 4 and lit.shape[1] <= 380, lit.shape
assert yellow_pixels(lit) > 50
assert yellow_pixels(render_caption(texts, None, 380)) == 0
# narrow width wraps onto more lines
assert render_caption(texts, 0, 200).shape[0] > lit.shape[0]

print("ok")
