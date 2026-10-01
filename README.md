# LyricLift

**Song production recommendations in under 2.5 seconds.**

LyricLift is a text analytics tool that reads original song lyrics and recommends production settings (danceability, loudness, acousticness, energy) based on 10,000+ hit songs from 2000 to 2019.

### Live page

**[kbansal4.github.io/lyriclift](https://kbansal4.github.io/lyriclift)** - full project write-up: the problem, how the model works, a walkthrough using The Weeknd's "Blinding Lights," and model comparisons.

### How it works

1. **Dataset** - 10,000+ songs (2000-2019) with lyrics and four production metrics per track.
2. **Sentiment check** - VADER sentiment analysis scores lyrics on a five-point scale (very positive to very negative), with hand-tuned thresholds.
3. **Find similar songs** - a Word2Vec model trained on the full lyric corpus embeds every song; cosine similarity surfaces the 5 closest songs with matching sentiment.
4. **Recommend the sound** - averages the production metrics of those 5 songs and translates the numbers into concrete production advice.

Word2Vec was picked over TF-IDF (too shallow for lyrics) and BERT (accurate but too slow) after benchmarking all three.

### Files

- `index.html` - the project page
- `Textw2v.py` - the final model code
- `LyricLift-Deck.pptx` - project slide deck

### Credits

By Kush Bansal and team - Text Analytics course, NC State Institute for Advanced Analytics, Fall 2024.
