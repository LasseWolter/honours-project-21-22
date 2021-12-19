# Gilick et al. model

### Missing packages in requirements.txt

- pandas
- tqdm
- nltk

### Setup

- running the python script on CPU by using `map_location=torch.device('cpu')` yields different results
  - possibly due to issue mentioned here:
    - https://github.com/open-mmlab/mmdetection/issues/1077
      > Since the custom ops are compiled with CUDA support, they do not work on the CPU-only environment. We will consider adding support for that.

### Model training

- the `--include-words' option seems to be like the laughter next to speech option in our evaluation
  - comments state that they didn't use this for the paper which lets me conclude that they also looked for 'laughter-only' snippets and only used them for training
- they use subsamples of the laughter segments instead of the whole regions - WHY?

  - they store a subsample with the segment itself - this is the default subsample used for validation
  - during training the subsampling happens on regularly to get more variety

- `featurize_melspec()` if used to turn audio file into features

  - optional offset/duration can be passed

- loads all training data in memory upfront using a pre-created hashtable that maps from file-locations to pre-loaded audio files in bytes

### Evaluation

- why do they use bootstrap_metrics()-function and only use a sample of their output values?

# General

### File Format

- `.TextGrid` files
  - created by Praat: Praat is a free computer software package for speech analysis in phonetics.
- can be parsed using this python library
  - praat-textgrids: https://pypi.org/project/praat-textgrids/

### Preprocessing ICSI

- wrong/questionable laughter transcriptions
  -> after listening to a few laughter samples manually
- possibly exclude audio where there was no transcription available for that channel
  - like Knox and Mirghafori

### Evaluation

- check for possible time-shift of transcript and audio

  - there is a note about this in the ICSI notes
    - http://www1.icsi.berkeley.edu/Speech/mr/icsimc_doc/overview.txt

- think about difference between creating frames from audio (samples) and timestamps
  - in our evaluation we get frames from timestamps not from samples
    - thus, dividing by 100 to get 10ms frames is correct.
  - if you get frames from samples you need to consider the sample rate
    - e.g. 41 000hz -> 41 000 samples per second
