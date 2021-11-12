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
