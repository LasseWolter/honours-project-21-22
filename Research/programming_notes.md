### Setup

- running the python script on CPU by using `map_location=torch.device('cpu')` yields different results
  - possibly due to issue mentioned here:
    - https://github.com/open-mmlab/mmdetection/issues/1077
      > Since the custom ops are compiled with CUDA support, they do not work on the CPU-only environment. We will consider adding support for that.

### Evaluation

- check for possible time-shift of transcript and audio
  - there is a note about this in the ICSI notes
    - http://www1.icsi.berkeley.edu/Speech/mr/icsimc_doc/overview.txt

### File Format

- `.TextGrid` files
  - created by Praat: Praat is a free computer software package for speech analysis in phonetics.
- can be parsed using this python library
  - praat-textgrids: https://pypi.org/project/praat-textgrids/
