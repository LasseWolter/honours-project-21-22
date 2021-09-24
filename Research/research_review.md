# Reserach Review

### Cnn Architectures For Large-Scale Audio Classification - 01/2017
_source: https://arxiv.org/pdf/1609.09430.pdf_

Quote from intro:
AED = Audio Event Detection
> Historically, AED has been addressed with features such as
> MFCCs and classifiers based on GMMs, HMMs, NMF, or SVMs
> [8, 9, 10, 11]. More recent approaches use some form of DNN,
> including CNNs [12] and RNNs [13].

- using Youtube-100M
- 70M vids - 5.24M hours  
- 30K labels

- whole soundtrack classification
    - assign one keyword to a whole video

---

### The Benefit Of Temporally-Strong Labels In Audio Event Classification
_source: https://arxiv.org/abs/2105.07031_

-uses https://research.google.com/audioset/download_strong.html dataset
- which describes clips of varying length - manually chosen by annotators

---
### Novel approach for detecting applause in continuous meeting speech - 2011
_source: https://ieeexplore-ieee-org.ezproxy.is.ed.ac.uk/document/5941827_
> The proposed approach is based on the short - time autocorrelation
> function features - decay factor, first local minimum 
> and band energy ratio.
- 4 features with thresholds
- compared conventional method using mel frequency cepstral coefficients (MFCC) feature vectors and GMM as classifier
    - 36 feature vector 12 of the 13 MFCC and their first and second derivative

| method | precision rate | recall rate | F1 score |
| -------| ---------------| ------------| ---------|
| proposed        |  94.40% | 90.75% | 92.54% |
| conventional    |  67.47% | 96.13% | 79.29% |

---
### Robust Laughter Detection in Noisy Environments - 09/21 
_source: https://www.isca-speech.org/archive/pdfs/interspeech_2021/gillick21_interspeech.pdf_
- uses new dataset on top of AudioSet corpus
    - with precise segmentations for the start and end points of each laugh
- prior work performs badly in noisy environment
<span style="color:green">
    - How noisy do we expect the input of our domain to be?
        + Each input is separate
        - poor mic quality
        - background noise in the home office
</span>
- AudioSet only specifies _"laughter occurred in this 10sec snippet"_ but not where
- Results suggest that finely-segmented and in-domain data annotations are important 
    - without the new finely-segmented annotation of the AudioSet data the results aren't as good

---

### Laughter and Filler Detection in Naturalistic Audio - 2015
_source: https://utd-ir.tdl.org/bitstream/handle/10735.1/5058/JECS-3626-4639.10.pdf?sequence=1&isAllowed=y_
- proposes a simpler approach with features that are a combination of spectral features and pitch information
    - a CNN runs on top for classification
---
### Other possible Papers
- Getting the last laugh: Automatic laughter segmentation in meetings
    - https://scholar.google.com/citations?view_op=view_citation&hl=en&user=2FtK1gUAAAAJ&citation_for_view=2FtK1gUAAAAJ:UebtZRa9Y70C
- Capturing, Representing, and Interacting with Laughter
    - https://dl.acm.org/doi/10.1145/3173574.3173932
### Questions
- Which paper first used CNNs instead of other techniques for audio classification?
- How much should I talk about the history of audio processing in general? 



