# Reserach Review

### CNN Architectures For Large-Scale Audio Classification - 01/2017
_source: https://arxiv.org/pdf/1609.09430.pdf_

**Possibly use CNN Architecture for our classification problem**

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
- whole eval set and 5% of the training set - chosen at random
- conclusion states that a classifier trained on the large 'weakly-labeled' dataset can be improved via-fine-tuning
- evaluations were on fixed-size frames 
    - but strong labelling suggests directly predicting the segment boundaries  
        <span style="color:green">
        -> that's interesting for us!
        </span> 

![Evaluation](imgs/strong_labels_eval.png)

---
# Applause
### Novel approach for detecting applause in continuous meeting speech - 2011
_source: https://ieeexplore-ieee-org.ezproxy.is.ed.ac.uk/document/5941827_
> The proposed approach is based on the short - time autocorrelation
> function features - decay factor, first local minimum 
> and band energy ratio.
- 4 features with thresholds
- compared conventional method using mel frequency cepstral coefficients (MFCC) feature vectors and GMM as classifier
    - 36 feature vector 12 of the 13 MFCC and their first and second derivative
- dataset: 4 hours 37minutes of meeting speech
    -**unpublished**

| method | precision rate | recall rate | F1 score |
| -------| ---------------| ------------| ---------|
| proposed        |  94.40% | 90.75% | 92.54% |
| conventional    |  67.47% | 96.13% | 79.29% |

---
### Heterogeneous mixture models using sparse representation features for applause and laugh detection - 2011
_source: https://ieeexplore-ieee-org.ezproxy.is.ed.ac.uk/stamp/stamp.jsp?tp=&arnumber=6064620_
- first read doesn't convince me
    - it's already 10 years old and the technique is likely deprecated
    - **dataset unpublished**
        - quality of data unclear
        - 20 hours of video (from Youku) in total
        - but only ~1.77h in DB if all segments are 8s long:
        > The  database  includes  800  
        > segments of each sound effect. Each segment is about 3-
        > 8s  long  and  totally  about  1hour  data  for  each  sound  
        > effect. All the audio recordings were converted to 
        > monaural wave format at a sampling frequency of 8kHz 
        > and  quantized  16bits.  Furthermore,  the  audio  signals  
        > have  been  normalized,  so  that  they  have  zero  mean  
        > amplitude  with  unit  variance  in  order  to  remove  any  
        > factors related to the recording conditions. 
---
### Applause Sound Detection - 2011
_source: https://www-aes-org.ezproxy.is.ed.ac.uk/tmpFiles/JAES/20210927/JAES_V59_4_PG213hirez.pdf_ 
- very small dataset - 1.75h if all snippets were 30s...
    - **unpublished**
> The data set for training and test comprises 210
> excerpts of commercial recordings of between 9- and 30-
> second length each.
- real-time detection with low latency!
<span style="color:green">
    - that's interesting for us!
</span>

**Confusion Matrix**: binary classification using MFCC and LDD + delta and sigma features
| | Predicted Applause | Predicted No |
|---| ---| ---|
| Applause | 83 |   2| 
| No Applause | 3  | 12  |
- best performance using a combination of MFCC and LLD(low-level descriptors)
    - using MLP (Multilayer Perceptron) and SVM with radial basis functions
- recognises 95% of applause correctly
---
### Characteristics-based effective applause detection for meeting speech - 2009
_source: https://www-sciencedirect-com.ezproxy.is.ed.ac.uk/science/article/pii/S0165168409000759/pdfft?md5=b6f3199e457436e57db04012f64a90a3&pid=1-s2.0-S0165168409000759-main.pdf_

- uses decision tree instead of complex statistical model
    - faster decisions possible 
    - interesting for our problem

- Data: 50h of meeting speech (multi-participant)
    - **not published**

| Parameters | proposed algor. | traditional algor. |
| --- | --- | ---|
| Precision | 94.34 | 91 |
| Recall   | 98.04  | 94.12 |
|F1 - measure | 96.15 | 92.53 | 
| Computational Time (min)| 59.4 | 92.5 |
---
### Discriminative Feature Selection for Applause Sounds Detection 
_source: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4279120_

- doesn't convince me
    - the improvements don't seem that relevant to me
![Performance](./imgs/Performance_of_various_feature_sets.png)



# Laughter
### Robust Laughter Detection in Noisy Environments - 09/21 
_source: https://www.isca-speech.org/archive/pdfs/interspeech_2021/gillick21_interspeech.pdf_
- Update on the original paper:
    - **Capturing, Representing, and Interacting with Laughter (2018)**
        - https://dl.acm.org/doi/pdf/10.1145/3173574.3173932
    - trained on Switchboard dataset
    - more general, exploring interaction with laughter - qualitative
    > This work contributes a qualitative evaluation of our prototype
    > system for capturing, representing, and interacting with laughter. 
- Augmentation of AudioSet:
    - https://github.com/jrgillick/laughter-detection/tree/master/data/audioset
    - 148 minutes of audio
        - 58min of laughter -> 1492 distinct laughter events
- evaluated on both Switchboard and augmented AudioSet
- uses new dataset on top of AudioSet corpus
    - with precise segmentations for the start and end points of each laugh
- prior work performs badly in noisy environment  

<span style="color:green">

- How noisy do we expect the input of our domain to be?
    + Each input is separate
    - poor mic quality
    - background noise in the home office

</span>

![Results](imgs/Robust_Laughter_detection_res.png)
- AudioSet only specifies _"laughter occurred in this 10sec snippet"_ but not where
- Results suggest that finely-segmented and in-domain data annotations are important 
    - without the new finely-segmented annotation of the AudioSet data the results aren't as good


---

### Laughter and Filler Detection in Naturalistic Audio - 2015
_source: https://utd-ir.tdl.org/bitstream/handle/10735.1/5058/JECS-3626-4639.10.pdf?sequence=1&isAllowed=y_
- proposes a simpler approach with features that are a combination of spectral features and pitch information
    - a CNN runs on top for classification
---
### Quantitative Laughter Detection, Measurement, and Classification—A Critical Survey (2016)
_source: https://ieeexplore-ieee-org.ezproxy.is.ed.ac.uk/stamp/stamp.jsp?tp=&arnumber=7403873_
- _[GS-Ranked 6th "laughter detection"]_

> laughter as a multimodal social and emotion expression behaviour   
_(taken from conclusion)_

> but unfortunately, a comprehensive theory of laughter has
> not yet been developed.
_(taken from conclusion)_

> The purpose of this survey is to bring together the different
> results obtained in different fields, to both present all the possible
> methods to quantify laughter and try to draw a comprehensive
> physiological model of this complex human behavior.
> _(taken from introduction)_

> The best performances—from 70% to 90% correct
> classification rate—have been obtained using Mel-frequency
> cepstral coefficients and perceptual linear prediction over stan-
> dard audio spectrum features, and combining classifiers that
> use both spectral features and prosodic information. This is not
> very surprising as laughter is tailored on the human hearing
> apparatus.

- more general reflection on the state of research back then
- suggests the use of multimodal works combining studies on acoustic analysis, with the ones on respiratory and physiological changes as well as the ones on facial expression
- talks quite a bit about applications and less technical facts
---

### K. P. Truong and D. A. Van Leeuwen, “Automatic detection of laughter,” in Proc. 9th Eur. Conf. Speech Commun. Technol., 2005, pp. 485–488.
- _source: http://lands.let.ru.nl/literature/leeuwen.2005.3.pdf_

- uses ICSI Meeting Cor-pus and the Dutch CGN corpus
- looks at 'paralinguistic events' for classification of speaker's emotinoal state
    - laughter was annoted most often -> thus, focus on laugher detection
- features tried:
    - PLP (Perceptual Linear Prediction) -> **Best Performance**
    - per frame - pitch & energy 
    - global - pitch & voicing 
        - **Promising** because pretty good performance with only 6 features per segment
        - suggests further research to possibly combine this with PLP
    - modulation spectrum features
- Discrimination: GMM (Gaussian Mixture Models)

- works on predifined segments with binary output (contains laughter/does not contain laughter)
    - suggests HMMs for laughter detection with automatic detection of boundaries

---

### Truong, K. P., & Van Leeuwen, D. A. (2007). Automatic discrimination between laughter and speech. Speech Communication, 49(2), 144-158. 
- _source: https://www.sciencedirect.com/science/article/pii/S0167639307000027_
- follow-up of the paper above
- uses ICSI Meeting Cor-pus and the Dutch CGN corpus
    - segments average duration: ~2s

- long-term goal: Emotion classification 
    - laugh detector is one step towards that
- Features tried: spectral(=relating to the amount of vibration at each individual frequency) and prosodic(=relating to the rhythm and intonation)
- Discriminators: GMM, SVM and MLP

- mentions possibility of gender and cultural discrimination 
- used segments of ONLY laughter or ONLY speech

- performs well (EER < 10%) on CGN (desktop recordings in Dutch) even though it's trained on ICSI (close-talk mic per speaker in English)
 
---
### L. S. Kennedy and D. P. Ellis, “Laughter detection in meetings,” in Proc. NIST ICASSP Meeting Recog. Workshop, Montreal, Canada, 2004, pp. 118–121
- _source: https://academiccommons.columbia.edu/doi/10.7916/D8JD565R_
- _[GS-Ranked 1st "laughter detection"]_

- trained on ICSI Bmr-subset -> using just two table-top microphones
- laughter-event defined when multiple people laugh at once
    - different to ours
- data split into 1sec windows
- Features tried:
    - MFCCs
    - delta MFCCs
    - modulation spectrum
    - spatial cues
- Discrimination: SVM 
- **Best Performance:** MFFCs
    - only using first 6 of MFCCs gives same performance to all 13 coefficients
    - Recall: 87%
    - False Alarm Rate: 13%
- percentage of people laughing is highly correlated to distance from SVM decision boundary
    - possibility to use 'certainty' of classification as predictor of the intensity of the laughter event
- good performance on CMU and NIST recordings
- bad performance on LDC recordings - WHY COULD THAT BE? 

---
### M. Knox, “Improving frame based automatic laughter detection,” Univ. California, Berkeley, CA, USA, EE225D Class Project, 2007V
- _source: https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.910.1294&rep=rep1&type=pdf_
- _[GS-Ranked 2nd "laughter detection"]_
- uses ICSI database
> we hope that our work serves as a baseline for future work on
> frame-by-frame laughter recognition on the Meetings database,
> which provides an excellent testbed for laughter research  
    - states that ICSI db is a good testbed for laughter research

- good introduction mentioning both Kennedy and Truong
- goal: examine the many usees of laughter in the future
- Features tried: 
    - MFCCs
    - fundamental frequency
    - rms frequency
    - ac peak
- Discrimination: NN with 1 hidden layer 

- **Best Performance:** MFCCs + ac peaks (combined)


---
### Other possible Papers
- Getting the last laugh: Automatic laughter segmentation in meetings - 2008
    - https://scholar.google.com/citations?view_op=view_citation&hl=en&user=2FtK1gUAAAAJ&citation_for_view=2FtK1gUAAAAJ:UebtZRa9Y70C
    - uses  ICSI Meeting Recorder Corpus
    - 78.5% precision rate and 85.3% recall rate 
        - not better than other papers listed
- Improved Audio-Visual Laughter Detection Via Multi-Scale Multi-Resolution Image Texture Features and Classifier Fusion
    - https://ieeexplore.ieee.org/abstract/document/8461611
    - _Can we use the audio analysis separately from the video?_
    - uses MAHNOB db

- Highlight sound effects detection in audio stream
    - https://ieeexplore.ieee.org/abstract/document/1221242
    - _Could mention that in background chapter as another use of laughter detection in the past_
    - used HMM

- Bachorowski, J. A., Smoski, M. J., & Owren, M. J. (2001). The acoustic features of human laughter. The Journal of the Acoustical Society of America, 110(3), 1581-1597.
    - https://asa.scitation.org/doi/abs/10.1121/1.1391244
    - apparently also mentions gender issue

- Ohara, R., 2004. Analysis of a laughing voice and the method of laughter in dialogue speech. Unpublished Masters Thesis, Nara Institute of Science and Technology.
    - distinguishes different types of laugh in Japanese
    - COULD BE A STRETCH GOAL FOR THE PROJECT -> FOLLOW UP ON THE PROJECT 

### Other possible resources
- "Optimized time series filters for detecting laughter and filler events"
    - link: http://publicatio.bibl.u-szeged.hu/14571/7/3311351_cimlap_tartj.pdf 
    - **Can I get access to this talk?**
### Observation
There are quite a few papers using **audio-visual** detectors, meaning they combine the results of a separate audio and video classifier
- Not desirable for our project
    - complexity
    - privacy implications

**Examples:**
- Berker Turker, Bekir et al. “Audio-Facial Laughter Detection in Naturalistic Dyadic Conversations.” IEEE transactions on affective computing 8.4 (2017): 534–545. Web.
- Studies from Koc University Provide New Data on Affective Computing (Audio-Facial Laughter Detection in Naturalistic Dyadic Conversations). NewsRX LLC, 2018. Print.
- Petridis, Stavros, Brais Martinez, and Maja Pantic. “The MAHNOB Laughter Database.” Image and vision computing 31.2 (2013): 186–202. Web.
    - _corresponding corpus_
- S. Petridis and M. Pantic, “Audiovisual laughter detection based on temporal features,” in Proc. 10th Int. Conf. Multimodal Interfaces, 2008, pp. 37–44.
    - _[GS-Ranked 5th "laughter detection"]_

### Questions
- Which paper first used CNNs instead of other techniques for audio classification?
- How much should I talk about the history of audio processing in general? 
- If applause snippets are group actions, might that be a problem for our usecase?
- what are delta and sigma features?
- How do we address gender differences?

# Projects
### IDEO Laughter Project (2018)
_source: https://www.ideo.com/blog/why-your-office-needs-a-laugh-detector_
- used AudioSet
- 87% accuracy with LSTM model
> applying batch normalization to the LSTM input was very 
> important for getting the model to converge  
-> got this insight from https://github.com/ganesh-srinivas/laughter/ (2017)

- chose LSTM over logistic regression model because LSTM was able to handle variable length input
    - logistic regression needed same input length as training data, namely 10s snippets

