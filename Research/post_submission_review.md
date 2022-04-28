Truong and van Leeuwen 2007:
- talk lots about features and how to distinguish between laughter and speech
    - today: deep learning approach
- they don't state how they got the speech segments 
- they manually filtered out 3574 laughter segments 
    - that's less than half of the segments that I used
        - but they didn't use the whole ICSI corpus

Cai et al. performance:
- precision: 92.95%
- recall: 86.88%

Knox 2006
- first approach using SVMs was a lot less efficient
    - I think it's because they MFCCs, not necessarily the SVM
        - for the NN they used the raw spectrogram (deep learning approach)
            - they used MFCCs and deep learning
            - even though both use MFCCs, for SVMs the features passed to it need to be crafted by hand whereas the raw MFCCs can be passed to the NN 

prosodic features seems to be a bit vague
- this paper lists MFCCs under prosodic features 
    - http://www.apsipa.org/proceedings_2013/papers/162_Emotion-recognition-Suzuki-2930025.pdf