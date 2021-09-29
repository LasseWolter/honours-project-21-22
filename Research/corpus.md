# Corpus to use for Project

### Google audio set
source: https://research.google.com/audioset/

**General**
- 2.1 million annotated videos
- 5.8 thousand hours of audio
- 527 classes
- labeled 10s snippets

**Applause**
- Path: Human sounds > Human group actions
- Sub-categories: None
- Overall: 2247 videos - 6.2 hours
- est. accuracy: 90% (9/10)
- Details: https://research.google.com/audioset//dataset/applause.html

**Laughter**
- Path: Human sounds > Human voice > Laughter 
- Sub-categories: Baby laughter, Giggle, Snicker, Belly laugh, Chuckle/chortle
- Overall: 5696 videos - 15.8 hours
- est. accuracy: 100% (10/10)
- Details: https://research.google.com/audioset/dataset/laughter.html

**Background**
_source: https://storage.googleapis.com/pub-tools-public-publication-data/pdf/45857.pdf_
- **Motivation**: Create a comprehensive dataset for AES like ImageNet is for tasks like object detection 
    >  Comparable problems such as object detection in images have
    >  reaped enormous benefits from comprehensive datasets – principally ImageNet

    > ImageNet appears to have been a
    > major factor driving these developments, yet nothing of this scale
    > exists for sound sources.
- **Ontology**: 
    - match immediate human understanding of a sound
    - individual categories should be distinguishable by a 'typical' listener
    - ideally sound classification based on sound alone (no visual or contextual cues)
    - Hierarchical structure - to aid annotators in most specific annotation 

**Strong labeled version
- https://research.google.com/audioset/download_strong.html dataset
- which describes clips of varying length - manually chosen by annotators
- whole eval set and 5% of the training set - chosen at random
    - 290h training data
    - laughter: ~0.79h / ~48min 
    - applause: ~0.31h / ~20min

- conclusion states that a classifier trained on the large 'weakly-labelled' dataset can be improved via-fine-tuning
- evaluations were on fixed-size frames 
    - but strong labelling suggests directly predicting the segment boundaries  
        <span style="color:green">
        -> that's interesting for us!
        </span> 

### Other corpora
- ICSI meetings database (2004)
    - http://www1.icsi.berkeley.edu/Speech/mr/icsimc_doc/overview.txt
    - recorded in a conference room with a microphone for each speaker
    - ~72 hours of transcribed meeting speech 
        - includes laughter 
        - does NOT include applause
- Switchboard (telephone conversations) (1997)
    - https://catalog.ldc.upenn.edu/LDC97S62
    - ~260 hours of transcribed speech 
        - release: August 1997
        - last transcription update: 01/29/03
        - includes laughter 
        - does NOT include applause
- SSPNet-Mobile Corpus(telephone conversations)
    - University of Glasgow usage only? 
    - ~12h of annotated phone conversations
        - contains laughter but no applause
- There are more of this type
    - usually the annotation contain laughter but not applause
