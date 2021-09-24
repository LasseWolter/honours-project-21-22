# Corpus to use for Project

### Google audio set
source: https://research.google.com/audioset/

**General**
- 2.1 million annotated videos
- 5.8 thousand hours of audio
- 527 classes

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

### Other corpora
- ICSI meetings database 
    - recorded in a conference room with a microphone for each speaker
- Switchboard (telephone conversations)
- SSPNet (telephone conversations)
