# Thesis

- good background chapter
  - there is not that much research done in this area
  - add section relating to other Realtime-ML algorithms
    - e.g. Wake word detection
      - needs to with run very low computational power
        - usually buffers and discards audio
        - first simple NN with low precision -> filter out candidates
        - second slightly more complex NN with higher precision -> select candidates

# Possible Scopes of the project

### 1 Focus on practicality

- focus on practicality of such a system
  - semantics of laughter -> linguistic aspects of laughter
    - laughter is part of our communication and conveys meaning
      - not all laughter is the same
  - doubts the practicality of the system as whole:
    - will the output of the system match what phillip wadler imagined?
- get more realistic and accurate training data by training an ASR model to detect laughter bounds for transcription
  - 'quite heavy lifting for the outcome' -> recommendation is 2

### 2 Focus on low-latency and low-power consumption

- such a system would run all the time during a meeting
  - thus, power consumption must be minimal (-> like wake word detection)
- developing something in this area might be useful for other use cases
  - e.g. counting the amount of laughter that happens during a meeting
    - without actually recording the meeting -> PRO for privacy
