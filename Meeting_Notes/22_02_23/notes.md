- check validation during training

  - ideally validate on whole validation set (if performance allows it)
  - change size of batches + increase log frequency (DONE)

- Use 1.0 for precision if no predictions are made? (TP/ TP+FP) where TP+FP=0​

  - yes, use that (one end of prec/recall curve)

- Do I need to explain strongly labeled data?

  - yes, Ondrej didn't know about it
  - give brief explanation at first use then refer back to

- Use 'we' or 'I' in the thesis?

  - if it's just me saying/thinking/doing something say 'I'
  - if it's me and the reader: 'we' is fine

- Spelling of dataframe/dataset (data frame/data set)

  - dataset seems fine in ML literature
  - generally check what literature does and do that as well
  - avoid jargon if you can

    - possibly use table instead of dataframe
      - makes it easier for the reader

- Should images/graphs/tables go in the appendix?
  - if I want the reader to see them, put them in the main body
  - appendix is for additions/sidenotes/etc.
    - something the body can refer to for additional information
- How should I reference contributions to lhotse?
  - list of Pull-Request
    - if it's 3: main body
    - if it's 20: appendix and refer to it
    - use clickable links in PDF

**General advice on Thesis**:

- put yourself in the shoes of the reader
- guide the reader
  - think about prior knowledge of the reader
  - adding sentences like the following helps guiding:
    - 'If you are familiar with ML concepts you can skip this section and move to section X'
- 'the more pictures the better' - Philip :D

- What is loss?
  - give clear definition of metrics used in graphs/tables
  - can mention that in 'Evaluation Metrics' section
