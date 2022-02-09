- Ondrej: "Your results look like you are doing VAD -> triggering on any noise"

- Ondrej suggests creating a for loop to automate the process and then trying lots of adjustments like
  - change class balance
  - train longer/shorter (-> overfitting)
    - plot prec/recall on validation batches during training to spot this
      - do this validation avery epoch
  - more data
  - change feature representation

=> all these investigations can go into my thesis

- Philip: "You can extend the period where you can implement stuff when you write your thesis in parallel"

- show that in theory your algorithm can run in real time

  - that's enough for the thesis but describe this carefully
    - compare that to why you think Gillick et al's model can't run in real-time

- align work/time spent with the amount of conent in the thesis for that particular section

  - mention that dataloading/training part took a long time
    - refer to the open source contribution you made to lhotse
      - a framework in the Speech-Community that will likely be used a lot more later

- mentioning crude analysis (like the one on librosa) is still better than no analysis
