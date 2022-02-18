g Log:

### Wednesday, 27.10.21

- Running experiments for first 5 `Bed`-Meetings with
  - min_length = 0.2s
  - thresholds = 0.1, 0.2..., 0.9
- Script for downloading `Bed`-Meetings is

### Thursday, 28.10.21

- Realised that 9 different thresholds for each channel will take a long time
  - After some manual testing decided on 4 thresholds (0.2,0.4,0.6,0.8) for further testing
- Running experiments for `Bed`-Meeting 6-11 with 4 thresholds each

- created simple 'splay'-bash-function which allows to play .shp files directly

```
# play sph file
splay() {
  sph2pipe "$1" > "out.wav" && play "out.wav" "${@:2}";
}
```

- Textgrid has following format:

```
xmin = 0.0
xmax = 3491.7732565980336
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "laughter"
        xmin = 0.0
        xmax = 3491.7732565980336
        intervals: size = 6
            intervals [1]:
                xmin = 0.0
                xmax = 1092.9973749000299
                text = ""
            intervals [2]:
                xmin = 1092.9973749000299
                xmax = 1093.5322345758389
                text = "laugh"
            intervals [3]:
                xmin = 1093.5322345758389
                xmax = 2956.3089568076443
                text = ""
            intervals [4]:
                xmin = 2956.3089568076443
                xmax = 2957.0996189371012
                text = "laugh"
            intervals [5]:
                xmin = 2957.0996189371012
                xmax = 3490.9128301630367
                text = ""
            intervals [6]:
                xmin = 3490.9128301630367
                xmax = 3491.7732565980336
                text = "laugh"
```

- all laughter events are in one item/IntervalTier with name `laughter'
- the interval size of this tier is double the size of laugther events
  - includes intervals between two detected laughter-events
- if the interval text=="laughter" we have a laughter event:

  - xmin and xmax give the boundaries of this event

- concat all chan4\*.wav files with in between sound (break.wav)

  - bash:

  ```
    sox $(for f in chan4*.wav; do echo -n "$f break.wav "; done) output.wav
  ```

  - allows for easier manual evaluation

- **Question**: If you evaluate manually you might get different impression
  - laugh events that occur directly next to speech will not be considered correct
    when evaluated automatically (since we filtered for laugther-without-speech in transcripts)
    - when listening to them manually keeping them might be desirable

### Friday, 29.10.21

- Moved already analysed meeting files (Bdb, Bed) to archive folder on DFS
  - Then removed them from all scratch disks on the cluster
- Downloaded new meeting files (Bmr) to the DFS
- Created experiment scripts for the first 10 Bmr-meetings
  - run those experiments with 4 different thresholds
- ran experiments remaining Bmr-meetings
- ran experiments for meetings up to Bro014

### Saturday, 30.10.21

- Copied outputs up from DFS to AFS
  - currently got output up to Bro014
- Run experiments for Bro15 up to Bro28
- Run experiments for remaining meetings (Bsr, Btr, Buw)

### Tuesday, 02.11.21

- code cleanup/refactor
  - changed module structure for clearer code
  - added analyse.py: for analysing the results of the predictions
- Added to laughs_to_wav.py to also allow reading from csv files
  - thus, steps to get laughter wavs from transcript are:
    1. parse transcript into pandas dataframe
    2. write pandas dataframe to .csv
    3. run laughs_to_wav.py on this .csv
- used code above to generate breath-laugh audios (see `output_processing/Buw_breath_laugh/`)
  - imo, they shouldn't be considered for our project

# Friday, 05.11.21

- decided against the portion library for representing intervals
  - I feel like the interval library would be overkill https://github.com/AlexandreDecan/portion and just add unnecessary complexity to the code
  - The solution using sets seems simpler and sufficient for our project.
    IDEA for solution using sets:
- Turn each predicted laugh segment into a set of laugh frames
- Turn each transcribed laugh segment into a set of laugh frames
- Turn all invalid segments (where laugh occurs next to speech) into a set of invalid frames
  - granularity should be defined according to existing literature
- union all laugh frames of a particular channel
- subtract all invalid frames
- correctly predicted frames = intersection of predicted and transcribed frames

CORRECTION:

- this idea can quite nicely be realised using closed intervals from the Portion library
- first dirty version is implemented and works

  - but a lot slower than before: need to look at performance
  - recall and precision both went up a bit but recall is still very low: need to investigate this further

- For future experiments change the `get_laughter` function in laugh_segmenter to allow faster processing of several thresholds/min_lengths

- Should filter out events shorter than min_length from evaluation?

  - I included that but doesn't make a big difference

- Highest 20 recalls can be seen here:
- **73.32%** is the maximum...need to investigate further

```
       Unnamed: 0 meeting  threshold  precision    recall

  256 256 Btr001 0.2 0.244948 0.733196
  10 10 Bmr025 0.2 0.320030 0.712310
  6 6 Btr002 0.2 0.217203 0.695080
  254 254 Btr001 0.4 0.400939 0.578860
  181 181 Bed003 0.1 0.108910 0.578799
  235 235 Bmr026 0.2 0.269870 0.573287
  117 117 Bmr021 0.2 0.226558 0.570482
  148 148 Bns003 0.2 0.240635 0.557725
  167 167 Bsr001 0.2 0.241915 0.550706
  171 171 Bed013 0.2 0.184142 0.506721
  17 17 Bro015 0.2 0.069715 0.503831
  268 268 Bmr027 0.2 0.245510 0.498998
  4 4 Btr002 0.4 0.371077 0.498949
  8 8 Bmr025 0.4 0.551730 0.495047
  204 204 Bro017 0.2 0.298791 0.494727
  29 29 Bmr020 0.2 0.183779 0.494482
  58 58 Bmr007 0.2 0.245637 0.493386
  144 144 Bmr016 0.2 0.237648 0.492793
  293 293 Bmr024 0.2 0.170377 0.480184
  163 163 Bmr003 0.2 0.154977 0.479608
```

- after printing out the total predicted and total transcribed times it's very
  quite obvious why the recall is so low. The total predicted time is way lower.
  If the tot_pred_time is only half of the tot_tranc_time this means that we can get
  a maximum of 50% recall... need to investigate this further

# Saturday, 06.11.21

Comparing new set method including removed mixed_laugh_segements to original method
(indices are odd because I removed the thresholds 0.1,0.3,0.5,0.7,0.9 due to small amount of data)

```
**Original** method:
threshold precision recall
mean mean
1 0.2 0.154860 0.312307
3 0.4 0.344310 0.175887
5 0.6 0.515102 0.078559
7 0.8 0.650163 0.029492

**New** method:
threshold precision recall
mean mean
1 0.2 0.173080 0.352611
3 0.4 0.373184 0.193568
5 0.6 0.543314 0.083950
7 0.8 0.667196 0.030589
```

**MADE A MISTAKE HERE**

- the new method wasn't properly exectued since the tot-pred-length was still taken to be the length of all predictions
  instead of only the valid ones after subtracting the 'mixed-laughs'
- if this is changed the precision goes up significantly

```
  threshold precision recall
  mean mean
  1 0.2 0.205677 0.352611
  3 0.4 0.494141 0.193568
  5 0.6 0.747847 0.083950
  7 0.8 0.904399 0.030589
```

**Outcome**

- recall and precision increase, but recall is still very low
- both versions still represented in code (as OLD version and NEW version)
- do further checkup on this

### Comparing inclusion vs. exclusion of breath-laugh (before above mistake was fixed)

#### Before fixing mistake above

```
**Including breath-laugh**
threshold precision recall
mean mean
1 0.2 0.173080 0.352611
3 0.4 0.373184 0.193568
5 0.6 0.543314 0.083950
7 0.8 0.667196 0.030589
tot length: 14714.835000000001
tot events: 8714

**Excluding breath-laugh**
threshold precision recall
mean mean
1 0.2 0.171825 0.374076
3 0.4 0.372363 0.206755
5 0.6 0.542973 0.089204
7 0.8 0.667171 0.032287
tot length: 13837.871000000003
tot events: 7902
```

**Outcome**

- 812 'breath-laugh'-events were excluded
- recall increases slightly with little variation in precision
- **still**, recall is very low
  - There must be a different reason for this

#### After fixing mistake above

```
**Including breath-laugh**
threshold precision recall
mean mean
1 0.2 0.205677 0.352611
3 0.4 0.494141 0.193568
5 0.6 0.747847 0.083950
7 0.8 0.904399 0.030589
tot length: 14714.835000000001
tot events: 8714

**Excluding breath-laugh**
threshold precision recall
mean mean
1 0.2 0.204184 0.374076
3 0.4 0.492798 0.206755
5 0.6 0.747397 0.089204
7 0.8 0.904363 0.032287
tot length: 13837.871000000003
tot events: 7902
```

**Outcome**

- slight improve in recall
- **still**, recall is very low
  - There must be a different reason for this

# Monday - Thursday, 8-11.11.21

- improved parts of Thesis bg-chapter according to feedback
  - reread Knox and Mirghafori and added more detail about it
  - found out that 3 of the cited papers that use ICSI, actually use the Bmr-subset of the ICSI corpus

# Friday 12.11.21

- Looked over some general AED-papers to look for latency factors of real-time AED

  - ideas:
    - decoding?
    - buffer size/window size
      - if 1s around the frame is considered we need to wait half a second before we get the full context
    - preprocessing applied
      - if I need to do some complex transformation first to get the input to my model that will take time

- installing dependencies for laughter-gillick didn't work with my local python installation and venv

  - worked with conda though
  - cpu version of torch needs to be installed directly from the torch-website
    - see guide https://pytorch.org/get-started/locally/

- found a real-time-ish project on github

  - after a brief look at the code, this approach seems to wait for silence before outputting
    - that's not suitable for us -> latency way to high

- to load the checkpoint onto the cpu
  - documentation: https://pytorch.org/tutorials/beginner/saving_loading_models.html

```
  checkpoint = torch.load(model_path+'/best.pth.tar', lambda storage, loc: storage)
  print(checkpoint.keys())
  model.load_state_dict(checkpoint['state_dict'])
```

- created `orange evals` - folder with orange-scripts for evaluation

  - 4 orange scripts up to now:
    - `general_evaluation` needs further editing
    - 2 scripts for looking at the segment length in Bmr031
    - `stats_with_different_min_length` needs further investigation
      - seems like the increased min_length almost doesn't change the number of predictions
        - do all the correct predictions lie in the transcribed segments that are quite long?
          - otherwise the number of 'valid' predictions should reduce bc all the shorter segments should be discarded...

- looked at paper results when trained on one dataset and evaluated on a different one
  - they still get recall values from 70-90%
  - check which parameters they used for min_length and threshold

# Tuesday-Wednesday, 16-17.11.21

- created aggregate laughter length plots
- created "aggregate predicted laughter length/aggregate transcribed laughter length"-ratio plots
  - possible next plot:
    - ratio of "recall/ratio-calculated-above"
      - states how much of the possible recall (defined by this ratio) was achieved

# Friday, 19.11.21

Real-time-factor calculation

- min_length parameter is only used in post processing (like threshhold)

  - thus it doesn't influence the calculation of the RTF (real-time-factor)
    - because it depends on prediction over the whole audio length and THEN filtering afterwards
    - BUT if the predictions have to happen in realtime, then it has an influence

- CPU thinkstation
  | Audio Duration | Iterations run | average RTF |
  | ---|--- | --- |
  | 3s | 20 | 1.31|
  | 30s | 20 | 1.41 |
  | 120s | 10 | 1.49|

- CPU Appleton tower
  | Audio Duration | Iterations run | average RTF |
  | ---|--- | --- |
  | 3s | 20 | 0.63 |
  | 30s | 20 | 0.84|
  | 120s | 10 | 0.81|

- GPU Appleton tower
  | Audio Duration | Iterations run | average RTF |
  | ---|--- | --- |
  | 3s | 20 | 0.14 |
  | 30s | 20 | 0.10|
  | 120s | 20 | 0.10|
  | 300s | 10 | 0.10|

- using alter_req.txt as alternative requirements.txt file for python env creation works on AT computer

_alter_req.txt:_

```
librosa==0.8.1
nltk==3.6.5
numpy==1.20.0
pandas==1.3.4
praatio==3.8.0
pyloudnorm==0.1.0
tensorboardX==1.9
tgt==1.4.4
torch
tqdm==4.62.3
```

# Tuesday, 14.12.21

- went through the code of Gillick et al. in more detail
- the two things I need to change and pass to the DataLoader are:
  1. train_df
  2. audio_hash: a mapping from audio_paths to the actual audio files

# Saturday 08.01.22

- using the train_df created yesterday and trying to use if with Gillick et al.'s model
- one error caused is the following:

  > ValueError: can't extend empty axis 0 using modes other than 'constant' or 'empty'

  - can be fixed by adding `pad_mode` parameter 'empty' to librosa.feature.melspectrogram()
    in audio_utils
  - this error was caused by an empty array being passed to the melspectrogram-function
    - I didn't fully understand why this happened
    - I moved the melspectrogram function into the dataloader itself and didn't get empty arrays
      - not sure why this was the case when the function in audio_utils was used

- current error happens in the forward pass:
  > running_mean should contain 448 elements not 64

# Sunday 09.01.22

- looked into audio-processing and audio-signal theory
  - helped to understand the different parts that audio signals are made up off and how
    features can be extracted from there
    - the videos looked at are from: https://www.youtube.com/channel/UCZPFjMe1uRSirmSpznqvJfQ
- fixed bug: current data loader was using the whole laugther segments - NOT the subsamples
  - after this change now all spectrograms are of the same shape
- fixed issue with wrong dimension from yesterday
  - problem seems to be the different sample rate of ICSI data
    - changing the sr to 8000 works
      - however, this isn't wanted -> unnecessary data loss
      - need to find a better way -> possibly adjusting the hop_length or NN-structure
- changed create_train_df.py script to create_data_df.py and adjusted to create three distinct dfs
  - one for training, one for validation, one for testing
  - need to load validation set in Gillick's train code
    - currently just using train_df as validation data

# Monday 10.01.22

- fixed Bug: Can now use the normal melspec function
  - problem was that the subsampled file passed to the melspec-function in audio utils was subsampled again
  - adjusted function in audio_utils accordingly
- adjusted create dfs script to also create 3 'dummy_dfs' only using the data from one meeting

  - this can be used for debugging

- training with dummy data seems to work. But seems like validation set is to small atm

  - val_batches_per_log = 0 which means that no batches are used for evaluation
    - currently the reason is that the number of validation samples is smaller than half of the batch size
      - thus, the division in torch_utils.py rounds to 0

- currently using the normal res_net without augmentation because I don't have noise audio files
  - check again how much worse the normal resnet performed in Gillick et al.'s paper

# Tuesday 11.01.22

Command used to run train.py on certain cluster node (here 'landonia12')
`sbatch -w landonia12 --array=1-1%10 cluster_scripts/laughter_train.sh cluster_scripts/experiment.txt --cpus-per-task=4 --gres=gpu4 --mem=8000`

- can be helpful if data is already present on disk of a certain machine

-Getting following error when trying to connect to MLP-cluster

```
Could not chdir to home directory /home/s1660656: Transport endpoint is not connected
rm: cannot remove '/home/s1660656/.last_login': Transport endpoint is not connected
-bash: /home/s1660656/.last_login: Transport endpoint is not connected
-bash: cd: /home/s1660656: Transport endpoint is not connected
-bash: /home/s1660656/.bash_profile: Transport endpoint is not connected
```

- loaded data onto machines: **landonia12** and **landonia04**

# Wednesday 12.01.22

- Continued training

  - checkpointing works but training is really slow
    - only 30 batches in 2 hours (960 segments -> 8s audio processed per minute)
      - discussed this issue in the meeting for more details see meeting notes for 12.01.22

- Added another parameter to train.py: data_dfs_dir
  - allows to specify the dataframes used for train/val/test split
- **Some thoughts on training**
  - How did Knox and Mighafori train
    - is training with fixed 1s segments sensible?
  - random split vs. even split across meetings
  - balanced vs. imbalenced classes (speech vs. laughter)
  - does the melspec transformation run on GPU?
    - does any preprocessing run on GPU in their code?

# Saturday, 15.01.22

- updated store_all_icsi_audio.py script
  - added function to split the set of training folders into subsets
    - the whole training data couldn't be loaded into one hash

# Monday, 17.01.22:

- changed 'i_gpu' alias on mlp cluster to allows connecting to a specific node with default settings
- debugging why the training takes so long
  - when the training runs only ~20% of one GPU are used
    - is this because the GPU waits for the data to be ready -> the dataloading is the bottleneck?
    - started writing some code to evaluate dataloading
      - such evaluations will also go in my thesis as they show how I progressed and which issues I ran into
      - also mention the quality of the existing code in the thesis and how this made working with it more complicated

# Wednesday, 19.01.22

- evaluating loading time of dataloader

**Dataloader loading times**

**SINGLE MEETING**

- 6 different audio files
  - command: `s_train.audio_path.unique().size `

_on Thinkstation CPU - Intel(R) Core(TM) i5-6500 CPU @ 3.20GHz_

| num_of_batches | total_time [s] | av_time_per_batch [s] | num_of_workers |
| -------------- | -------------- | --------------------- | -------------- |
| 1              | 60.73          | 60.73                 | 8              |
| 3              | 191.08         | 63.69                 | 8              |
| 5              | 300.04         | 60.01                 | 8              |

_on AT GPU Machine_

| num_of_batches | total_time [s] | av_time_per_batch [s] | num_of_workers |
| -------------- | -------------- | --------------------- | -------------- |
| 5              | 194.13         | 38.83                 | 8              |

_on MLP-Cluster GPU Machine_ -> **loading from DFS**

- allocated memory: 16000MB

| num_of_batches | total_time [s] | av_time_per_batch [s] | num_of_workers |
| -------------- | -------------- | --------------------- | -------------- |
| 1              | 79.26          | 79.26                 | 8              |
| 5              | 392.16         | 78.43                 | 8              |

_on MLP-Cluster GPU Machine_ -> **loading from scratch disk**

| num_of_batches | total_time [s] | av_time_per_batch [s] | num_of_workers |
| -------------- | -------------- | --------------------- | -------------- |
| 1              | 80.71          | 80.71                 | 8              |
| 1              | 75.07          | 75.07                 | 24 + 32GB mem  |
| 5              | 392.16         | 78.43                 | 8              |

**FULL DATASET**

- 135 different audio files loaded in the first 160 rows of the dataframe (5 batches \* 32 rows)
  - command: `l_train.iloc[:160, :].audio_path.unique().size`

_on MLP-Cluster GPU machine_
_Note_: Dataloading is still done by the CPU

| num_of_batches | total_time [s] | av_time_per_batch [s] | num_of_workers |
| -------------- | -------------- | --------------------- | -------------- |
| 5              | 809.78         | 161.96                | 8              |

# Thursday, 20.01.22

- messaged with Ondrej about slow dataloading

  - he suggested that it might be due to shorter recording length in the switchboard dataset
    - this seems to be true:
      - 260 hours and 2400 conversations means 6.5min of average length for a recording
        - source: https://catalog.ldc.upenn.edu/LDC97S62

- briefly looked into preloading data and storing them in a hash like Gillick et al.
  - decided against it as it seems to complicate the code even more
  - going with Ondrej's suggestion to write the dataloading from scratch
    - possibly using lhotse (https://lhotse.readthedocs.io/en/latest/)

# Friday, 21.01.22

- started looking at lhotse

  - they have a predefined recipe for ICSI-dataset to load data and transcripts into corresponding classes
    - Recording and Supervision class
  - the recipe contains a train/val/test split to minimise speaker overlap
    > This recipe, however, to be self-contained factors out training (67.5 hours), development (2.2 hours
    > and evaluation (2.8 hours) sets in a way to minimise the speaker-overlap between different partitions,
    > and to avoid known issues with available recordings during evaluation. This recipe follows [2] where
    > dev and eval sets are making use of {Bmr021, Bns00} and {Bmr013, Bmr018, Bro021} meetings, respectively.
    - https://github.com/lhotse-speech/lhotse/blob/master/lhotse/recipes/icsi.py

- why are the mic channels split into different types in the ICSI-recipe
  - even though the channels of different types are used to record people (according to preambles.mrt)
    - need to check if that has an impact on the results

```
MIC_TO_CHANNELS = {
    "ihm": [1, 2, 3, 4, 5, 6, 8, 9], # we include 6 since it is used as back-off from some speakers for which no headset-mic exists
    "sdm": [6],
    "mdm": ["E", "F", 6, 7],
    "ihm-mix": [],
}
```

This mic to channel assignment is a bit confusing to me since according the preamble.mrt there are also people recorded on channel A and B (e.g. in meeting Bdb001)

# Saturday, 22.01.22

- Looked into Lhotse more closely

  - realised that icsi-recipe isn't part of current release
    - did some adjustments to icsi recipe to make it work

- important point about Lhotse usage: **Cut recordings first, then extract features and store them**:

  > We retrieve the arrays by loading the whole feature matrix from disk and selecting the relevant
  > region (e.g. specified by a cut). Therefore it makes sense to cut the recordings first,
  > and then extract the features for them to avoid loading unnecessary data from disk (especially for very long recordings).

- I find it rather confusing that the audio-dir is a required argument whereas the transcripts dir is not required

  - further, it seems like the audio dir could be the toplevel dir where the 'speech' and 'transcripts' folder reside in
    - I'd suggest that the audio-dir should be passed as the path to the 'speech' dir and the transcripts-dir should be passed as the path to the 'transcripts' dir
      - this would make it clearer for me

- Why does the recipe use integer IDs for the channels instead of the hexadecimal IDs used in the ICSI transcripts

  - is this more efficient?

- Some channels are missing in the MIC_TO_CHANNELS-dict: e.g. channelA and channelB
  - thus, they are not even downloaded which is clearly wrong

# Monday, 24.01.22

- according to this part of the lhotse docs (https://lhotse.readthedocs.io/en/v0.6_g/datasets.html), it's recommended to use precomputed features:
  > In general, pre-computed features can be greatly compressed (we achieve 70% size reduction with regard to un-compressed features), and so the I/O load on your computing infrastructure will be much smaller than if you read the recordings directly. This is especially valuable when working with network file systems (NFS) that are typically used in computational grids for storage. When your experiment is I/O bound, then it is best to use pre-computed features.

came up with following plan:

1. Split create_data_df to match the train/val/test split used by lhotse
2. load those data_dfs in lhotse data_loader script
3. create cuts for each in the data_dfs
4. Compute and store features for all these cuts in the format used by Gillick et al.
5. Create pytorch dataloader for these cuts, that loads the already computed feature from disk

- finished step 1 to 3
- step 4: computing and storing features works as well, making use of the CutSet.from_cuts

  - **Bug**: seems like passing 'num_jobs>1' to the `compute_and_store_features` function makes it run forever

  - alternatively one can pass a dict to the CutSet constructor
  - still need to work on creating the features that match Gillick et. al.'s code

- step 5: Running into some errors when using predefined dataset classes from lhotse

  - possibly need to write one myself
    - check these two jupyter notebooks for examples:
      1. https://colab.research.google.com/github/lhotse-speech/lhotse-speech.github.io/blob/master/notebooks/lhotse-introduction.ipynb#scrollTo=A2nhNy355NaF
      2. https://colab.research.google.com/drive/1HKSYPsWx_HoCdrnLpaPdYj5zwlPsM3NH?pli=1&authuser=1#scrollTo=Rx6Rhquw9Na0

- Getting correct shape with this FbankConfig:

  - first import using `from lhotse import FbankConfig`
  - `f2 = Fbank(FbankConfig(num_filters=128, frame_shift=0.02275))`
    - 0.2275 = 16 000 / 364 -> [frame_rate / hop_length]

- I probably need to create a custom feature extractor

  - check this part of the docs: https://lhotse.readthedocs.io/en/v0.12_cm/features.html#creating-custom-feature-extractor

- lhotse doc says there are two types of manifests but compute_and_store_features() only creates a single .lca file

  > There are two types of manifests:
  > one describing the feature extractor;
  > one describing the extracted feature matrices.
  > The feature extractor manifest is mapped to a Python configuration dataclass. An example for spectrogram:

  - in contrast, using save_audio creates a directory in which a recording/audio_file for each cut is stored
  - if you increase the num_jobs there are more than one file created (as many as there are workers)
    - but it also runs forever as described above
    - this doesn't seem to solve the issue because it doesn't create separate files per cut
      - **Fix**: This can be solved by passing the following paramter `storage_type=LilcomFilesWriter`
      - this stores each matrix in a single file
        - but all these files have random ids - how to find the feature representation of a specific segment?

- **Fix** problem with num_jobs was due to dev-version of lhotse

  - fix by using stable release and adding icsi.py to recipes folder and adding reference in **init**.py of recipes folder
    - currently using a symlink to the icsi.py in my own fork (located in HonoursProject folder)

# Tuesday, 25.01.22

Question from yesterday:

> but all these files have random ids - how to find the feature representation of a specific segment?

- **Answer**: The compute_and_store_features function returns a cutset that contains these ids mapped to the corresponding recording
  - this cutset needs to be passed to the dataset to create a dataset that has those information

**Question for meeting tomorrow**
What are the different dataloaders and dataset classes provided by Lhotse? Do you think I can use one of them?

# Wednesday, 26.01.22

- created a script to compare librosa audio loading from different offsets
  - `./laughter_detection/misc_scripts/check_librosa_loading_times.py`
  - found out that the loading time is proportional to the offset
    - for table, see `./Meeting_Notes/26_01_22/Meeting_26_01_22.pdf`

# Friday, 28.01.22

Microphone Abbreviation Meaning:

- ihm: Individual Head Microphone
- mdm: Medium Distance Microphone?
- sdm: Short Distance Microphone?

- Created pull request with added IHM channels

- Possibly fix this issue from worklog 22.01.22 and create a PR?:

```
- I find it rather confusing that the audio-dir is a required argument whereas the transcripts dir is not required

  - further, it seems like the audio dir could be the toplevel dir where the 'speech' and 'transcripts' folder reside in
    - I'd suggest that the audio-dir should be passed as the path to the 'speech' dir and the transcripts-dir should be passed as the path to the 'transcripts' dir
      - this would make it clearer for me
```

- stored chan_to_idx map on disk such that I can reload it from there and use the standard lhotse recipe without modifying it

  - before I modiefied the prepare_icsi function to also return the chan_to_idx map

- problem with num_workers seems to be present again

  - possibly look into this

- found way to store cutsets as .jsonl file

  - cutset.to_jsonl(<path>) / cutset.from_json(<path>)

- I ran into an issue with loading the cutset (with precomputed features) from disk instead of recomputing them

  - I wondered why the changes didn't work but I just reloaded the cutset from disk which had the wrong supervisions

- Created LadDataset (Laugh Activity Detection) which is a modiefied version of Lhotse's VadDataset
- created a supervision segment for each cut containing the information if the segment is laughter or not

# Monday, 31.01.22

- added load_data.py and lad.py to jrgillick repo and use it in a modiefied train script called `train_lhotse.py`

  - this train script makes use of the lhotse dataloader
  - needed to make some adjustments to make it work, e.g.:
    - changing the way signals and labels are loaded from a batch (using 'inputs' and 'is_laugh' keys)
    - adding a dimension to fit the given structure by Gillick et al.

- first version works: 100 Epochs ran sucessfully and in reasonable time
  - used same set (of 226 segements) for training and online validation (-> not good, just proof of work)

# Tuesday, 01.02.22

- use the following command to install lhotse at the state of my icis-update commit as pip package:
  `pip install git+https://github.com/lhotse-speech/lhotse@f1b66b8a8db2ea93e87dcb9db3991f6dd473b89d`

- changed name from 'val' to 'dev' to match lhotse's naming scheme

- got the following warnings when manifests for the whole ICSI data were created:

```
Parsing ICSI mrt files: 76it [00:05, 14.27it/s]
Preparing audio: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 75/75 [04:52<00:00,  3.90s/it]
Preparing supervision:  12%|██████████████████                                                                                                                                     | 9/75 [00:00<00:00, 87.80it/s]WARNING:root:Segment Bed002-2-271 exceeds recording duration. Not adding to supervisions.
WARNING:root:Segment Bmr019-4-128 exceeds recording duration. Not adding to supervisions.
WARNING:root:Segment Bmr028-0-293 exceeds recording duration. Not adding to supervisions.
Preparing supervision:  25%|██████████████████████████████████████                                                                                                                | 19/75 [00:00<00:00, 90.72it/s]WARNING:root:Segment Bro028-1-320 exceeds recording duration. Not adding to supervisions.
Preparing supervision:  39%|██████████████████████████████████████████████████████████                                                                                            | 29/75 [00:00<00:00, 65.56it/s]WARNING:root:Segment Bmr015-1-150 exceeds recording duration. Not adding to supervisions.
Preparing supervision:  55%|██████████████████████████████████████████████████████████████████████████████████                                                                    | 41/75 [00:00<00:00, 80.27it/s]WARNING:root:Segment Bro021-1-201 exceeds recording duration. Not adding to supervisions.
Preparing supervision:  71%|██████████████████████████████████████████████████████████████████████████████████████████████████████████                                            | 53/75 [00:00<00:00, 89.93it/s]WARNING:root:Segment Bmr031-5-156 exceeds recording duration. Not adding to supervisions.
WARNING:root:Segment Bro027-1-462 exceeds recording duration. Not adding to supervisions.
Preparing supervision:  84%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                        | 63/75 [00:00<00:00, 92.87it/s]WARNING:root:Segment Bmr005-4-353 exceeds recording duration. Not adding to supervisions.
Preparing supervision: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 75/75 [00:00<00:00, 87.43it/s]
Write cutset_dict to disk...
```

More expressive tqdm output when running multiple workers would be useful

- atm it just shows the number of workers that have completed which doesn't give a good estimate of how long one worker will take to finish

`compute_and_store_features` is a function of the CutSet class which calls the `extract_from_samples_and_store` of the extractor-instance passed to it

- The `FeatureExtractor` class is an abstract class that defines the structure of all FeatureExtractors

  - a feature extractor is initialised by passing a config object (a dataclass) to the init-function

- added script for downloading dummy data to repo for easier debugging on different machines

- the feature shape we are looking for is (44,128)

- why does it sometimes work with multiple jobs and sometimes it doesn't

  - in function `compute_and_store_features`
  - first thought it might be the Config that is passed to the Fbank extractor - but using the normal Fbank() without config still doesn't work

- some parameters in the FbankConfig are the problem
  - they seem to stop the execution when num_jobs > 1 is used
  - opened issue on github: https://github.com/lhotse-speech/lhotse/issues/559

# Wednesday, 02.02.21

- check what Fbank-extractor actually returns

  - plot the returned features and see if that seems reasonable
    - is the volume already transformed to logscale?

- What is multithreaded BLAS?
  > Basic Linear Algebra Subprograms (BLAS) is a specification that prescribes a set of low-level routines for performing common linear algebra operations such as vector addition, scalar multiplication, dot products, linear combinations, and matrix multiplication.

source: https://en.wikipedia.org/wiki/BLAS

- Gillick et al. used the whole dataset

  > Of 2435 total conversations, we partition the dataset into
  > 2159 for training, 119 for development, and 157 for testing,
  > using the same splits as Ryokai et al. [ 7]. Aggregate statistics
  > on the timing information across these partitions is summarized
  > in Table 1.

- how did that fit into memory? (-> ~180 hours of training data)

  - a sample rate of 8000 compared to 16000 means half the size for the same length of audio
  - Ondrey says the following and assumes they had a machine with large enough RAM to load all data into memory:
    > 180 hours of data takes aproximately 10GB of RAM (180 hours _ 3600 seconds per hour _ 8000 samples per second \* 2 bytes / 1024^3). With 16000 samples per second, which is your case, it would take 20GB of RAM. Does that make sense?

- some feature files (.llc) seem corrupted/invalid: Following error is thrown:
  - until now it's only been 1 file

> ValueError: lilcom: Length of string was too short
> [extra info] When calling: MonoCut.load_features(args=(MonoCut(id='train_14174', start=1414.59, duration=1.0, channel=7, supervisions=[SupervisionSegment(id='sup_train_14174', recording_id='Bro003', start=1414.59, duration=1.0, channel=7, text=None, language=None, speaker=None, gender=None, custom={'is_laugh': 1}, alignment=None)], features=Features(type='kaldi-fbank', num_frames=44, num_features=128, frame_shift=0.02275, sampling_rate=16000, start=1414.59, duration=1.0, storage_type='lilcom_files', storage_path='data/icsi/lhotse/feats', storage_key='2c3/2c3be5eb-0dde-4f3f-b812-a28406d2c44e.llc', recording_id=None, channels=7), recording=Recording(id='Bro003', sources=[AudioSource(type='file', channels=[0], source='data/icsi/speech/Bro003/chan0.sph'), AudioSource(type='file', channels=[1], source='data/icsi/speech/Bro003/chan1.sph'), AudioSource(type='file', channels=[2], source='data/icsi/speech/Bro003/chan2.sph'), AudioSource(type='file', channels=[3], source='data/icsi/speech/Bro003/chan3.sph'), AudioSource(type='file', channels=[4], source='data/icsi/speech/Bro003/chan4.sph'), AudioSource(type='file', channels=[5], source='data/icsi/speech/Bro003/chan6.sph'), AudioSource(type='file', channels=[6], source='data/icsi/speech/Bro003/chan7.sph'), AudioSource(type='file', channels=[7], source='data/icsi/speech/Bro003/chan8.sph'), AudioSource(type='file', channels=[8], source='data/icsi/speech/Bro003/chan9.sph')], sampling_rate=16000, num_samples=83565782, duration=5222.861375, transforms=None), custom=None),) kwargs={})

**Quick fix**: replace with other feature-representation: `cp 2c3979e9-fb7d-445a-a533-598ea2e9b363.llc 2c3be5eb-0dde-4f3f-b812-a28406d2c44e.llc`

- shouldn't make a difference in such a large dataset

- Gillick et al. do 100 000 steps

  - match that for the first model for comparison

- command used to run the train_lhotse on certain cluster node:
  `sbatch -w landonia12 --array=1-10%1 cluster_scripts/laughter_train.sh cluster_scripts/train_exp.txt --cpus-per-task=8 --gres=gpu4 --mem=32000`
  meaning: sbatch --array=1-${NR_EXPTS}%${MAX_PARALLEL_JOBS} mnist_arrayjob.sh $EXPT_FILE

  - MAX_PARALLEL_JOBS is set 1 because each excution will need the checkpoint data of the prior execution
  - I adapted `train_lhotse.py` in such a way that it runs 10 epochs and then exits
  - thus, 10\*10=100 epochs (split in 10 sequential subjobs) are run by the command above

- default setting is to apply gradients after each batch

  - can be changed by `--gradient_accumulation_steps`-flag

- renaming thesis?

  - still stating the motivation (original title) is important as it affected my choices
    - e.g. choice of the corpus

- Created lhotse-pull request to show a warning if num_jobs >1 and torch.get_num_threads>1
  - this addresses the issue I faced when using num_jobs>1

### 07.02.22

- realised that I need to adapt the laughter recognitions script to apply data transformation like lhotse-preprocssing

- check how Gillick et al. did the training

- the shape of the training features and the length of the final prediction units is related (for Gillick et al. 23ms)

  - the shape of features is (44,128) meaning that we have 44 frames for 1 second which equates to 1000/44 =~23ms which is the unit used for inference

- for our ICSI dataset we have 16000 frames and also created feature representation of shape (44,128) which means we should also use 23ms long units for inference

From: https://github.com/lhotse-speech/lhotse/issues/573

> Yes, it will still be efficient.
> I’d say if your training examples are fixed (eg ASR training, cut == supervision) then first cut, then extract features. If your training examples are dynamic (eg you’re sampling chunks for self supervised training or VAD etc) then it’s definitely better to extract, then cut.

- for training with subsampling it's probably best to extract first and then subsample since I won't know the exact cuts beforehand

  - and according to pzelasko's comment above the sampling will still be efficient

- should the inference generator return 1 or 44 frames?

- `eb05e753d56a9ba7e42e29b14a6dbc4bf21930d2`almost works but throws an error that tensors of different dimension cannot be stacked
  - I assume that this is due to missing padding for the last segment

### 08.02.22

- Analyse.py: temporary adjustment to account for changed TextGrid file names

  - originally the convention was and in the future will be 'chanN.TextGrid'
  - changed it temporarily to 'laughter_chanN.TextGrid' to make it work with current output

- added sys-argument to analyse.py to make passing the output directory easier

First analysis works - recall significantly higher:

```
  threshold precision              recall           valid_pred_laughs
                 mean    median      mean    median              mean  median
0       0.1  0.009289  0.009289  0.926635  0.926635            2342.0  2342.0
1       0.2  0.018489  0.018489  0.917084  0.917084            1897.0  1897.0
2       0.3  0.033927  0.033927  0.897891  0.897891            1103.0  1103.0
3       0.4  0.057323  0.057323  0.860460  0.860460             623.0   623.0
4       0.5  0.089995  0.089995  0.776256  0.776256             344.0   344.0
5       0.6  0.136735  0.136735  0.649098  0.649098             202.5   202.5
6       0.7  0.199281  0.199281  0.477080  0.477080             112.5   112.5
7       0.8  0.275497  0.275497  0.346883  0.346883              59.0    59.0
8       0.9  0.373416  0.373416  0.176774  0.176774              26.0    26.0
```

- very low precision, check the feature representation I'm using

  - also didn't apply augmentation
  - talk about best feature representation in the meeting tomorrow
    - and how to implement that in lhotse

### 09.02.22

- number of samples seems to be incorrect

  - realised this when trying to plot the audio

    - it uses `num_samples` as value for scaling the x-axis which is way to long in my case

  - `num_samples` wasn't the problem but the fact that 0 was always the start of the x-linspace used in the plot
    - changing that to `self.start` and the end to `self.start+self.duration` fixes the problem
      - Not sure about this: Need to check which expression for 'end' is correct
      - create a PR with this adjustment

- created visualise_feats notesbook to compare fbank and spectrogram extractor

  - also see email conversation with Ondrej
  - for now I'd say that using Fbank-Features seems reasonable
    - Fbank is 'kinda applied on top of Melspectrograms'
      - would have to look this up to have a clearer understanding

- got lots of input during the meeting today

  - see presentation and meeting notes

- paper to look at: https://asmp-eurasipjournals.springeropen.com/articles/10.1186/s13636-020-00176-2

_Idea for increasing performance:_

- use more 'speech data' because randomly sampling a small amount (like I do now) will likely yield
  lots of 'silence'-segment which is possibly why the performance looked like VAD to Ondrej

### 10.02.22

- trying to use GPU for computing features (on cluster)

  - checked with interactive session and the feature extraction doesn't seem to use GPU

- seems like Kaldi's Fbank computation on torchaudio doesn't support GPU

  - https://github.com/pytorch/audio/issues/613

- added timing the duration of training to train_lhotse.py

- more helpful output when audio file is missing -> lotse load_audio?

  - current error is not very descriptive

- `kaldifeat` can't be installed because the version of Cmake on the cluster is too old...

  - trying to install a new version of cmake from source
  - needed to install cmake version manually and add this to .bashrc
    `export PATH=/home/$USER/cmake-3.22.2/bin/:$PATH`

  - but still an error that CUDA_TOOLKIT_ROOT_DIR ins't found
    - possibly it's here: `./lib/python3.9/site-packages/conda/common/cuda.py`
    - possibly here: `/home/s1660656/miniconda3/pkgs/cudatoolkit-11.3.1-h2bc3f7f_2`

- trying to just run it on cpu with 16 parallel jobs

  - saves the time of debugging the requirements for using kaldifeat

- crated new version of data_dfs with 10 speech segments per laugh

  - want to see if that improves the performance
  - put normal data_dfs in `old_data_dfs` folder for now
    - need to find a better way to do this

- modified the load_data.py directly on the cluster to use 16 jobs and load the data from scratch disk

  - I hardcoded this for now, need to find a better implementation for this as well

- running the job on `landonia04` where all the speech data is copied to
  command run: `sbatch cluster_scripts/load_data_job.sh --cpus-per-task=16 --mem=32000`
  - need to commit the `load_data_job.sh` to repo as well

### 11.02.22

Issue with the comman yesterday night:

- used old manifest were the paths are on DFS and don't use scratch disk

  - thus the audio files weren't found

- need to make sure that when using a new audio_dir that manifest is recreated as well

- landonia04 went down so now I'm using landonia08

  - unfortunately now all the data has to be copied to that scratch disk again

- note that it's not 72hours \* 6 channels of speech
  - it's that amount of audio data, yes, but this contains lots of silence because participants don't talk simultaneously in one meeting
  - this also affects training when sampling speech segments at random

### 14.02.22

- transcript dir on scratch disk: `disk/scratch/s1660656/icsi/data`
- this is where the speech data is at on scratch disk: `disk/scratch/s1660656/icsi/data/speech/all`
- feats data location on disk: `/disk/scratch/s1660656/icsi/data/feats`

- trying to use this command to install kaldifeat (with adjusted versions) lead to conflicts in dependencies
  `conda install -c kaldifeat -c pytorch -c conda-forge kaldifeat python=3.9 cudatoolkit=11.3 pytorch`

  - using the default command from https://github.com/csukuangfj/kaldifeat seems to work:
    `conda install -c kaldifeat -c pytorch -c conda-forge kaldifeat python=3.8 cudatoolkit=11.1 pytorch=1.8.1`
    - to add torchaudio to this
      `conda install -c kaldifeat -c pytorch -c conda-forge kaldifeat python=3.8 cudatoolkit=11.1 pytorch=1.8.1 torchaudio`

- is it a problem to use a mixture of `conda install` and `pip install` ?

### 15.02.22

- program feat-compute code in a way that one error doesn't throw away all progress

- got this error when trying to compute features:

```
lhotse.audio.DurationMismatchError: Requested more audio (1.0s) than available (0.93s)
[extra info] When calling: Recording.load_audio(args=(Recording(id='Bmr019', sources=[AudioSource(type='file', channels=[0], source='data/icsi/speech/Bmr019/chan0.sph'), AudioSource(type='file', channels=[1], source='data/icsi/speech/Bmr019/chan1.sph'), AudioSource(type='file', channels=[2], source='data/icsi/speech/Bmr019/chan2.sph'), AudioSource(type='file', channels=[3], source='data/icsi/speech/Bmr019/chan3.sph'), AudioSource(type='file', channels=[4], source='data/icsi/speech/Bmr019/chan4.sph'), AudioSource(type='file', channels=[5], source='data/icsi/speech/Bmr019/chan5.sph'), AudioSource(type='file', channels=[6], source='data/icsi/speech/Bmr019/chan6.sph'), AudioSource(type='file', channels=[7], source='data/icsi/speech/Bmr019/chan7.sph'), AudioSource(type='file', channels=[8], source='data/icsi/speech/Bmr019/chan8.sph'), AudioSource(type='file', channels=[9], source='data/icsi/speech/Bmr019/chanA.sph'), AudioSource(type='file', channels=[10], source='data/icsi/speech/Bmr019/chanB.sph')], sampling_rate=16000, num_samples=57378219, duration=3586.1386875, transforms=None),) kwargs={'channels': 8, 'offset': -0.07, 'duration': 1.0})
[extra info] When calling: MonoCut.load_audio(args=(MonoCut(id='train_39309', start=-0.07, duration=1.0, channel=8, supervisions=[SupervisionSegment(id='sup_train_39309', recording_id='Bmr019', start=-0.07, duration=1.0, channel=8, text=None, language=None, speaker=None, gender=None, custom={'is_laugh': 0}, alignment=None)], features=None, recording=Recording(id='Bmr019', sources=[AudioSource(type='file', channels=[0], source='data/icsi/speech/Bmr019/chan0.sph'), AudioSource(type='file', channels=[1], source='data/icsi/speech/Bmr019/chan1.sph'), AudioSource(type='file', channels=[2], source='data/icsi/speech/Bmr019/chan2.sph'), AudioSource(type='file', channels=[3], source='data/icsi/speech/Bmr019/chan3.sph'), AudioSource(type='file', channels=[4], source='data/icsi/speech/Bmr019/chan4.sph'), AudioSource(type='file', channels=[5], source='data/icsi/speech/Bmr019/chan5.sph'), AudioSource(type='file', channels=[6], source='data/icsi/speech/Bmr019/chan6.sph'), AudioSource(type='file', channels=[7], source='data/icsi/speech/Bmr019/chan7.sph'), AudioSource(type='file', channels=[8], source='data/icsi/speech/Bmr019/chan8.sph'), AudioSource(type='file', channels=[9], source='data/icsi/speech/Bmr019/chanA.sph'), AudioSource(type='file', channels=[10], source='data/icsi/speech/Bmr019/chanB.sph')], sampling_rate=16000, num_samples=57378219, duration=3586.1386875, transforms=None), custom=None),) kwargs={})
```

After checking the `train_df` I found that there are `sub_starts` that are less than 0

- what if a segment is at the very end of the file

  - there could be the same issue because at the moment all segements are 1s long
    - if the laugh was shorter than 1s and at the very end we could go over the end of the audio
    - need to include padding for such cases -> that way allow for segments shorter than 1s
    - discarding all short laughs is not an option because there are 30723 in the training data
      - probably best to fix that and only create speech segments that are at least 1s?

- data_df dir isn't needed in training script anymore because it handles precomputed features

  - possibly need to use it when doing resampling

- started training experiment with metrics logging on MLP cluster (10 rounds with 10epochs each)
- started feature computation on local machine for 10-to-1 dataset

### 16.02.22

- refactored some of the existing thesis (up to line 212) using advice from dissertation writing workshop

  - trying to make it clearer and more concise

- changed `create_data_df.py` to

  - use the minimum of sub_sample_duration and segment duration as sub_sample_duration
    - this ensures that no subsample runs over the end of an audio track
  - only create speech segments >= 1s

- adapted `compute_features` to pad segments to a minimum of 1s. This ensures that all features have the same shape

  - this is necessary due to the change in create_data_dfs which now also outputs sub_samples with duration less than 1s

- command I use to compute features:
  `python compute_features.py --audio_dir data/icsi/speech/ --transcript_dir data/icsi/ --data_df_dir data/icsi/1_to_10_data_dfs/ --output_dir data/icsi/feats/1_to_10`

### 17.02.22

- copied over 1_to_10 feats from thinkstation to MLP cluster
- running training with 1_to_10 feats using:
  `sbatch -w landonia04 --array=1-10%1 cluster_scripts/train_laugh_job.sh cluster_scripts/train_exp.txt --cpus-per-task=8 --gres=gpu2 --mem=32000`

- ran into issue:
  `AssertionError: MonoCut db32b95d-8f99-4205-bd07-c1785467af0a: supervision sup_train_91863 has a mismatched recording_id (expected None, supervision has Buw001)`

Only padded recording not supervision
Problem was that lhotse loaded transcriptions recursively and thus, loaded some transcripts twice.

- setting up google colab env for running experiments

- refactoring code to use .env variables instead of argparse

  - easier to reproduce
  - easier to handle because one doesn't have to remember all the arguments for each execution

- refactor code structure
  Current structure has two repos

1. Forked from Gillick et al. (../laughter-detection-jrgillick-local)
   (removed **pycache** and ./env dirs)

```
laughter-detection-jrgillick-local/
├── checkpoints
│   ├── comparisons
│   ├── current
│   ├── icsi
│   └── in_use
├── cluster_scripts
│   ├── copy_checkpoints.sh
│   ├── eval_exp.txt
│   ├── eval_laugh_job.sh
│   ├── gen_eval_exp.py
│   ├── gen_train_exp.py
│   ├── remove_icsi_data.sh
│   ├── run_one.sh
│   ├── train_exp_bmr.txt
│   ├── train_exp_small.txt
│   ├── train_exp.txt
│   └── train_laugh_job.sh
├── compute_features.py
├── configs.py
├── data
│   ├── audioset
│   ├── icsi
│   └── rtf_samples
├── eval_output
│   └── short
├── lad.py
├── laugh_segmenter.py
├── laughter-detection-interactive.ipynb
├── LICENSE
├── load_data.py
├── models.py
├── notebooks
├── README.md
├── requirements.txt
├── scripts
│   ├── aggregate_switchboard_annotations.py
│   ├── aggregrate_audioset_annotations.py
│   ├── audio_set_loading.py
│   ├── download_audioset_metadata.sh
│   ├── download_audio_set_mp3s.py
│   ├── Evaluation
│   ├── make_switchboard_text_dataset.py
│   ├── store_all_audioset_laughter_audio.py
│   ├── store_all_icsi_audio.py
│   └── store_all_switchboard_audio.py
├── segment_laughter.py
├── train_lhotse.py
├── train.py
└── utils
    ├── audio_utils.py
    ├── data_loaders.py
    ├── dataset_utils.py
    ├── __pycache__
    ├── text_utils.py
    └── torch_utils.py

23 directories, 45 files

```

2. the code used for evaluation and analysing outputs (../git_repo/laughter_detection)

```
git_repo/laughter_detection/
├── analyse.py
├── config.py
├── create_data_df.py
├── data
│   ├── data_dfs
│   └── icsi
├── eval_output
│   ├── Bmr021
│   └── Bns001
├── lad.py
├── misc_scripts
│   └── check_librosa_loading_times.py
├── output_processing
│   ├── break.wav
│   ├── concat_laughs.sh
│   └── laughs_to_wav.py
├── preprocess.py
├── results -> /afs/inf.ed.ac.uk/user/s16/s1660656/Documents/Semester_7/Honours_Project/results
├── transcript_parsing
│   ├── data
│   ├── filter_all_laughs.sh
│   ├── filter_laugh_only.sh
│   ├── parse.py
│   ├── __pycache__
│   └── xpath_command.txt
├── utils.py
└── visualise.py
```

Created new repo with all these files in one repo:

- https://github.com/LasseWolter/laughter-detection-icsi

  - can be structured better: currently lots of file at root level

- updated lhotse-icsi-recipe (Bugfixes)

  - new dir structure created by download_icsi wasn't used by the prepare_icsi
  - zip file in target dir wasn't found

- updated compute_features script to use variables from an .env file instead of argparse

From Lhotse docs(https://lhotse.readthedocs.io/en/latest/api.html#lhotse-s-feature-extractors):
might be useful for realtime inference

```
Update January 2022: These modules now expose a new API function called “online_inference” that may be used to compute the features when the audio is streaming. The implementation is stateless, and passes the waveform remainders back to the user to feed them to the modules once new data becomes available. The implementation is compatible with JIT scripting via TorchScript.
```

Created google collab project to compute features:

- https://colab.research.google.com/drive/1-e6UOsRg47-Uh_mGvmxftTiNO9miIYt-#scrollTo=-yPsJSA9g7UD
- using GPU with `kaldifeat` doesn't throw an error but GPU utilisation doesn't go up
  - took 8minutes for 2 meetings
    - 8\*(75meeeting/2) = 300minutes = 5h
      - that doesn't seem much faster than running it on my CPU
      - what's the reason for this?
        - possibly I don't need to spend time on debugging the GPU stuff on the cluster then?

### 18.02.22

- I'd find it helpful if you could see the tqdm-bar for parallel computation as well

  - this allows to keep track of the computations
    - isn't it normal that you would do such long feature computations at once (e.g. 5hours)?
    - if you don't have the individual progress bars you don't know the progress at all.

Comments from Ondrej today:

- feature structure: 100 frames and 40 bins
  - check Gillick et al. code for that
    - how do I need to change their structure
- mix audio with noise - lhotse has an implementation for it
- change the speech sampling according to the transcriptions
- Vad: how does that work in Lhotse
- check the false positives - see if they are mostly speech (VAD) or silence
- precision/recall curve to show how the threshold could be changed

- Refactored parse.py to also filter out speech and noise
  - now I get 8720 instead of 8420 laughter snippets
    - not a problem. Just make sure that it's stated consistently in the thesis

Questions about ResNet Structure:

- what does AvgPool2d(4) do?
- what do the dropout stages do?
- what does the batchnorm do?

### 19.02.22

- first thing to do: Create feature representation of the whole ICSI corpus using 100x40 Fbank representation
