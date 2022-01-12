# Log:

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

# Wednesday
- Continued training
  - checkpointing works but training is really slow
    - only 30 batches in 2 hours (960 segments -> 8s audio processed per minute)
      - discussed this issue in the meeting for more details see meeting notes for 12.01.22

- Added another parameter to train.py: data_dfs_dir 
  - allows to specify the dataframes used for train/val/test split