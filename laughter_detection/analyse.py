import textgrids
import os
import pandas as pd
from matplotlib import pyplot as plt
import subprocess

from transcript_parsing import parse

##################################################
# PARSE TEXTGRID
##################################################


def textgrid_to_list(full_path, params):
    # There are more recorded channels than participants
    # thus, not all channels are mapped to a participant
    # We focus on those that are mapped to a participant
    if params['chan_id'] not in parse.chan_to_part[params['meeting_id']].keys():
        return []
    interval_list = []
    grid = textgrids.TextGrid(full_path)
    for interval in grid['laughter']:
        # TODO: Change for breath laugh?!
        if str(interval.text) == 'laugh':
            part_id = parse.chan_to_part[params['meeting_id']][params['chan_id']]
            interval_list.append([params['meeting_id'], part_id, params['chan_id'], interval.xmin,
                                  interval.xmax, interval.xmax-interval.xmin, params['threshold'], str(interval.text)])
    return interval_list


def textgrid_to_df(file_path):
    tot_list = []
    for filename in os.listdir(file_path):
        if filename.endswith('.TextGrid'):
            full_path = os.path.join(file_path, filename)
            params = get_params_from_path(full_path)
            tot_list += textgrid_to_list(full_path,
                                         params)

    cols = ['Meeting', 'ID', 'Channel', 'Start', 'End', 'Length', 'Threshold', 'Type']
    df = pd.DataFrame(tot_list, columns=cols)
    return df


def get_params_from_path(path):
    '''
    Input: path
    Output: dict of parameters
    '''
    params = {}
    path = os.path.normpath(path)
    # First cut of .TextGrid
    # then split for to get parameters which are given by dir-names
    params_list = path.replace('.TextGrid', '').split('/')
    chan_id = params_list[-1]
    # Check if filename follows convention -> 'chanN.TextGrid'
    if not chan_id.startswith('chan'):
        raise NameError(
            "Did you follow the naming convention for channel .TextGrid-files -> 'chanN.TextGrid'")

    params['chan_id'] = chan_id
    params['min_length'] = params_list[-2]

    # Strip the 't_' prefix and turn threshold into float
    thr = params_list[-3].replace('t_','')
    params['threshold'] = float(thr) 
    meeting_id = params_list[-4]
    # Check if meeting ID is valid -> B**NNN
    if not len(meeting_id) == 6:  # All IDs are 6 chars long
        raise NameError(
            "Did you follow the required directory structure? all chanN.TextGrid files \
            need to be in a directory with its meeting ID as name -> e.g. B**NNN")

    params['meeting_id'] = meeting_id
    return params


##################################################
# ANALYSE
##################################################
laugh_index = {}


def create_laugh_index(df):
    global laugh_index
    """
    Creates a laugh_index with all transcribed laughter events
    per particpant per meeting
    dict structure:
    { 
        meeting_id: {
            tot_laugh_len: INT
            part_id: [(start,end), (start,end)]
            part_id: [(start,end), (start,end)]
        }
        ...
    }
    """
    meeting_groups = df.groupby(['Meeting'])
    for meeting_id, meeting_df in meeting_groups:
        laugh_index[meeting_id] = {}
        laugh_index[meeting_id]['tot_laugh_len'] = 0

        # Ensure rows are sorted by 'Start'-time in ascending order
        part_groups = meeting_df.sort_values('Start').groupby(['ID'])
        for part_id, part_df in part_groups:
            laugh_index[meeting_id][part_id] = []
            for index, row in part_df.iterrows():
                laugh_index[meeting_id][part_id].append(
                    (row['Start'], row['End']))
                laugh_index[meeting_id]['tot_laugh_len'] += row['Length']

    return laugh_index


def laugh_match(pred_laugh):
    '''
    Checks if a predicted laugh event overlaps with a transcribed laugh event
    Returns: (time_predicted_correctly, time_predicted_falsely)

    TODO: Case not considered: If the predicted event spans across two transcribed events
            ->  this probably isn't very likely because transcribed events rarely occur back to back 
                but this should still be accouted for
    '''
    meeting_id = pred_laugh['Meeting']
    part_id = pred_laugh['ID']
    pred_start = pred_laugh['Start']
    pred_end = pred_laugh['End']
    pred_length = pred_laugh['Length']
    
    if part_id not in laugh_index[meeting_id].keys():
        # No laugh events transcribed for this participant - all false
        return(0, pred_length)
    for laugh in laugh_index[meeting_id][part_id]:
        transc_start = laugh[0]
        transc_end = laugh[1]
        if pred_start > transc_end:
            continue
        # Predicted start is smaller than transcribed end of this event
        # Check if predicted start is also larger than transcribed start
        # Then predicted start lies within the transcribed interval
        elif pred_start >= transc_start:
            if pred_end <= transc_end:
                # Whole prediction lies within transcription bounds, all correct
                return (pred_length, 0)
            else:
                # Predicted Start lies within bounds, end doesn't
                return (transc_end - pred_start, pred_end - transc_end)
        # Predicted start doesn't lie within bounds of transcribed event
        # Check if predicted end does
        elif pred_end < transc_start:
            # whole prediction lies in front of transcription -> check next transcribed interval
            continue
        elif transc_start < pred_end:
            if pred_end < transc_end: # predicted end lies within interval, start doesn't
                return (pred_end - transc_start, transc_start - pred_start)
            else: # Prediction spans across whole transcription and more
                return (transc_end - transc_start, (transc_start-pred_start + pred_end-transc_end))

    # Didn't match any transcribed interval - all false
    return(0, pred_length)

def eval_preds(df, print_stats=False):
    # If there are no predictions, return []
    if df.size == 0:
        return [] 

    meeting_id = df.iloc[0]['Meeting']
    threshold = df.iloc[0]['Threshold']
    tot_predicted_time = 0
    tot_corr_pred_time = 0
    tot_incorr_pred_time = 0
    tot_transc_laugh_time = laugh_index[meeting_id]['tot_laugh_len']
    for _, row in df.iterrows():
        tot_predicted_time += row['Length']
        corr, incorr = laugh_match(row)
        tot_corr_pred_time += corr
        tot_incorr_pred_time += incorr
    
    prec = tot_corr_pred_time/tot_predicted_time
    recall = tot_corr_pred_time/tot_transc_laugh_time


    if(print_stats):
        print(f'total transcribed time: {tot_transc_laugh_time:.2f}\n'
                f'total predicted time: {tot_predicted_time:.2f}\n'
                f'correct: {tot_corr_pred_time:.2f}\n'
                f'incorrect: {tot_incorr_pred_time:.2f}\n')

        print(f'Meeting: {meeting_id}\n'
              f'Threshold: {threshold}\n'
              f'Precision: {prec:.4f}\n'
              f'Recall: {recall:.4f}\n')

    return[meeting_id, threshold, prec, recall]

##################################################
# PREPROCESSING
##################################################
def remove_breath_laugh(df):
    """
    Remove all events of type breath laugh
    after manual evaluation of samples breath-laughs are maybe
    suitable for our project
    TODO: Decide on this
    """
    return df[df['Type'] != 'breath-laugh']


##################################################
# OTHER
##################################################

# Used to generate .csv file of a subset of laughter events
# e.g. for generation of corresponding .wav-files using
# ./output_processing/laughs_to_wav.py
def breath_laugh_to_csv(df):
    df = df[df['Type'] == 'breath-laugh']
    df.to_csv('breath_laugh.csv')

def create_evaluation_df(path, no_cache=False):
    """
    Creates a dataframe summarising evaluation metrics per meeting for each parameter-set
    """ 
    if no_cache or not os.path.isfile('.cache/eval_df.csv'):
        all_evals = []
        for meeting in os.listdir(path):
            print(f'Evaluating meeting {meeting}...')
            meeting_path = os.path.join(path, meeting)
            for threshold in os.listdir(meeting_path):
                threshold_dir = os.path.join(meeting_path, threshold)
                for min_length in os.listdir(threshold_dir):
                    textgrid_dir = os.path.join(threshold_dir, min_length)
                    pred_laughs = textgrid_to_df(textgrid_dir)
                    all_evals.append(eval_preds(pred_laughs))
        
        cols = ['meeting', 'threshold', 'precision', 'recall']
        if len(cols) != len(all_evals[0]):
            raise Exception(f'List returned by eval_preds() has wrong length. Expected length: {len(cols)}. Found: {len(all_evals[0])}.')
        eval_df = pd.DataFrame(all_evals, columns=cols)
        if not os.path.isdir('.cache'):
            subprocess.run(['mkdir', '.cache'])
        eval_df.to_csv('.cache/eval_df.csv')
    else:
        eval_df = pd.read_csv('.cache/eval_df.csv')

    return eval_df

def calc_sum_stats(eval_df):
    """
    Calculate summary statistics across all meetings per parameter-set 
    """
    sum_stats = eval_df.groupby('threshold')[['precision','recall']].agg(['mean']).reset_index()
    # Filter thresholds
    #sum_stats = sum_stats[sum_stats['threshold'].isin([0.2,0.4,0.6,0.8])]
    return sum_stats

def plot_prec_recall_curve(stats):
    """
    Input: statistics dataframe across all meetings
    Plots a precision recall curve for the given dataframe
    """
    if 'recall' not in stats.columns or 'precision' not in stats.columns:
        raise LookupError(f'Missing precision or recall column in passed dataframe. Found coloumns: {stats.columns}')
    plt.plot(stats['recall'], stats['precision'], 'b--')
    plt.plot(stats['recall'], stats['precision'], 'ro')
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.show()

def main():
    transc_laughs = parse.laugh_only_df
    # transc_laughs = remove_breath_laugh(transc_laughs)
    create_laugh_index(transc_laughs)

    outputs_path = './output_processing/outputs/'
    eval_df = create_evaluation_df(outputs_path, no_cache=False)

    sum_stats = calc_sum_stats(eval_df)
    print(sum_stats)
    
    

if __name__ == "__main__":
    #pred_laughs = textgrid_to_df('./output_processing/outputs/Bro017/t_0.4/l_0.2')
    #pred_laughs.to_csv('Bro017_0.4.csv')
    main()
