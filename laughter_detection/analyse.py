import textgrids
import os
import pandas as pd
from matplotlib import pyplot as plt
import subprocess
import portion as P

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
            part_id = parse.chan_to_part[params['meeting_id']
                                         ][params['chan_id']]
            seg_length = interval.xmax - interval.xmin
            interval_list.append([params['meeting_id'], part_id, params['chan_id'], interval.xmin,
                                  interval.xmax, seg_length, params['threshold'], str(interval.text)])
    return interval_list


def textgrid_to_df(file_path):
    tot_list = []
    for filename in os.listdir(file_path):
        if filename.endswith('.TextGrid'):
            full_path = os.path.join(file_path, filename)
            params = get_params_from_path(full_path)
            tot_list += textgrid_to_list(full_path,
                                         params)

    cols = ['Meeting', 'ID', 'Channel', 'Start',
            'End', 'Length', 'Threshold', 'Type']
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
    thr = params_list[-3].replace('t_', '')
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
# PREPROCESSING
##################################################
LAUGH_INDEX = {}
MIXED_LAUGH_INDEX = {}
MIN_LENGTH = 0.2

# Factor determines the frame size
# 100 means 1sec gets split into 100 10ms intervals
FRAME_FACTOR = 100


def remove_breath_laugh(df):
    """
    Remove all events of type breath laugh
    after manual evaluation of samples breath-laughs are maybe
    suitable for our project
    TODO: Decide on this
    """
    return df[df['Type'] != 'breath-laugh']


def sec_to_frame(time):
    """ Return a time expressed as an integer representing the frame number """
    return round(time * FRAME_FACTOR)


def create_laugh_index(df):
    global LAUGH_INDEX
    """
    Creates a laugh_index with all transcribed laughter events
    per particpant per meeting
    dict structure:
    {
        meeting_id: {
            tot_laugh_len: INT
            part_id: [P.closed(start,end), P.closed(start,end)]
            part_id: [P.closed(start,end), P.closed(start,end)]
        }
        ...
    }
    """
    meeting_groups = df.groupby(['Meeting'])

    for meeting_id, meeting_df in meeting_groups:
        LAUGH_INDEX[meeting_id] = {}
        LAUGH_INDEX[meeting_id]['tot_laugh_len'] = 0
        LAUGH_INDEX[meeting_id]['tot_laugh_events'] = 0

        # Ensure rows are sorted by 'Start'-time in ascending order
        part_groups = meeting_df.sort_values('Start').groupby(['ID'])
        for part_id, part_df in part_groups:
            LAUGH_INDEX[meeting_id][part_id] = P.empty()
            for _, row in part_df.iterrows():
                # If the length is shorter than min_length passed to detection, skip
                if(row['Length'] < MIN_LENGTH):
                    continue
                start = sec_to_frame(row['Start'])
                end = sec_to_frame(row['End'])
                LAUGH_INDEX[meeting_id][part_id] = LAUGH_INDEX[meeting_id][part_id] | P.closed(
                    start, end)
                LAUGH_INDEX[meeting_id]['tot_laugh_len'] += row['Length']
                LAUGH_INDEX[meeting_id]['tot_laugh_events'] += 1


def create_mixed_laugh_index(df):
    global MIXED_LAUGH_INDEX
    """
    Creates a mixed_laugh_index with all transcribed laughter events
    occurring next to other sounds per particpant per meeting
    dict structure:
    {
        meeting_id: {
            tot_laugh_len: INT
            part_id: [P.closed(start,end), P.closed(start,end)]
            part_id: [P.closed(start,end), P.closed(start,end)]
        }
        ...
    }
    """
    meeting_groups = df.groupby(['Meeting'])
    for meeting_id, meeting_df in meeting_groups:
        MIXED_LAUGH_INDEX[meeting_id] = {}
        MIXED_LAUGH_INDEX[meeting_id]['tot_laugh_len'] = 0

        # Ensure rows are sorted by 'Start'-time in ascending order
        part_groups = meeting_df.sort_values('Start').groupby(['ID'])
        for part_id, part_df in part_groups:
            MIXED_LAUGH_INDEX[meeting_id][part_id] = P.empty()
            for _, row in part_df.iterrows():
                start = sec_to_frame(row['Start'])
                end = sec_to_frame(row['End'])
                MIXED_LAUGH_INDEX[meeting_id][part_id] = MIXED_LAUGH_INDEX[meeting_id][part_id] | P.closed(
                    start, end)
                MIXED_LAUGH_INDEX[meeting_id]['tot_laugh_len'] += row['Length']

##################################################
# ANALYSE
##################################################


def laugh_match(pred_laugh, meeting_id, part_id):
    '''
    Checks if a predicted laugh events for a particular meeting overlap with the
    transcribed laugh events for that meeting
    Input: P.Interval (Union of all laughter intervals for particular participant in one meeting)
    Returns: (time_predicted_correctly, time_predicted_falsely)
    '''

    if part_id in MIXED_LAUGH_INDEX[meeting_id].keys():
        # Remove laughter occurring in mixed settings because we don't evaluate them
        pred_laugh = pred_laugh - MIXED_LAUGH_INDEX[meeting_id][part_id]

    pred_length = len(list(P.iterate(pred_laugh, step=1)))/float(FRAME_FACTOR)

    if part_id not in LAUGH_INDEX[meeting_id].keys():
        # No laugh events transcribed for this participant - all false
        return(0, pred_length)

    # Get correct
    match = LAUGH_INDEX[meeting_id][part_id] & pred_laugh
    correct = len(list(P.iterate(match, step=1)))/float(FRAME_FACTOR)
    incorrect = pred_length - correct
    return(correct, incorrect)


def eval_preds(meeting_df, print_stats=False):
    """
    Calculate evaluation metrics for a particular meeting for a certain parameter set
    """

    # If there are no predictions, return []
    if meeting_df.size == 0:
        return []

    meeting_id = meeting_df.iloc[0]['Meeting']
    threshold = meeting_df.iloc[0]['Threshold']

    tot_predicted_time, tot_corr_pred_time, tot_incorr_pred_time = 0, 0, 0
    tot_transc_laugh_time = LAUGH_INDEX[meeting_id]['tot_laugh_len']

    group_by_part = meeting_df.groupby(['ID'])

    for part_id, part_df in group_by_part:
        part_pred_frames = P.empty()
        for _, row in part_df.iterrows():

            # Create interval representing predicted laughter frames
            pred_start_frame = sec_to_frame(row['Start'])
            pred_end_frame = sec_to_frame(row['End'])
            pred_laugh = P.closed(pred_start_frame, pred_end_frame)

            # Append interval to total predicted frames for this meeting
            part_pred_frames = part_pred_frames | pred_laugh
            # Old Version
            # tot_predicted_time += row['Length']

        corr, incorr = laugh_match(part_pred_frames, meeting_id, part_id)
        tot_corr_pred_time += corr
        tot_incorr_pred_time += incorr

    # New version
    tot_predicted_time = tot_corr_pred_time + tot_incorr_pred_time
    # If there are no predictions for this meeting -> precision=1
    if tot_predicted_time == 0:
        prec = 1
    else:
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
# OTHER
##################################################


def laugh_df_to_csv(df):
    """
    Used to generate .csv file of a subset of laughter events (e.g. breath-laughs)
    e.g. for generating .wav-files using
    ./output_processing/laughs_to_wav.py from this .csv
    """
    df = df[df['Type'] == 'breath-laugh']
    df.to_csv('breath_laugh.csv')


def create_evaluation_df(path, use_cache=False):
    """
    Creates a dataframe summarising evaluation metrics per meeting for each parameter-set
    """
    if not use_cache or not os.path.isfile('.cache/eval_df.csv'):
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
            raise Exception(
                f'List returned by eval_preds() has wrong length. Expected length: {len(cols)}. Found: {len(all_evals[0])}.')
        eval_df = pd.DataFrame(all_evals, columns=cols)
        if not os.path.isdir('.cache'):
            subprocess.run(['mkdir', '.cache'])
        eval_df.to_csv('.cache/eval_df.csv')
    else:
        print("-----------------------------------------")
        print("NO NEW EVALUATION - USING CACHED VERSION")
        print("-----------------------------------------")
        eval_df = pd.read_csv('.cache/eval_df.csv')

    return eval_df


def calc_sum_stats(eval_df):
    """
    Calculate summary statistics across all meetings per parameter-set
    """
    sum_stats = eval_df.groupby('threshold')[
        ['precision', 'recall']].agg(['mean']).reset_index()
    # Filter thresholds
    # sum_stats = sum_stats[sum_stats['threshold'].isin([0.2,0.4,0.6,0.8])]
    return sum_stats


def plot_prec_recall_curve(stats):
    """
    Input: statistics dataframe across all meetings
    Plots a precision recall curve for the given dataframe
    """
    if 'recall' not in stats.columns or 'precision' not in stats.columns:
        raise LookupError(
            f'Missing precision or recall column in passed dataframe. Found columns: {stats.columns}')
    plt.plot(stats['recall'], stats['precision'], 'b--')
    plt.plot(stats['recall'], stats['precision'], 'ro')
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.show()


def main():
    transc_laughs = parse.laugh_only_df
    # Preprocessing applied before laugh_indices are created
    #transc_laughs = remove_breath_laugh(transc_laughs)
    create_laugh_index(transc_laughs)
    create_mixed_laugh_index(parse.mixed_laugh_df)
    outputs_path = './output_processing/outputs/'
    eval_df = create_evaluation_df(outputs_path)

    sum_stats = calc_sum_stats(eval_df)
    print(sum_stats)

    acc_len = 0
    acc_ev = 0
    for meeting in LAUGH_INDEX.keys():
        acc_len += LAUGH_INDEX[meeting]['tot_laugh_len']
        acc_ev += LAUGH_INDEX[meeting]['tot_laugh_events']
        # print(f"tot len: {laugh_index[meeting]['tot_laugh_len']}")
        # print(f"num of events: {laugh_index[meeting]['tot_laugh_events']}")

    print(f'tot length: {acc_len}')
    print(f'tot events: {acc_ev}')


if __name__ == "__main__":
    # pred_laughs = textgrid_to_df('./output_processing/outputs/Bro017/t_0.4/l_0.2')
    # pred_laughs.to_csv('Bro017_0.4.csv')
    main()
