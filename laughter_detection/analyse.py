from matplotlib.ticker import MaxNLocator
import textgrids
import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import subprocess
import numpy as np
import portion as P

import config as cfg
from transcript_parsing import parse
import preprocess as prep
import utils

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

    cols = ['meeting_id', 'part_id', 'chan', 'start',
            'end', 'length', 'threshold', 'type']
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
# ANALYSE
##################################################


def laugh_match(pred_laugh, meeting_id, part_id):
    '''
    Checks if a predicted laugh events for a particular meeting overlap with the
    transcribed laugh events for that meeting
    Input: P.Interval (Union of all laughter intervals for particular participant in one meeting)
    Returns: (time_predicted_correctly, time_predicted_falsely)
    '''

    if part_id in prep.invalid_index[meeting_id].keys():
        # Remove laughter occurring in mixed settings because we don't evaluate them
        pred_laugh = pred_laugh - prep.invalid_index[meeting_id][part_id]

    pred_length = utils.to_sec(utils.p_len(pred_laugh))

    if part_id not in prep.laugh_index[meeting_id].keys():
        # No laugh events transcribed for this participant - all false
        return(0, pred_length)

    # Get correct
    match = prep.laugh_index[meeting_id][part_id] & pred_laugh
    correct = utils.to_sec(utils.p_len(match))
    incorrect = pred_length - correct
    return(correct, incorrect)


def eval_preds(meeting_df, print_stats=False):
    """
    Calculate evaluation metrics for a particular meeting for a certain parameter set
    """

    # If there are no predictions, return []
    if meeting_df.size == 0:
        return []

    meeting_id = meeting_df.iloc[0]['meeting_id']
    threshold = meeting_df.iloc[0]['threshold']

    tot_predicted_time, tot_corr_pred_time, tot_incorr_pred_time = 0, 0, 0
    tot_transc_laugh_time = prep.laugh_index[meeting_id]['tot_len']
    num_of_tranc_laughs = parse.laugh_only_df[parse.laugh_only_df['meeting_id']
                                              == meeting_id].shape[0]
    num_of_pred_laughs = meeting_df.shape[0]

    # Count by
    num_of_VALID_pred_laughs = 0

    group_by_part = meeting_df.groupby(['part_id'])

    for part_id, part_df in group_by_part:
        part_pred_frames = P.empty()
        for _, row in part_df.iterrows():

            # Create interval representing predicted laughter frames
            pred_start_frame = utils.to_frames(row['start'])
            pred_end_frame = utils.to_frames(row['end'])
            pred_laugh = P.closed(pred_start_frame, pred_end_frame)

            # If the there are no invalid frames for this participant
            # or if the laugh frame doesn't lie in an invalid section -> increase num of valid predictions
            if part_id not in prep.invalid_index[meeting_id].keys() or \
                    not prep.invalid_index[meeting_id][part_id].contains(pred_laugh):
                num_of_VALID_pred_laughs += 1

            # Append interval to total predicted frames for this participant
            part_pred_frames = part_pred_frames | pred_laugh
            # Old Version
            # tot_predicted_time += row['length']

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
    if tot_transc_laugh_time == 0:
        # If there is no positive data, recall doesn't mean anything -> thus, NaN
        recall = float('NaN')
    else:
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

    return[meeting_id, threshold, prec, recall, round(tot_corr_pred_time, 2), round(tot_predicted_time, 2),
           round(tot_transc_laugh_time, 2), num_of_pred_laughs, num_of_VALID_pred_laughs, num_of_tranc_laughs]


def create_evaluation_df(path, use_cache=False):
    """
    Creates a dataframe summarising evaluation metrics per meeting for each parameter-set
    """
    if not use_cache or not os.path.isfile('.cache/eval_df.csv'):
        all_evals = []
        for meeting in os.listdir(path):
            #print(f'Evaluating meeting {meeting}...')
            meeting_path = os.path.join(path, meeting)
            for threshold in os.listdir(meeting_path):
                threshold_dir = os.path.join(meeting_path, threshold)
                for min_length in os.listdir(threshold_dir):
                    textgrid_dir = os.path.join(threshold_dir, min_length)
                    pred_laughs = textgrid_to_df(textgrid_dir)
                    all_evals.append(eval_preds(pred_laughs))

        cols = ['meeting', 'threshold', 'precision', 'recall',
                'corr_pred_time', 'tot_pred_time', 'tot_transc_time', 'num_of_pred_laughs', 'valid_pred_laughs', 'num_of_transc_laughs']
        if len(cols) != len(all_evals[0]):
            raise Exception(
                f'List returned by eval_preds() has wrong length. Expected length: {len(cols)}. Found: {len(all_evals[0])}.')
        eval_df = pd.DataFrame(all_evals, columns=cols)
        if not os.path.isdir('.cache'):
            subprocess.run(['mkdir', '.cache'])
        eval_df.to_csv('.cache/eval_df.csv', index=False)
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
    # Now aggregate stats across meetings
    # sum_stats = eval_df.groupby('threshold')[
    #     ['precision', 'recall']].mean().reset_index()

    sum_stats = eval_df.groupby('threshold')[
        ['precision', 'recall', 'valid_pred_laughs']].agg(['mean', 'median']).reset_index()
    # Filter thresholds
    # sum_stats = sum_stats[sum_stats['threshold'].isin([0.2,0.4,0.6,0.8])]
    return sum_stats


##################################################
# PLOTS
##################################################
def plot_prec_recall_curve(df):
    """
    Input: statistics dataframe across all meetings
    Plots a precision recall curve for the given dataframe
    """
    if 'recall' not in df.columns or 'precision' not in df.columns:
        raise LookupError(
            f'Missing precision or recall column in passed dataframe. Found columns: {df.columns}')
    plt.plot(df['recall'], df['precision'], 'b--')
    plt.plot(df['recall'], df['precision'], 'ro')
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.show()


def plot_aggregated_laughter_length_dist(df, threshold, save_dir=''):
    '''
    Plots histogram of aggregated laughter lengths for predicted and transcribed events
    Only predictions with the given threshhold are considered
        - helps compare how predictions and transcriptions compare in their length 
    '''
    fig, axs = plt.subplots(3, 1, figsize=(6, 8))

    df = df[df['threshold'] == threshold]

    print(axs)
    cols = ['tot_pred_time', 'tot_transc_time']

    for col in cols:
        sns.distplot(x=df[col], ax=axs[0], label=col,
                     bins=range(0, 1000, 50), kde=False)
    axs[0].set_xlim([0, 1000])
    axs[0].grid()

    for col in cols:
        sns.distplot(x=df[col], ax=axs[1], label=col,
                     bins=range(0, 500, 10), kde=False)
    axs[1].set_xlim([0, 500])
    axs[1].grid()

    for col in cols:
        sns.distplot(x=df[col], ax=axs[2], label=col,
                     bins=range(0, 60, 1), kde=False)
    axs[2].set_xlim([0, 60])
    axs[2].set_xlabel('Aggregated Length [s]')
    axs[2].grid()

    # Add big axis and hide ticks and tick lables on this big axis
    # only used to create shared y-label
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False,
                    bottom=False, left=False, right=False)

    fig.legend(cols, loc='upper right')
    plt.ylabel('Frequency')
    pred_median, transc_median = df[cols].median().round(2)

    tot_pred_meetings = df.shape[0]
    tot_meetings = 75
    plt.text(
        -0.1, 1.015, f'av-meeting-length:{56}min\nav-pred-time:{pred_median}s\nav-transc-time:{transc_median}s'
        f'\nMeetings containing\nlaughter predictions:{tot_pred_meetings}/{tot_meetings}')
    plt.title(
        f'Aggregated length of\n laughter per meeting\nthreshold: {threshold}')

    if save_dir != '':
        path = os.path.join(save_dir, f'agg_length_dist_{threshold}.png')
        plt.savefig(path)
    plt.show()


def plot_agg_pred_time_ratio_dist(df, threshold, save_dir=""):
    """
    Plots a distribution of following ratio per meeting:
        total_predicted_laughter_time / total_transcribed_laughter_time
    """
    df = df[df['threshold'] == threshold]
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    cols = ['tot_pred_time', 'tot_transc_time']
    rate_df = (df[cols[0]] / df[cols[1]]) * 100
    sns.histplot(rate_df, alpha=0.5, ax=ax)
    ax.set_xlabel('Ratio (pred_time/tranc_time)[%]')
    ax.set_ylabel('Frequency')

    # Only display integers on y-axis
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis='y')

    tot_pred_meetings = df.shape[0]
    tot_meetings = 75

    # Calculate and display median in plot
    median = round(rate_df.median(), 2)
    mean = round(rate_df.mean(), 2)
    plt.vlines(median, 0, 3, colors=['r'],
               linestyles=['dashed'], label='median')
    plt.vlines(mean, 0, 3, colors=['b'],
               linestyles=['dashed'], label='mean')

    plt.legend()

    # Calculate recall
    stats = calc_sum_stats(df)
    prec_meadian = round(stats['precision'].loc[0]['median'] * 100, 2)
    prec_mean = round(stats['precision'].loc[0]['mean'] * 100, 2)

    recall_median = round(stats['recall'].loc[0]['median'] * 100, 2)
    recall_mean = round(stats['recall'].loc[0]['mean'] * 100, 2)

    plt.figtext(
        0.01, 0.02, f'MEDIAN:\nRatio: {median}%\nRecall: {recall_median}%\nPrecision: {prec_meadian}%')
    plt.figtext(
        0.17, 0.02, f'| MEAN:\n| Ratio: {mean}%\n| Recall: {recall_mean}%\n| Precision: {prec_mean}%')
    plt.figtext(
        0.1, 0.9, f'Meetings containing\nlaughter predictions: {tot_pred_meetings}/{tot_meetings}')
    plt.title(
        f'Ratio of predicted laughter time\n to transcribed laughter time\nthreshold: {threshold}')

    if save_dir != '':
        path = os.path.join(
            save_dir, f'pred_to_transc_length_ratio_dist{threshold}.png')
        plt.savefig(path)
    plt.show()


##################################################
# OTHER
##################################################
MIN_LENGTH = cfg.model['min_length']


def laugh_df_to_csv(df):
    """
    Used to generate .csv file of a subset of laughter events (e.g. breath-laughs)
    e.g. for generating .wav-files using
    ./output_processing/laughs_to_wav.py from this .csv
    """
    df = df[df['type'] == 'breath-laugh']
    df.to_csv('breath_laugh.csv')


# TODO: rewrite this function (if needed) to work with new structure
def stats_for_different_min_length(preds_path):
    global MIN_LENGTH

    # Rounding to compensate np.arrange output inaccuracy (e.g.0.600000000001)
    lengths = list(np.arange(5.2, 8.2, 0.2).round(1))

    # This will contain each df with summary stats for different min_length values
    df_list = []

    for min_length in lengths:
        MIN_LENGTH = min_length
        print(f"Using min_laugh_length: {MIN_LENGTH}")

        # Need to recreate laughter indices and eval_df because min_length was changed
        # First create laughter segment indices
        # Mixed laugh index needs to be created first (see implementation of laugh_index)
        # NEED TO CHANGED THE FOLLOWING TWO LINES
        # create_mixed_laugh_index(parse.invalid_df)
        # create_laugh_index(parse.laugh_only_df)

        # Then create or load eval_df -> stats for each meeting
        eval_df = create_evaluation_df(preds_path)

        # Now calculate summary stats with new eval_df
        min_length_df = calc_sum_stats(eval_df)
        min_length_df['min_length'] = MIN_LENGTH
        print(min_length_df)
        df_list.append(min_length_df)

        # Print out the number of laughter events left for this min_length
        acc_len = 0
        acc_ev = 0
        for meeting in prep.laugh_index.keys():
            acc_len += prep.laugh_index[meeting]['tot_len']
            acc_ev += prep.laugh_index[meeting]['tot_events']
        print(f'Agg. laugh length: {acc_len:.2f}')
        print(f'Total laugh events: {acc_ev}')
        # Print number of invalid laughter events for this min_length
        acc_len = 0
        acc_ev = 0
        for meeting in prep.invalid_index.keys():
            acc_len += prep.invalid_index[meeting]['tot_len']
            acc_ev += prep.invalid_index[meeting]['tot_events']
        print(f'Agg. invalid laugh length: {acc_len:.2f}')
        print(f'Total invalid laugh events: {acc_ev}')

    tot_df = pd.concat(df_list)
    tot_df.to_csv('sum_stats_for_different_min_lengths.csv')


def create_csvs_for_meeting(meeting_id, preds_path):
    """
    Writes 2 csv files to disk:
        1) containing the transcribed laughter events for this meeting
        2) containing all predicted laughter events (for threshholds: 0.2, 0.4, 0.6, 0.8)
            - thus, duplicates are possible -> take this into account when analysing
    """
    tranc_laughs = parse.laugh_only_df[parse.laugh_only_df['meeting_id'] == meeting_id]
    tranc_laughs.to_csv(f'{meeting_id}_transc.csv')

    meeting_path = os.path.join(preds_path, meeting_id)
    # Get predictions for different threshholds
    df1 = textgrid_to_df(
        f'{meeting_path}/t_0.2/l_0.2')
    df2 = textgrid_to_df(
        f'{meeting_path}/t_0.4/l_0.2')
    df3 = textgrid_to_df(
        f'{meeting_path}/t_0.6/l_0.2')
    df4 = textgrid_to_df(
        f'{meeting_path}/t_0.8/l_0.2')
    # Concat them and write them to file
    result = pd.concat([df1, df2, df3, df4])
    result.to_csv(f'{meeting_id}_preds.csv')


def main():
    # Path that contains all predicted laughs in separate dirs for each parameter
    preds_path = './output_processing/outputs/'
    # Then create or load eval_df -> stats for each meeting
    eval_df = create_evaluation_df(preds_path)

    # stats_for_different_min_length(preds_path)
    sum_stats = calc_sum_stats(eval_df)
    print(sum_stats)

    # Create plots for different thresholds
    # for t in [.2, .4, .6, .8]:
    #     plot_aggregated_laughter_length_dist(eval_df, t, save_dir='./imgs/')
    #     plot_agg_pred_time_ratio_dist(eval_df, t, save_dir='./imgs/')


if __name__ == "__main__":
    main()
