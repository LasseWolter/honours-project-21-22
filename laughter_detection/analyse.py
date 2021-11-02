import textgrids
import os
import pandas as pd

from transcript_parsing import cfg

def textgrid_to_list(full_path, meeting_id, chan_id):
    # There are more recorded channels than participants
    # thus, not all channels are mapped to a participant
    # We focus on those that are mapped to a participant
    if chan_id not in cfg.chan_to_part[meeting_id].keys():
        return []
    interval_list = []
    grid = textgrids.TextGrid(full_path)
    for interval in grid['laughter']:
        # TODO: Change for breath laugh?!
        if str(interval.text) == 'laugh':
            part_id = cfg.chan_to_part[meeting_id][chan_id]
            interval_list.append([meeting_id, part_id, chan_id, interval.xmin,
                                  interval.xmax, interval.xmax-interval.xmin, str(interval.text)])
    return interval_list


def textgrid_to_df(file_path):
    tot_list = []
    for filename in os.listdir(file_path):
        if filename.endswith('.TextGrid'):
            full_path = os.path.join(file_path, filename)
            params = get_params_from_path(full_path)
            tot_list += textgrid_to_list(full_path, params["meeting_id"], params["chan_id"])

    cols = ['Meeting', 'ID', 'Channel', 'Start', 'End', 'Length', 'Type']
    df = pd.DataFrame(tot_list, columns=cols)
    print(df)

def get_params_from_path(path):
    '''
    Input: path
    Output: dict of parameters
    '''
    params = {}
    path = os.path.normpath(path)
    # First cut of .TextGrid
    # then split for to get parameters which are given by dir-names 
    params_list = path.replace('.TextGrid','').split('/')
    chan_id = params_list[-1]
    # Check if filename follows convention -> 'chanN.TextGrid'
    if not chan_id.startswith('chan'):
        raise NameError(
            "Did you follow the naming convention for channel .TextGrid-files -> 'chanN.TextGrid'")

    params['chan_id'] = chan_id
    params['min_length'] = params_list[-2]
    params['threshold'] = params_list[-3]
    meeting_id = params_list[-4] 
    # Check if meeting ID is valid -> B**NNN
    if not len(meeting_id) == 6:  # All IDs are 6 chars long
        raise NameError(
            "Did you follow the required directory structure? all chanN.TextGrid files \
            need to be in a directory with its meeting ID as name -> e.g. B**NNN")

    params['meeting_id'] = meeting_id 
    return params

    

def main(): 
    path = './output_processing/outputs/Bdb001/t_0.1/l_0.2/'
    df = textgrid_to_df(path)
    print(df)

if __name__ == "__main__":
    main()
