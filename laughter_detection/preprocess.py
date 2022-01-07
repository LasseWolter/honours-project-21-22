import config as cfg
import utils
import portion as P
from transcript_parsing import parse

laugh_index = {}
invalid_index = {}


def seg_invalid(row):
    '''
    This functions specifies what makes a segment invalid 
    Input: row defining an audio segment with the following columns:
        - ['meeting_id', 'part_id', 'chan', 'start', 'end', 'length', 'type']
    '''
    # If the length is shorter than min_length passed to detection algorithm, mark invalid
    #   - empirically tested -> this doesn't apply to many segments
    return (row['length'] < cfg.model["min_length"] or row['type'] == 'breath-laugh')


def append_to_index(index, row, meeting_id, part_id):
    '''
    Append this segment to invalid segments index
    '''
    start = utils.to_frames(row['start'])
    end = utils.to_frames(row['end'])

    # Append to existing intervals or create new dict entry
    if part_id in index[meeting_id].keys():
        index[meeting_id][part_id] = index[meeting_id][part_id] | P.closed(
            start, end)
    else:
        index[meeting_id][part_id] = P.closed(start, end)

    index[meeting_id]['tot_len'] += row['length']
    index[meeting_id]['tot_events'] += 1
    return index


def create_laugh_index(df):
    """
    Creates a laugh_index with all transcribed laughter events
    per particpant per meeting
    The segments are stored as disjunction of closed intervals (using portion library)
    dict structure:
    {
        meeting_id: {
            tot_len: INT, 
            tot_events: INT,
            part_id: P.closed(start,end) | P.closed(start,end),
            part_id: P.closed(start,end)| P.closed(start,end)
        }
        ...
    }
    """
    global laugh_index, invalid_index

    if invalid_index == {}:
        raise RuntimeError(
            "INVALID_INDEX needs to be created before LAUGH_INDEX")
    meeting_groups = df.groupby(['meeting_id'])

    for meeting_id, meeting_df in meeting_groups:
        laugh_index[meeting_id] = {}
        laugh_index[meeting_id]['tot_len'] = 0
        laugh_index[meeting_id]['tot_events'] = 0

        # Ensure rows are sorted by 'start'-time in ascending order
        part_groups = meeting_df.sort_values('start').groupby(['part_id'])
        for part_id, part_df in part_groups:
            laugh_index[meeting_id][part_id] = P.empty()
            for _, row in part_df.iterrows():
                # If segment is invalid, append to invalid segments index
                if seg_invalid(row):
                    invalid_index = append_to_index(
                        invalid_index, row, meeting_id, part_id)
                    continue

                # If segment is valid, append to laugh segments index
                laugh_index = append_to_index(
                    laugh_index, row, meeting_id, part_id)


def create_invalid_index(df):
    global invalid_index
    """
    Creates an invalid_index with all segments invalid for our project
    e.g. transcribed laughter events occurring next to other sounds 
    The segments are stored as disjunction of closed intervals (using portion library) per participant per meeting
    dict structure (same as laugh_index):
    {
        meeting_id: {
            tot_len: INT,
            tot_events: INT,
            part_id: P.closed(start,end) | P.closed(start,end),
            part_id: P.closed(start,end) | P.closed(start,end)
        }
        ...
    }
    """
    meeting_groups = df.groupby(['meeting_id'])
    for meeting_id, meeting_df in meeting_groups:
        invalid_index[meeting_id] = {}
        invalid_index[meeting_id]['tot_len'] = 0
        invalid_index[meeting_id]['tot_events'] = 0

        # Ensure rows are sorted by 'start'-time in ascending order
        part_groups = meeting_df.sort_values('start').groupby(['part_id'])
        for part_id, part_df in part_groups:
            for _, row in part_df.iterrows():
                invalid_index = append_to_index(
                    invalid_index, row, meeting_id, part_id)


#############################################
# EXECUTED ON IMPORT
#############################################

# First create invalid index and then laughter index
# Invalid index needs to be created first since filtered out segments
# during laugh index creation are added to the invalid index
create_invalid_index(parse.invalid_df)
create_laugh_index(parse.laugh_only_df)
