from lxml import etree
# Using lxml instead of xml.etree.ElementTree because it has full XPath support
# xml.etree.ElementTree only supports basic XPath syntax
import os
import pandas as pd
import textgrids
import sys

"""
Explanation of the xPath-expression:
    - get all Segment tags which have a VocalSound tag with the following properties as child:
        - Description attribute contains 'laugh'
        - preceding-sibling and following-sibling TextElements contain no text after stripping whitespace (-> normalize-space)
        - The VocalSound Tag is the only child (-> count(./*) < 2)
"""
xpath_exp = "//Segment[VocalSound[contains(@Description,'laugh')][preceding-sibling::text() \
        [normalize-space()=''] and following-sibling::text()[normalize-space()='']] and count(./*) < 2]"

CHAN_TO_PART = {}  # Global index mapping channel to participant per meeting
PART_TO_CHAN = {}  # Global index mapping participant to channel per meeting


def parse_xml_to_list(xml_seg, meeting_id):
    part_id = xml_seg.get('Participant')
    start = float(xml_seg.get('StartTime'))
    end = float(xml_seg.get('EndTime'))
    length = end-start
    # [0] is the first child tag which is guaranteed to be a VocalSound
    # due to the XPath expression used for parsing the XML document
    l_type = xml_seg[0].get('Description')
    chan_id = PART_TO_CHAN[meeting_id][part_id]
    return [part_id, chan_id, start, end, length, l_type]


def laughs_to_list(filename, meeting_id):
    seg_list = []
    tree = etree.parse(filename)
    laugh_segs = tree.xpath(xpath_exp)

    for seg in laugh_segs:
        seg_list.append([meeting_id] + parse_xml_to_list(seg, meeting_id))
    return seg_list


def laughs_to_df(file_path):
    tot_laugh_segs = []
    files = []
    file_dir = ""
    # If a directory is given take all .mrt files
    # otherwise only take given file
    if os.path.isdir(file_path):
        file_dir = file_path
        for filename in os.listdir(file_path):
            if filename.endswith('.mrt'):
                files.append(filename)
    else:
        if file_path.endswith('.mrt'):
            files.append(file_path)

    for filename in files:
        # Match particular file or all .mrt files
        # First split for cutting of .mrt
        # second split for discarding parent dirs
        meeting_id = filename.split('.')[0].split('/')[-1]
        full_path = os.path.join(file_dir, filename)
        tot_laugh_segs += laughs_to_list(full_path, meeting_id)

    cols = ['Meeting', 'ID', 'Channel', 'Start', 'End', 'Length', 'Type']
    df = pd.DataFrame(tot_laugh_segs, columns=cols)
    return df


def print_stats(df):
    print(df)
    print('avg-snippet-length: {:.2f}s'.format(df['Length'].mean()))
    print('Number of laughter only snippets: {}'.format(df.shape[0]))
    tot_dur = df['Length'].sum()
    print('Total laughter duration in three formats: \n- {:.2f}h \n- {:.2f}min \n- {:.2f}s'.format(
        (tot_dur / 3600), (tot_dur / 60), tot_dur))


def parse_preambles():
    '''
    Input: filepath of the preambles.mrt
    Output: Tuple of 2 Dicts 
        1) Dict: (meeting_id) -> (dict(chan_id -> participant_id)) 
        2) Dict: (meeting_id) -> (dict(participant_id -> chan_id)) 
    '''
    chan_to_part = {}
    tree = etree.parse('data/preambles.mrt')
    meetings = tree.xpath('//Meeting')
    for meeting in meetings:
        id = meeting.get('Session')
        part_map = {}
        for part in meeting.xpath('./Preamble/Participants/Participant'):
            part_map[part.get('Channel')] = part.get('Name')

        chan_to_part[id] = part_map

    part_to_channel = {}
    for meeting_id in chan_to_part.keys():
        part_to_channel[meeting_id] = {part_id: chan_id for (
            chan_id, part_id) in chan_to_part[meeting_id].items()}

    return (chan_to_part, part_to_channel)


def textgrid_to_list(full_path, meeting_id, chan_id):
    interval_list = []
    grid = textgrids.TextGrid(full_path)
    for interval in grid['laughter']:
        # TODO: Change for breath laugh?!
        if str(interval.text) == 'laugh':
            part_id = CHAN_TO_PART[meeting_id][chan_id]
            interval_list.append([meeting_id, part_id, chan_id, interval.xmin,
                                  interval.xmax, interval.xmax-interval.xmin, str(interval.text)])
    return interval_list


def textgrid_to_df(file_path):
    tot_list = []
    for filename in os.listdir(file_path):
        if filename.endswith('.TextGrid'):
            full_path = os.path.join(file_path, filename)
            # First split for cutting of .TextGrid
            # second split for discarding parent dirs
            path_list = full_path.split('.')[0].split('/')

            # ASSUMES that directory has a meeting ID as name -> B**NNN
            meeting_id = path_list[-2]

            # ASSUMES that channel files are stored using name convention -> 'chanN.TextGrid'
            chan_id = path_list[-1]
            if not chan_id.startswith('chan'):
                raise NameError(
                    "Did you follow the naming convention for channel .TextGrid-files -> 'chanN.TextGrid'")

            tot_list += textgrid_to_list(full_path, meeting_id, chan_id)

    cols = ['Meeting', 'ID', 'Channel', 'Start', 'End', 'Length', 'Type']
    df = pd.DataFrame(tot_list, columns=cols)
    print(df)


def main():
    global CHAN_TO_PART, PART_TO_CHAN

    if (len(sys.argv) < 3):
        print("Usage: parse.py <.mrt-file or .mrt dir> <TextGrid-dir>")
        return

    # Populate channel to participant index with info from preambles files
    CHAN_TO_PART, PART_TO_CHAN = parse_preambles()
    transcript_path = sys.argv[1]
    textgrid_dir = sys.argv[2]
    textgrid_to_df(textgrid_dir)
    laugh_df = laughs_to_df(transcript_path)
    print_stats(laugh_df)


if __name__ == "__main__":
    main()
