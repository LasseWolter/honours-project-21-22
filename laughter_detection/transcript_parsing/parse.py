from lxml import etree
# Using lxml instead of xml.etree.ElementTree because it has full XPath support
# xml.etree.ElementTree only supports basic XPath syntax
import os
import pandas as pd
import sys

chan_to_part = {}  # index mapping channel to participant per meeting
part_to_chan = {}  # index mapping participant to channel per meeting
laugh_only_df = pd.DataFrame()  # dataframe containing transcribed laugh only events
mixed_laugh_df = pd.DataFrame() # dataframe containing mixed laugh events


def parse_preambles(path):
    global chan_to_part, part_to_chan
    '''
    Creates 2 id mappings
    1) Dict: (meeting_id) -> (dict(chan_id -> participant_id)) 
    2) Dict: (meeting_id) -> (dict(participant_id -> chan_id)) 
    '''
    dirname = os.path.dirname(__file__)
    preambles_path = os.path.join(dirname, path)
    chan_to_part = {}
    tree = etree.parse(preambles_path)
    meetings = tree.xpath('//Meeting')
    for meeting in meetings:
        id = meeting.get('Session')
        part_map = {}
        # Make sure that both Name and Channel attributes exist
        for part in meeting.xpath('./Preamble/Participants/Participant[@Name and @Channel]'):
            part_map[part.get('Channel')] = part.get('Name')

        chan_to_part[id] = part_map

    part_to_chan = {}
    for meeting_id in chan_to_part.keys():
        part_to_chan[meeting_id] = {part_id: chan_id for (
            chan_id, part_id) in chan_to_part[meeting_id].items()}


def parse_xml_to_list(xml_seg, meeting_id):
    '''
    Input: xml laughter segment as etree Element, meeting id
    Output: list of features representing this laughter segment
    '''
    part_id = xml_seg.get('Participant')
    start = float(xml_seg.get('StartTime'))
    end = float(xml_seg.get('EndTime'))
    length = end-start
    # [0] is the first child tag which is guaranteed to be a VocalSound
    # due to the XPath expression used for parsing the XML document

    # In case there are multiple tags in this segment get the first laugh tag 
    # for the type description. If there are more than on laugh tags the
    # description from the first will be taken
    first_laugh_tag = xml_seg.xpath("./VocalSound[contains(@Description, 'laugh')]")[0]
    l_type = first_laugh_tag.get('Description')
    # Make sure that this participant actually has a corresponding audio channel
    if part_id not in part_to_chan[meeting_id].keys():
        return []
    chan_id = part_to_chan[meeting_id][part_id]
    return [meeting_id, part_id, chan_id, start, end, length, l_type]


def laughs_to_list(filename, meeting_id):
    """
    Returns two list: 
        1) List containing segments laughter only (no text or other sounds surrounding it)
        2) List containing segments of mixed laughter (laughter surrounding by other sounds)
    """
    laugh_mixed_list = []
    laugh_only_list = []

    # Get all segments that contain some kind of laughter (e.g. 'laugh', 'breath-laugh')
    xpath_exp = "//Segment[VocalSound[contains(@Description,'laugh')]]"
    tree = etree.parse(filename)
    laugh_segs = tree.xpath(xpath_exp)

    # For each laughter segment classify it as laugh only or mixed laugh
    # mixed laugh means that the laugh occurred next to speech or any other sound
    for seg in laugh_segs:
        seg_as_list = parse_xml_to_list(seg, meeting_id)
        # Check if there is no surrounding text and no other Sound tags
        if seg.text.strip() == '' and len(seg.getchildren()) == 1:
            laugh_only_list.append(seg_as_list)
        else:
            laugh_mixed_list.append(seg_as_list)

    return laugh_only_list, laugh_mixed_list


def parse_transcripts(path):
    '''
    Parse transcripts and store laughs in laugh_df
    '''
    global laugh_only_df, mixed_laugh_df

    tot_laugh_only_segs = []
    tot_mixed_laugh_segs = []
    files = []
    file_dir = ""
    # If a directory is given take all .mrt files
    # otherwise only take given file
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, path)

    # Match particular file or all .mrt files
    if os.path.isdir(path):
        file_dir = path
        for filename in os.listdir(path):
            if filename.endswith('.mrt'):
                files.append(filename)
    else:
        if path.endswith('.mrt'):
            files.append(path)

    # Iterate over all .mrt files
    for filename in files:
        # Get meeting id by getting the basename and stripping the extension
        basename = os.path.basename(filename)
        meeting_id = os.path.splitext(basename)[0]
        full_path = os.path.join(file_dir, filename)
        laugh_only, mixed_laugh = laughs_to_list(full_path, meeting_id)
        tot_laugh_only_segs += laugh_only
        tot_mixed_laugh_segs += mixed_laugh

    cols = ['Meeting', 'ID', 'Channel', 'Start', 'End', 'Length', 'Type']
    laugh_only_df = pd.DataFrame(tot_laugh_only_segs, columns=cols)
    mixed_laugh_df = pd.DataFrame(tot_mixed_laugh_segs, columns=cols)



def print_stats(df):
    print(df)
    if df.size == 0:
        print('Empty DataFrame')
        return
    print('avg-snippet-length: {:.2f}s'.format(df['Length'].mean()))
    print('Number of laughter only snippets: {}'.format(df.shape[0]))
    tot_dur = df['Length'].sum()
    print('Total laughter duration in three formats: \n- {:.2f}h \n- {:.2f}min \n- {:.2f}s'.format(
        (tot_dur / 3600), (tot_dur / 60), tot_dur))


# Parse transcripts and preambles on import
parse_preambles('./data/preambles.mrt')
parse_transcripts('./data/')


def main():
    print_stats(laugh_only_df)
    print_stats(mixed_laugh_df)


if __name__ == "__main__":
    main()
