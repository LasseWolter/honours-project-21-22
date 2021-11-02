from lxml import etree
# Using lxml instead of xml.etree.ElementTree because it has full XPath support
# xml.etree.ElementTree only supports basic XPath syntax
import cfg
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



def parse_xml_to_list(xml_seg, meeting_id):
    part_id = xml_seg.get('Participant')
    start = float(xml_seg.get('StartTime'))
    end = float(xml_seg.get('EndTime'))
    length = end-start
    # [0] is the first child tag which is guaranteed to be a VocalSound
    # due to the XPath expression used for parsing the XML document
    l_type = xml_seg[0].get('Description')
    # Make sure that this participant actually has a corresponding audio channel
    if part_id not in cfg.part_to_chan[meeting_id].keys():
        return []
    chan_id = cfg.part_to_chan[meeting_id][part_id]
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

def main():
    if (len(sys.argv) < 2):
        print("Usage: parse.py <.mrt-file or .mrt dir>")
        return

    # Populate channel to participant index with info from preambles files
    transcript_path = sys.argv[1]
    laugh_df = laughs_to_df(transcript_path)
    print_stats(laugh_df)


if __name__ == "__main__":
    main()
