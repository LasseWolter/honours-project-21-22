from lxml import etree
# Using lxml instead of xml.etree.ElementTree because it has full XPath support
# xml.etree.ElementTree only supports basic XPath syntax
import os
import pandas as pd
import textgrids

xpath_exp = "//Segment[VocalSound[contains(@Description,'laugh')][preceding-sibling::text() \
        [normalize-space()=''] and following-sibling::text()[normalize-space()='']] and count(./*) < 2]"


def parse_xml_to_list(xml_seg):
    part_id = xml_seg.get('Participant')
    start = float(xml_seg.get('StartTime'))
    end = float(xml_seg.get('EndTime'))
    length = end-start
    # [0] is the first child tag which is guaranteed to be a VocalSound
    # due to the XPath expression used for parsing the XML document
    l_type = xml_seg[0].get('Description')
    return [part_id, start, end, length, l_type]


def laughs_to_list(filename, meeting_id):
    seg_list = []
    tree = etree.parse(filename)
    laugh_segs = tree.xpath(xpath_exp)

    for seg in laugh_segs:
        seg_list.append([meeting_id] + parse_xml_to_list(seg))
    return seg_list


def laughs_to_df():
    tot_laugh_segs = []
    for filename in os.listdir('data'):
        if filename.endswith('Bed015.mrt'):
            # First split for cutting of .mrt
            # second split for discarding parent dirs
            meeting_id=filename.split('.')[0].split('/')[-1]
            full_path = os.path.join('data', filename)
            tot_laugh_segs += laughs_to_list(full_path, meeting_id)

    cols = ['Meeting', 'ID', 'Start', 'End', 'Length', 'Type']
    df = pd.DataFrame(tot_laugh_segs, columns=cols)
    print(df[df["ID"]=="mn015"])
    print('avg-snippet-length: {} '.format(df['Length'].mean()))


def textgrid_to_list(filename, meeting_id):
    interval_list = []
    grid = textgrids.TextGrid(filename)
    for interval in grid['laughter']:
        # TODO: Change for belly laugh?!
        if str(interval.text) == 'laugh':
            interval_list.append([meeting_id, 'xxx', interval.xmin, 
                interval.xmax, interval.xmax-interval.xmin, str(interval.text)])
    return interval_list

def textgrid_to_df():
    tot_list=[]
    for filename in os.listdir('data'):
        if filename.endswith('chan0_laughter.TextGrid'):
            full_path = os.path.join('data', filename)
            # First split for cutting of .mrt
            # second split for discarding parent dirs
            meeting_id=filename.split('.')[0].split('/')[-1]
            tot_list += textgrid_to_list(full_path, meeting_id)

    cols = ['Meeting', 'ID', 'Start', 'End', 'Length', 'Type']
    df = pd.DataFrame(tot_list, columns=cols)
    print(df)

def main():
    textgrid_to_df()
    laughs_to_df()


if __name__ == "__main__":
    main()
