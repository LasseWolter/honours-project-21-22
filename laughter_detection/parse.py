from lxml import etree
# Using lxml instead of xml.etree.ElementTree because it has full XPath support 
# xml.etree.ElementTree only supports basic XPath syntax
import os

xpath_exp = "//Segment[VocalSound[contains(@Description,'laugh')][preceding-sibling::text() \
        [normalize-space()=''] and following-sibling::text()[normalize-space()='']] and count(./*) < 2]"



class Segment:
    def __init__(self, start, end,l_type ):
        self.start_time = float(start)
        self.end_time = float(end)
        self.laugh_type= l_type
        self.seg_len = self.end_time - self.start_time 


    @classmethod
    def from_xml_segment(cls, xml_seg):
        """
        Takes an etree element representing a Segment tag in XML
        """
        start = xml_seg.get('StartTime')
        end = xml_seg.get('EndTime')
        # [0] is the first child tag which is guaranteed to be a VocalSound
        # due to the XPath expression used for parsing the XML document
        l_type = xml_seg[0].get('Description')
        return cls(start, end, l_type)

    def __repr__(self):
        return "{} - {} \t [{:.2f}] => Type: {}".format(self.start_time, self.end_time, \
                self.seg_len, self.laugh_type)

    def __str__(self):
        return "{} - {} \t [{:.2f}] => Type: {}".format(self.start_time, self.end_time, \
                self.seg_len, self.laugh_type)

def get_laugh_segments(filename):
    print(filename)
    segs_list = [] 
    tree = etree.parse(filename)
    laugh_segs = tree.xpath(xpath_exp)
    for seg in laugh_segs:
        new_seg = Segment.from_xml_segment(seg)
        segs_list.append(new_seg)
    return segs_list


def main():
    tot_laugh_segs = []
    for filename in os.listdir('data'):
        if filename.endswith('.mrt'):
            tot_laugh_segs += get_laugh_segments(os.path.join('data',filename))

    print(len(tot_laugh_segs))
    avg_seg_len = 0

    for seg in tot_laugh_segs:
        avg_seg_len += (seg.end_time-seg.start_time)/len(tot_laugh_segs)

    print(avg_seg_len)

    
if __name__ == "__main__":
    main()
