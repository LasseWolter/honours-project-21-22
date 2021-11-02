from lxml import etree
import os 

chan_to_part = {}  # index mapping channel to participant per meeting
part_to_chan = {}  # index mapping participant to channel per meeting

def parse_preambles():
    global chan_to_part, part_to_chan
    '''
    Creates 2 id mappings
    1) Dict: (meeting_id) -> (dict(chan_id -> participant_id)) 
    2) Dict: (meeting_id) -> (dict(participant_id -> chan_id)) 
    '''
    dirname = os.path.dirname(__file__)
    preambles_path = os.path.join(dirname, 'data/preambles.mrt')
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


# Create id-mappings
parse_preambles()

