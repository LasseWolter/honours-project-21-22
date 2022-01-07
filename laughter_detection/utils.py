import config as cfg
import portion as P


def to_frames(time_in_sec):
    '''
    Represent time in seconds as number of frames. 
    Frame duration is defined in config
    '''
    # Calculate fps (1000ms/frame_duration)
    factor = 1000/cfg.model['frame_duration']
    return round(time_in_sec*factor)


def to_sec(num_of_frames):
    '''
    Turn time in number of frames to time in seconds.
    Frame duration is defined in config
    '''
    # Calculate fps (1000ms/frame_duration)
    factor = 1000/cfg.model['frame_duration']
    return num_of_frames/factor


def p_len(p_interval):
    '''
    Takes an interval of portion's Interval class and returns its (accumulated) length. 
    Portion's Interval class includes disjunctions of atomic intervals.

    E.g. p_len( (P.closed(1,3) | P.closed(10,11)) ) = 5
    '''
    # Iterate over the (disjunction of) interval(s) with step-size 1
    # Then count the number of elements in the list
    return len(list(P.iterate(p_interval, step=1)))
