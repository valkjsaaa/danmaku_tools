import argparse
import subprocess
import xml.dom.minidom as DOM
from dateutil.parser import parse
from xml.parsers.expat import ExpatError

parser = argparse.ArgumentParser(description='Merge BiliLiveReocrder XML')
parser.add_argument('xml_files', type=str, nargs='+', help='path to the danmaku file')
parser.add_argument('--offset_time', type=float, default=0, help='offset danmaku time')
parser.add_argument('--video_time', type=str, default="", help='use video length as the duration of the file')
parser.add_argument('--output', type=str, default=None, help='output path for the output XML', required=True)


def get_root_time(root_xml):
    record_info = root_xml.getElementsByTagName('BililiveRecorderRecordInfo')[0]
    record_start_time_str = record_info.getAttribute('start_time')
    record_start_time = parse(record_start_time_str)
    return record_start_time


def check_root_offset(element, offset_time):
    orig_time = 0
    for child in element.childNodes:
        if child.nodeName in ['sc', 'gift', 'guard']:
            orig_time = float(child.getAttribute('ts'))
            break
        elif child.nodeName in ['d']:
            orig_parameters_str = child.getAttribute('p').split(',')
            orig_time = float(orig_parameters_str[0])
            break
    if orig_time + offset_time < 0:
        offset_time = -orig_time
    return offset_time


def set_root_offset_time(element, offset_time):
    for child in element.childNodes:
        if child.nodeName in ['sc', 'gift', 'guard']:
            orig_time = float(child.getAttribute('ts'))
            new_time = orig_time + offset_time
            new_time_str = str(new_time)
            child.setAttribute('ts', new_time_str)
        elif child.nodeName in ['d']:
            orig_parameters_str = child.getAttribute('p').split(',')
            orig_time = float(orig_parameters_str[0])
            new_time = orig_time + offset_time
            new_parameters_str = [str(new_time)] + orig_parameters_str[1:]
            child.setAttribute('p', ','.join(new_parameters_str))


def add_root(orig_root, new_root, new_offset):
    for child in new_root.childNodes:
        if child.nodeName in ['sc', 'gift', 'guard']:
            orig_time = float(child.getAttribute('ts'))
            new_time = orig_time + new_offset
            new_time_str = str(new_time)
            child.setAttribute('ts', new_time_str)
            orig_root.appendChild(child)
        elif child.nodeName in ['d']:
            orig_parameters_str = child.getAttribute('p').split(',')
            orig_time = float(orig_parameters_str[0])
            new_time = orig_time + new_offset
            new_parameters_str = [str(new_time)] + orig_parameters_str[1:]
            child.setAttribute('p', ','.join(new_parameters_str))
            orig_root.appendChild(child)


if __name__ == '__main__':
    args = parser.parse_args()

    if len(args.xml_files) == 0:
        print("At least one XML files have to be passed as input.")

    try:
        doc = DOM.parse(args.xml_files[0])
    except ExpatError:
        # Read concat file
        with open(args.xml_files[0], encoding='utf-8') as f:
            args.xml_files = f.read().replace('"', '').split('\n')
        doc = DOM.parse(args.xml_files[0])

    root = doc.documentElement
    # all_flv = ""

    for child in root.childNodes:
        if child.nodeType == DOM.Node.TEXT_NODE:
            # Remove <DOM Text node "'\n  '">
            root.removeChild(child)

    if args.offset_time != 0:
        new_root_offset = check_root_offset(root, args.offset_time)
        set_root_offset_time(root, new_root_offset)
    else:
        new_root_offset = 0

    if args.video_time == "":
        old_root_time = get_root_time(root)
        for xml_file in args.xml_files[1:]:
            new_root = DOM.parse(xml_file).documentElement
            new_root_time = get_root_time(new_root)
            new_root_offset += (new_root_time - old_root_time).total_seconds()
            old_root_time = new_root_time
            add_root(root, new_root, new_root_offset)
    else:
        for i in range(len(args.xml_files) - 1):
            new_root = DOM.parse(args.xml_files[i + 1]).documentElement
            prev_xml_path = args.xml_files[i]
            base_file_path = prev_xml_path.rpartition('.')[0]
            flv_file_path = base_file_path + args.video_time
            total_seconds_str = subprocess.check_output(
                f'ffprobe -v error -show_entries format=duration '
                f'-of default=noprint_wrappers=1:nokey=1 "{flv_file_path}"', shell=True
            )
            # all_flv += flv_file_path + "\n"
            new_root_offset += float(total_seconds_str)
            add_root(root, new_root, new_root_offset)

    with open(args.output, 'w', encoding='utf-8') as f:
        doc.writexml(f, indent='', addindent='\t', newl='\n', encoding="utf-8")
