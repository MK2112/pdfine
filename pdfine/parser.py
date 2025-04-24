import re
import zlib
import binascii
from .utils import PDFParseError, logger
from typing import Dict, Any, List, Tuple, Union

"""
PDFine Parser
"""

# Utility Funcs
def _findall(pattern, data):
    return [m for m in re.finditer(pattern, data)]

def _decode_ascii85(data):
    try:
        # Remove delimiters if present
        if data.startswith(b'<~') and data.endswith(b'~>'):
            data = data[2:-2]
        return binascii.a85decode(data, adobe=True)
    except Exception as e:
        logger.warning(f'ASCII85 decode failed: {e}')
        return data

def _decode_lzw(data):
    # Python doesn't have built-in LZW; fallback to raw
    logger.warning('LZWDecode not supported natively; returning raw stream')
    return data

def _decode_dct(data):
    # DCTDecode is JPEG; just return the raw bytes
    return data

# Core Funcs
def parse_pdf_header(data: bytes) -> str:
    match = re.match(br'%PDF-(\d+\.\d+)', data)
    if not match:
        raise PDFParseError('Invalid PDF header')
    return match.group(0).decode('ascii')

def find_xref_offset(data: bytes) -> int:
    # Try to find last startxref (for incremental updates)
    idx = data.rfind(b'startxref')
    if idx == -1:
        raise PDFParseError('No startxref found')
    after = data[idx+9:idx+40]
    match = re.search(br'(\d+)', after)
    if not match:
        raise PDFParseError('No xref offset after startxref')
    return int(match.group(1))

def parse_xref_table(data: bytes, start: int) -> Dict[int, int]:
    xref = {}
    if data[start:start+4] == b'xref':
        # Standard xref table
        lines = data[start:].split(b'\n')
        i = 1
        while i < len(lines):
            hdr = lines[i].split()
            if len(hdr) == 2 and hdr[0].isdigit():
                obj_start, obj_count = map(int, hdr)
                i += 1
                for j in range(obj_count):
                    entry = lines[i+j]
                    offset = int(entry[:10])
                    xref[obj_start+j] = offset
                i += obj_count
            elif lines[i].startswith(b'trailer'):
                break
            else:
                i += 1
    else:
        # Brute-force search for all obj positions
        logger.warning('xref table not found at offset; using fallback object finder')
        for m in _findall(br'\n(\d+) (\d+) obj', data):
            obj_num = int(m.group(1))
            xref[obj_num] = m.start(0)
    return xref

def parse_trailer(data: bytes, xref_offset: int) -> Dict[str, Any]:
    pos = data.find(b'trailer', xref_offset)
    if pos == -1:
        logger.warning('No trailer after xref; using fallback')
        pos = data.find(b'trailer')
        if pos == -1:
            return {}
    end = data.find(b'>>', pos)
    trailer = data[pos:end+2]
    root = re.search(br'/Root (\d+) 0 R', trailer)
    return {'Root': int(root.group(1))} if root else {}

def parse_indirect_object(data: bytes, offset: int) -> Dict[str, Any]:
    start = data.find(b'obj', offset-20, offset+20)
    if start == -1:
        raise PDFParseError('obj keyword not found')
    header = data[offset-15:offset].split()
    try:
        obj_num = int(header[-2]); gen_num = int(header[-1])
    except Exception:
        obj_num, gen_num = -1, -1
    end = data.find(b'endobj', offset)
    content = data[start+3:end]
    return {'obj_num': obj_num, 'gen_num': gen_num, 'data': content}

def extract_objects(data: bytes, xref: Dict[int, int]) -> Dict[int, Dict]:
    objs = {}
    for num, off in xref.items():
        try:
            obj = parse_indirect_object(data, off)
            objs[num] = obj
        except Exception as e:
            logger.warning(f'Obj parse failed {num}: {e}')
    return objs

def _resolve_indirect(obj: Union[bytes, Dict], objects: Dict[int, Dict]) -> bytes:
    if isinstance(obj, dict) and 'data' in obj:
        return obj['data']
    if isinstance(obj, bytes):
        m = re.match(br'(\d+) 0 R', obj.strip())
        if m:
            obj_num = int(m.group(1))
            return objects.get(obj_num, {}).get('data', b'')
    return obj

def find_pages(objects: Dict[int, Dict]) -> List[int]:
    # Traverse /Pages tree if present
    page_nums = []
    for num, obj in objects.items():
        data = obj['data']
        if b'/Type /Pages' in data:
            # Recursively find kids
            kids = re.findall(br'/Kids \[(.*?)\]', data, re.DOTALL)
            for kidlist in kids:
                for m in re.finditer(br'(\d+) 0 R', kidlist):
                    kid_num = int(m.group(1))
                    if kid_num in objects:
                        kobj = objects[kid_num]
                        if b'/Type /Page' in kobj['data']:
                            page_nums.append(kid_num)
                        elif b'/Type /Pages' in kobj['data']:
                            # Nested pages
                            page_nums.extend(find_pages({kid_num: kobj}))
        elif b'/Type /Page' in data:
            page_nums.append(num)
    # Fallback: find all /Type /Page if tree is broken
    if not page_nums:
        for num, obj in objects.items():
            if b'/Type /Page' in obj['data']:
                page_nums.append(num)
    return sorted(set(page_nums))

def extract_content_streams(objects: Dict[int, Dict], page_num: int) -> List[bytes]:
    data = objects[page_num]['data']
    # Support both array and single reference
    refs = re.findall(br'/Contents (\[.*?\]|\d+ 0 R)', data, re.DOTALL)
    streams = []
    for ref in refs:
        if ref.startswith(b'['):
            for m in re.finditer(br'(\d+) 0 R', ref):
                obj = objects.get(int(m.group(1)))
                if obj:
                    streams.append(_decode_stream(obj['data']))
        else:
            m = re.match(br'(\d+) 0 R', ref)
            if m:
                obj = objects.get(int(m.group(1)))
                if obj:
                    streams.append(_decode_stream(obj['data']))
    # Fallback: inline stream
    if not streams and b'stream' in data:
        streams.append(_decode_stream(data))
    return streams

def _decode_stream(obj_data: bytes) -> bytes:
    start = obj_data.find(b'stream')
    end = obj_data.find(b'endstream')
    if start == -1 or end == -1:
        return b''
    stream = obj_data[start+6:end].strip(b'\r\n')
    filters = re.findall(br'/Filter ?/?([A-Za-z0-9]+)', obj_data)
    for f in filters:
        f = f.decode('ascii')
        if f == 'FlateDecode':
            try:
                stream = zlib.decompress(stream)
            except Exception as e:
                logger.warning(f'Flate decompression failed: {e}')
        elif f == 'LZWDecode':
            stream = _decode_lzw(stream)
        elif f == 'ASCII85Decode':
            stream = _decode_ascii85(stream)
        elif f == 'DCTDecode':
            stream = _decode_dct(stream)
        else:
            logger.warning(f'Unknown filter: {f}')
    return stream

def extract_text_from_stream(stream: bytes) -> str:
    # Heuristic: extract anything between BT ... ET (PDF text blocks)
    texts = []
    for m in re.finditer(br'BT(.*?)ET', stream, re.DOTALL):
        block = m.group(1)
        # Find Tj/TJ operators (text showing)
        for tj in re.finditer(br'\((.*?)\) ?Tj', block):
            try:
                txt = tj.group(1).decode('utf-8', errors='replace')
                texts.append(txt)
            except Exception:
                continue
        for tj in re.finditer(br'\[(.*?)\] ?TJ', block):
            arr = tj.group(1)
            for s in re.finditer(br'\((.*?)\)', arr):
                try:
                    txt = s.group(1).decode('utf-8', errors='replace')
                    texts.append(txt)
                except Exception:
                    continue
    # Extract all printable ASCII if nothing found
    if not texts:
        try:
            txt = stream.decode('utf-8', errors='replace')
            texts.append(re.sub(r'[^\x20-\x7E\n]', '', txt))
        except Exception:
            pass
    return '\n'.join(texts).strip()
def parse_pdf_header(data: bytes) -> str:
    match = re.match(br'%PDF-(\d+\.\d+)', data)
    if not match:
        raise PDFParseError('Invalid PDF header')
    return match.group(0).decode('ascii')

def find_xref_offset(data: bytes) -> int:
    idx = data.rfind(b'startxref')
    if idx == -1:
        raise PDFParseError('No startxref found')
    after = data[idx+9:idx+40]
    match = re.search(br'(\d+)', after)
    if not match:
        raise PDFParseError('No xref offset after startxref')
    return int(match.group(1))

def parse_xref_table(data: bytes, start: int) -> Dict[int, int]:
    xref = {}
    if data[start:start+4] != b'xref':
        raise PDFParseError('xref table not at offset')
    lines = data[start:].split(b'\n')
    i = 1
    while i < len(lines):
        hdr = lines[i].split()
        if len(hdr) == 2 and hdr[0].isdigit():
            obj_start, obj_count = map(int, hdr)
            i += 1
            for j in range(obj_count):
                entry = lines[i+j]
                offset = int(entry[:10])
                xref[obj_start+j] = offset
            i += obj_count
        elif lines[i].startswith(b'trailer'):
            break
        else:
            i += 1
    return xref

def parse_trailer(data: bytes, xref_offset: int) -> Dict[str, Any]:
    pos = data.find(b'trailer', xref_offset)
    if pos == -1:
        raise PDFParseError('No trailer after xref')
    end = data.find(b'>>', pos)
    trailer = data[pos:end+2]
    root = re.search(br'/Root (\d+) 0 R', trailer)
    if not root:
        raise PDFParseError('No /Root in trailer')
    return {'Root': int(root.group(1))}

def parse_indirect_object(data: bytes, offset: int) -> Dict[str, Any]:
    start = data.find(b'obj', offset-20, offset+20)
    if start == -1:
        raise PDFParseError('obj keyword not found')
    header = data[offset-15:offset].split()
    obj_num = int(header[-2]); gen_num = int(header[-1])
    end = data.find(b'endobj', offset)
    content = data[start+3:end]
    return {'obj_num': obj_num, 'gen_num': gen_num, 'data': content}

def extract_objects(data: bytes, xref: Dict[int, int]) -> Dict[int, Dict]:
    objs = {}
    for num, off in xref.items():
        try:
            obj = parse_indirect_object(data, off)
            objs[num] = obj
        except Exception as e:
            logger.warning(f'Obj parse failed {num}: {e}')
    return objs

def find_pages(objects: Dict[int, Dict]) -> list:
    pages = []
    for num, obj in objects.items():
        if b'/Type /Page' in obj['data']:
            pages.append(num)
    return pages

def extract_content_streams(objects: Dict[int, Dict], page_num: int) -> list:
    data = objects[page_num]['data']
    refs = re.findall(br'/Contents (\d+) 0 R', data)
    streams = []
    for r in refs:
        obj = objects.get(int(r))
        if obj:
            streams.append(_decode_stream(obj['data']))
    if not streams and b'stream' in data:
        streams.append(_decode_stream(data))
    return streams

def _decode_stream(obj_data: bytes) -> bytes:
    start = obj_data.find(b'stream')
    end = obj_data.find(b'endstream')
    if start == -1 or end == -1:
        return b''
    stream = obj_data[start+6:end].strip(b'\r\n')
    if b'/FlateDecode' in obj_data:
        try: stream = zlib.decompress(stream)
        except: logger.warning('Flate decompression failed')
    return stream
