from typing import Callable, List, Optional, Union
import os
from PIL import Image, PngImagePlugin
import piexif
import piexif.helper
import ast
import re
import torch

def crc_hash(string: str) -> str:  # 8 characters
    crc = zlib.crc32(string.encode())
    return format(crc & 0xFFFFFFFF, '08x')

def save_image_with_geninfo(image, geninfo, filename, extension=None, existing_pnginfo=None, pnginfo_section_name='parameters'):
    """
    Saves image to filename, including geninfo as text information for generation info.
    For PNG images, geninfo is added to existing pnginfo dictionary using the pnginfo_section_name argument as key.
    For JPG images, there's no dictionary and geninfo just replaces the EXIF description.
    """

    if extension is None:
        extension = os.path.splitext(filename)[1]

    image_format = Image.registered_extensions()[extension]

    if extension.lower() == '.png':
        existing_pnginfo = existing_pnginfo or {}
        existing_pnginfo[pnginfo_section_name] = geninfo

        pnginfo_data = PngImagePlugin.PngInfo()
        for k, v in (existing_pnginfo or {}).items():
            pnginfo_data.add_text(k, str(v))

        image.save(filename, format=image_format, quality=80, pnginfo=pnginfo_data)

    elif extension.lower() in (".jpg", ".jpeg", ".webp"):
        if image.mode == 'RGBA':
            image = image.convert("RGB")
        elif image.mode == 'I;16':
            image = image.point(lambda p: p * 0.0038910505836576).convert("RGB" if extension.lower() == ".webp" else "L")

        image.save(filename, format=image_format, quality=80, lossless=False)

        if geninfo is not None:
            exif_bytes = piexif.dump({
                "Exif": {
                    piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(geninfo or "", encoding="unicode")
                },
            })

            piexif.insert(exif_bytes, filename)
    elif extension.lower() == ".gif":
        image.save(filename, format=image_format, comment=geninfo)
    else:
        image.save(filename, format=image_format, quality=80)




import zlib
import base64

def compress_string(input_string):
    # Compress the input string
    compressed_data = zlib.compress(input_string.encode('utf-8'))
    # Encode the compressed data to base64
    base64_compressed_data = base64.b64encode(compressed_data)
    # Convert the base64 bytes back to string
    return base64_compressed_data.decode('utf-8')

def decompress_string(input_string):
    # Decode the input string from base64
    base64_compressed_data = input_string.encode('utf-8')
    compressed_data = base64.b64decode(base64_compressed_data)
    # Decompress the data
    decompressed_data = zlib.decompress(compressed_data)
    return decompressed_data.decode('utf-8')


def parse_params_from_image(img: Image.Image) -> dict:
    if isinstance(img, str):
        img = Image.open(img)
    geninfo,_ = read_info_from_image(img)
    if geninfo is None:
        return {}
    else:
        p = ast.literal_eval(geninfo)
        return p


def read_info_from_image(image: Image.Image) -> tuple[str | None, dict]:
    items = (image.info or {}).copy()

    geninfo = items.pop('parameters', None)

    if "exif" in items:
        exif_data = items["exif"]
        try:
            exif = piexif.load(exif_data)
        except OSError:
            # memory / exif was not valid so piexif tried to read from a file
            exif = None
        exif_comment = (exif or {}).get("Exif", {}).get(piexif.ExifIFD.UserComment, b'')
        try:
            exif_comment = piexif.helper.UserComment.load(exif_comment)
        except ValueError:
            exif_comment = exif_comment.decode('utf8', errors="ignore")

        if exif_comment:
            items['exif comment'] = exif_comment
            geninfo = exif_comment
    elif "comment" in items: # for gif
        geninfo = items["comment"].decode('utf8', errors="ignore")

    return geninfo, items

def str2num(string) -> Union[int, None]:
    # find a number at the end of a string, optionally enclosed in parentheses
    num_match = re.search(r'(?i)(-?0x[0-9a-f]+|-?\d+)\)?$', string)
    if num_match:
        num_str = num_match.group().strip("()")
        if num_str.startswith("0x") or num_str.startswith("-0x"):
            return int(num_str, 16)  # Convert hexadecimal string to integer
        else:
            return int(num_str)  # Convert decimal string to integer
    else:
        return None


def default_torch_device() -> str:
    if torch.backends.mps.is_available():
        default_device = "mps"
    elif torch.cuda.is_available():
        default_device = "cuda"
    else:
        default_device = "cpu"
    return default_device

