import base64
import numpy as np
import struct
import zlib

def write_png(img):
    img_rgba = np.flipud(np.stack((img,)*4, axis=-1)) # flip y-axis
    img_rgba[:, :, -1] = 255 # set alpha channel (png uses byte-order)
    buf = bytearray(img_rgba)
    width = img_rgba.shape[1]
    height = img_rgba.shape[0]

    # reverse the vertical line order and add null bytes at the start
    width_byte_4 = width * 4
    raw_data = b''.join(
        b'\x00' + buf[span:span + width_byte_4]
        for span in range((height - 1) * width_byte_4, -1, - width_byte_4)
    )

    def png_pack(png_tag, data):
        chunk_head = png_tag + data
        return (struct.pack("!I", len(data)) +
                chunk_head +
                struct.pack("!I", 0xFFFFFFFF & zlib.crc32(chunk_head)))

    data = b''.join([
        b'\x89PNG\r\n\x1a\n',
        png_pack(b'IHDR', struct.pack("!2I5B", width, height, 8, 6, 0, 0, 0)),
        png_pack(b'IDAT', zlib.compress(raw_data, 9)),
        png_pack(b'IEND', b'')])
    return base64.b64encode(data)
