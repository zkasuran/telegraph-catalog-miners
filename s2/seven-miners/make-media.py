"""SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
Build owned image/video fixtures with non-neural code. They are rendered graphics,
not real photographs, deepfakes or an authenticity benchmark.
"""
from pathlib import Path
import struct
import shutil
import subprocess
import zlib

ROOT=Path(__file__).resolve().parent

if __name__ == '__main__':
    fixtures=ROOT / 'fixtures'
    fixtures.mkdir(exist_ok=True)
    parent=ROOT.parent
    shutil.copyfile(parent / 'owned-media-probe.png',fixtures / 'owned-rendered.png')
    license_text=(ROOT / 'LICENSE').read_text()
    png=(fixtures / 'owned-rendered.png').read_bytes()
    body=b'License\0'+license_text.encode('latin-1')
    extra=struct.pack('>I',len(body))+b'tEXt'+body+struct.pack('>I',zlib.crc32(b'tEXt'+body)&0xffffffff)
    (fixtures / 'owned-rendered.png').write_bytes(png[:-12]+extra+png[-12:])
    command=['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y','-f','lavfi','-i','color=c=red:s=320x180:r=12:d=1',
        '-f','lavfi','-i','color=c=blue:s=320x180:r=12:d=1',
        '-filter_complex','[0:v]drawbox=x=20:y=50:w=60:h=60:color=blue:t=fill[a];[1:v]drawbox=x=200:y=70:w=50:h=50:color=lime:t=fill[b];[a][b]concat=n=2:v=1:a=0[v]',
        '-map','[v]','-an','-c:v','libx264','-preset','veryfast',
        '-pix_fmt','yuv420p','-movflags','+faststart','-metadata','comment='+license_text,str(fixtures / 'owned-rendered.mp4')]
    subprocess.run(command,check=True,timeout=30)
    assert (fixtures / 'owned-rendered.mp4').stat().st_size<262144
    print('Owned rendered PNG and two-second MP4 created')
