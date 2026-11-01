"""TIFF decoding against independently encoded pixels, without a browser/GPU."""

import hashlib
import json
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, TiffImagePlugin

APP = Path(__file__).resolve().parents[2] / "web-app"
LOADER = f"""
import fs from 'node:fs';import vm from 'node:vm';import {{createHash}} from 'node:crypto';
import {{decodeTiff}} from {json.dumps((APP / "ui/tiff-codec.js").as_uri())};
const sandbox={{console:{{log(){{}}}}}};sandbox.self=sandbox;vm.createContext(sandbox);
for(const name of ['pako-inflate-1.0.11.min.js','UTIF-3.1.0.js'])vm.runInContext(fs.readFileSync({json.dumps(str(APP / "vendor"))}+'/'+name,'utf8'),sandbox);
const decode=path=>{{const bytes=fs.readFileSync(path);return decodeTiff(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength),sandbox.UTIF);}};
"""


@unittest.skipUnless(shutil.which("node"), "Node is required for TIFF decoder contracts")
class BrowserTiffTests(unittest.TestCase):
    def test_rgb16_palette_and_rgba_pixels(self):
        with tempfile.TemporaryDirectory() as temporary:
            gray = np.random.default_rng(21).integers(0, 256, size=(13, 17), dtype=np.uint8)
            rgb = np.stack([gray, np.flipud(gray), np.fliplr(gray)], axis=2)
            alpha = np.full((*gray.shape, 1), 255, np.uint8)
            expected = np.concatenate([rgb, alpha], axis=2)
            path = Path(temporary) / "rgb16.tif"
            tags = TiffImagePlugin.ImageFileDirectory_v2()
            for key, value in {
                256: 17,
                257: 13,
                258: (16, 16, 16),
                259: 1,
                262: 2,
                273: 0,
                277: 3,
                278: 13,
                279: rgb.size * 2,
                284: 1,
            }.items():
                tags[key] = value
            path.write_bytes(
                b"II*\0"
                + struct.pack("<I", 8)
                + tags.tobytes(8)
                + (rgb.astype("<u2") * 257).tobytes()
            )
            palette = np.stack([np.arange(256), 255 - np.arange(256), np.arange(256)], axis=1)
            indexed = Image.frombytes("P", (17, 13), gray.tobytes())
            indexed.putpalette(palette.astype(np.uint8).tobytes())
            palette_path = Path(temporary) / "palette.tif"
            indexed.save(palette_path, compression="tiff_lzw")
            rgba_path = Path(temporary) / "rgba.tif"
            translucent = expected.copy()
            translucent[..., 3] = gray
            Image.fromarray(translucent).save(rgba_path)
            expected_palette = np.concatenate([palette[gray], alpha], axis=2).astype(np.uint8)
            paths = [str(p) for p in [path, palette_path, rgba_path]]
            hashes = json.loads(
                subprocess.check_output(
                    ["node", "--input-type=module"],
                    input=LOADER
                    + f"const paths={json.dumps(paths)};"
                    + "console.log(JSON.stringify(paths.map(p=>createHash('sha256').update(decode(p).rgba).digest('hex'))));",
                    text=True,
                )
            )
            self.assertEqual(
                hashes,
                [
                    hashlib.sha256(pixels.tobytes()).hexdigest()
                    for pixels in [expected, expected_palette, translucent]
                ],
            )

    def test_integer_depths_compressions_byte_orders_and_orientation(self):
        with tempfile.TemporaryDirectory() as temporary:
            cases = []

            def add(image, expected, **options):
                path = Path(temporary) / f"image_{len(cases)}.tiff"
                image.save(path, **options)
                cases.append(
                    dict(
                        path=str(path),
                        width=expected.shape[1],
                        height=expected.shape[0],
                        sha256=hashlib.sha256(expected.tobytes()).hexdigest(),
                    )
                )

            def rgba(gray):
                return np.concatenate(
                    [np.repeat(gray[..., None], 3, axis=2), np.full((*gray.shape, 1), 255)],
                    axis=2,
                ).astype(np.uint8)

            gray = np.random.default_rng(394).integers(0, 256, size=(37, 53), dtype=np.uint8)
            gray16 = gray.astype(np.uint16) * 256 + np.arange(53, dtype=np.uint16)[None, :]
            for compression in ["raw", "tiff_lzw", "tiff_adobe_deflate", "packbits"]:
                add(Image.fromarray(gray), rgba(gray), compression=compression)
                add(Image.fromarray(gray16), rgba(gray), compression=compression)
                add(Image.fromarray(gray16.astype(">u2")), rgba(gray), compression=compression)
            for values in [gray, gray16, gray16.astype(">u2")]:
                tags = TiffImagePlugin.ImageFileDirectory_v2()
                tags[317] = 2
                add(Image.fromarray(values), rgba(gray), compression="tiff_lzw", tiffinfo=tags)
            tags = TiffImagePlugin.ImageFileDirectory_v2()
            tags[262] = 0
            add(Image.fromarray(gray), rgba(gray), tiffinfo=tags)
            add(Image.fromarray(gray16), rgba(255 - gray), tiffinfo=tags)
            binary = gray > 127
            add(Image.fromarray(binary), rgba(binary.astype(np.uint8) * 255))
            rgb = np.stack([gray, np.flipud(gray), np.fliplr(gray)], axis=2)
            expected_rgb = np.concatenate([rgb, np.full((*gray.shape, 1), 255)], axis=2).astype(
                np.uint8
            )
            for compression in ["raw", "tiff_lzw", "tiff_adobe_deflate", "packbits"]:
                add(Image.fromarray(rgb), expected_rgb, compression=compression)
            for orientation, expected in enumerate(
                [
                    gray,
                    np.fliplr(gray),
                    gray[::-1, ::-1],
                    np.flipud(gray),
                    gray.T,
                    np.rot90(gray, -1),
                    gray.T[::-1, ::-1],
                    np.rot90(gray),
                ],
                1,
            ):
                tags = TiffImagePlugin.ImageFileDirectory_v2()
                tags[274] = orientation
                add(Image.fromarray(gray), rgba(expected), tiffinfo=tags)
            result = json.loads(
                subprocess.check_output(
                    ["node", "--input-type=module"],
                    input=LOADER
                    + f"const cases={json.dumps(cases)};"
                    + """
                    console.log(JSON.stringify(cases.map(c=>{const r=decode(c.path);return {
                      width:r.width,height:r.height,sha256:createHash('sha256').update(r.rgba).digest('hex')};})));
                    """,
                    text=True,
                )
            )
            self.assertEqual(result, [{k: c[k] for k in result[0]} for c in cases])

    def test_unsupported_types_stacks_and_truncated_files_fail_explicitly(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = []
            for name, image, options in [
                ("float", Image.fromarray(np.zeros((8, 8), np.float32)), {}),
                (
                    "stack",
                    Image.new("L", (8, 8)),
                    dict(save_all=True, append_images=[Image.new("L", (8, 8))]),
                ),
            ]:
                path = Path(temporary) / f"{name}.tif"
                image.save(path, **options)
                paths.append(str(path))
            broken = Path(temporary) / "broken.tif"
            broken.write_bytes(Path(paths[-1]).read_bytes()[:9])
            paths.append(str(broken))
            result = json.loads(
                subprocess.check_output(
                    ["node", "--input-type=module"],
                    input=LOADER
                    + f"const paths={json.dumps(paths)};"
                    + "console.log(JSON.stringify(paths.map(p=>{try{decode(p);return null;}catch(e){return e.message;}})));",
                    text=True,
                )
            )
            self.assertIn("floating-point", result[0])
            self.assertIn("Multipage", result[1])
            self.assertIn("directory", result[2])
