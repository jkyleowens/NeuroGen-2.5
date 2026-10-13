"""Embed SPIR-V so installed libraries have no runtime shader path dependency."""
import struct
import sys
from pathlib import Path
raw = Path(sys.argv[1]).read_bytes()
words = struct.unpack('<' + 'I' * (len(raw) // 4), raw)
Path(sys.argv[2]).write_text('#pragma once\n#include <cstdint>\ninline constexpr uint32_t lif_spv[] = {' + ','.join(map(str, words)) + '};\n')
