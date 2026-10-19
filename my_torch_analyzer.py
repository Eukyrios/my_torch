#!/usr/bin/env python3
# Entry point, copied to ./my_torch_analyzer by the Makefile.

import sys

from analyzer.cli import main

if __name__ == "__main__":
    sys.exit(main())
