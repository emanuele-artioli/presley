# PRESLEY package init

# Import sqlite3 FIRST, before anything can pull in torch. This is not
# stylistic -- on this host it is load-bearing.
#
# conda's `libicui18n.so.78`, which the stdlib `_sqlite3` extension depends on,
# needs `CXXABI_1.3.15`. Importing torch first pins the SYSTEM
# /usr/lib/x86_64-linux-gnu/libstdc++.so.6, which does not provide that symbol,
# so a later `import sqlite3` dies with:
#
#     ImportError: .../libstdc++.so.6: version `CXXABI_1.3.15' not found
#                  (required by .../libicui18n.so.78)
#
# `import sqlite3, torch` works; `import torch, sqlite3` does not. Once sqlite3
# is in sys.modules the ordering stops mattering, so doing it here -- at the top
# of the package every entry point imports before it reaches torch -- makes the
# whole codebase immune.
#
# This bit only after the results DB landed: modules that import torch and
# *then* reach `presley.db` (directly, or via a deferred in-function import) hit
# it at runtime rather than at import time, which is the worst place to find it.
# See tests/test_import_order.py.
import sqlite3  # noqa: F401
