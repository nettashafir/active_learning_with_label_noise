#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""IO utilities (adapted from Detectron)"""

import logging
import os
import re
import sys
from urllib import request as urlrequest


logger = logging.getLogger(__name__)

_PYCLS_BASE_URL = "https://dl.fbaipublicfiles.com/pycls"


def cache_url(url_or_file, cache_dir, base_url=_PYCLS_BASE_URL):
    """Download the file specified by the URL to the cache_dir and return the path to
    the cached file. If the argument is not a URL, simply return it as is.
    """
    is_url = re.match(r"^(?:http)s?://", url_or_file, re.IGNORECASE) is not None
    if not is_url:
        return url_or_file
    url = url_or_file
    assert url.startswith(base_url), "url must start with: {}".format(base_url)
    cache_file_path = url.replace(base_url, cache_dir)
    if os.path.exists(cache_file_path):
        return cache_file_path
    cache_file_dir = os.path.dirname(cache_file_path)
    if not os.path.exists(cache_file_dir):
        os.makedirs(cache_file_dir)
    logger.info("Downloading remote file {} to {}".format(url, cache_file_path))
    download_url(url, cache_file_path)
    return cache_file_path


def _progress_bar(count, total):
    """Report download progress. Credit:
    https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/27871113
    """
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = "=" * filled_len + "-" * (bar_len - filled_len)
    sys.stdout.write(
        "  [{}] {}% of {:.1f}MB file  \r".format(bar, percents, total / 1024 / 1024)
    )
    sys.stdout.flush()
    if count >= total:
        sys.stdout.write("\n")


def download_url(url, dst_file_path, chunk_size=8192, progress_hook=_progress_bar):
    """Download url and write it to dst_file_path. Credit:
    https://stackoverflow.com/questions/2028517/python-urllib2-progress-hook
    """
    req = urlrequest.Request(url)
    response = urlrequest.urlopen(req)
    total_size = response.info().get("Content-Length").strip()
    total_size = int(total_size)
    bytes_so_far = 0
    with open(dst_file_path, "wb") as f:
        while 1:
            chunk = response.read(chunk_size)
            bytes_so_far += len(chunk)
            if not chunk:
                break
            if progress_hook:
                progress_hook(bytes_so_far, total_size)
            f.write(chunk)
    return bytes_so_far
