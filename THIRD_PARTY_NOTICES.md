# Third-Party Notices

This repository contains MIT-licensed project source code and declares external
Python packages in `requirements.txt`. Those packages are installed separately
and retain their own licenses.

## PyMaxflow

`OPR_RestoreOriginalPixels` method `B` uses PyMaxflow 1.3.2 for graph-cut
segmentation. PyMaxflow is not vendored in this repository. It is distributed
separately under the GNU General Public License version 3 (GPL-3.0), according
to its package metadata and tagged source distribution. Its C++ graph-cut core
credits Vladimir Kolmogorov and related authors in the upstream project.

- Project: https://github.com/pmneila/PyMaxflow
- Version: 1.3.2
- License: GNU General Public License version 3
- Tagged source: https://github.com/pmneila/PyMaxflow/tree/v1.3.2

All other dependencies likewise remain subject to their respective licenses.
