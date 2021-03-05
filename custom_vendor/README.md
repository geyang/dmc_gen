# Installing `custom_vendor` modules

This codebase requires two custom versions of the `dm_control` and `dmc2gym` module. To install, run the following make file

```bash
make
```

which runs the following [./Makefile](./Makefile)

```makefile
SHELL := /bin/bash

default:
	cd dm_control && pip install -e . -q
	cd dmc2gym && pip install -e . -q

	echo "Now remember to add DMCGEN_DATA=`pwd`/data to your environment"
```

The two installation are both silent. Remember to add the `DMCGEN_DATA` environment variable for the background videos.

