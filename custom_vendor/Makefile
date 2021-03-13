SHELL := /bin/bash

default:
	cd dm_control && pip install -e . -q
	cd dmc2gym && pip install -e . -q

	echo "Now remember to add DMCGEN_DATA=`pwd`/data to your environment"
