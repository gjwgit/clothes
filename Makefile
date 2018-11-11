########################################################################
#
# Makefile for clothes recommender pre-built ML model
#
########################################################################

# List the files to be included in the .mlm package.

MODEL_FILES = 		\
	configure.sh	\
	demo.py		\
	print.py	\
	display.py	\
	score.py	\
	README.txt	\
	DESCRIPTION.yaml\
	PARAMETERS.py	\
	helpers_cntk.py \
	helpers.py	\
	data		\
	proc

# Include the standard Makefile template.

include ../git.mk
include ../clean.mk
include ../pandoc.mk
include ../mlhub.mk

clean::
	rm -rf README.txt output

realclean:: clean
	rm -f $(MODEL)_*.mlm
