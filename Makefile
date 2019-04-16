CMD   = python
FLAGS = install
DOC   = setup.py

all : setup

setup : $(DOC)
	$(CMD) $(DOC) install

script :
	$(CMD) -m image_structure.scripts.driver image_structure/scripts/inputs.dat

clean :
	rm -rf build/ dist/ *egg-info

