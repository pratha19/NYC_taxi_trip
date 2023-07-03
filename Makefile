SHELL=/bin/zsh

help: # Show help for each of the Makefile recipes.
	@grep -E '^[a-zA-Z0-9 -_]+:.*#'  Makefile | sort | while read -r l; do printf "\033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m:$$(echo $$l | cut -f 2- -d'#')\n"; done

hello: # Print hello world to test makefile is working
	@echo "Hello, World"

stream_maps: # Run Streamlit App for choosing between points or lines charts
	@echo "Running Streamlit App, with options for points or lines bokeh plots"
	streamlit run streamlit_maps.py 