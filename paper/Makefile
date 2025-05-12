#mcandrew

PYTHON ?= python3 -W ignore
STATES ?= 'all'

VENV_DIR := .wmm
VENV_PYTHON := $(VENV_DIR)/bin/python -W ignore

R ?= Rscript

run: build_env format_data build_num_infects build_time_between_dataset build_transmission_network_object\
build_timeseries_data viz_transmission_over_time viz_dist_epi_params viz_compartmental_model

build_env:
	@echo "build forecast environment"
	@$(PYTHON) -m venv ./$(VENV_DIR)
	@$(VENV_PYTHON) -m pip install -r requirements.txt
	@echo "Forecast environment installed"

format_data:
	@echo "Format contact data"
	@$(VENV_PYTHON) ./analysis_data/format_contat_data_for_analysis.py

build_num_infects:
	@echo "Build number of infections per infector"
	@$(VENV_PYTHON) ./analysis_data/build_num_infecitons.py

build_time_between_dataset:
	@echo "Build time between infections dataset"
	@$(VENV_PYTHON) ./analysis_data/build_time_between_infections.py

build_transmission_network_object:
	@echo "Build Transmission network in networkx"
	@$(VENV_PYTHON) ./analysis_data/build_transmission_network.py

build_timeseries_data:
	@echo "Build Time series data"
	@$(VENV_PYTHON) ./analysis_data/build_time_series_data.py

viz_transmission_over_time:
	@echo "Viz transmission network over time"
	@$(VENV_PYTHON) ./viz/network_over_time/viz.py

viz_dist_epi_params:
	@echo "Viz EPi Params"
	@$(VENV_PYTHON) ./viz/epi_params_over_time/viz.py

viz_compartmental_model:
	@echo "Viz Compartment"
	@$(VENV_PYTHON) ./viz/present_compartmental_wont_capture/data_for_viz.py
	@$(VENV_PYTHON) ./viz/present_compartmental_wont_capture/viz.py





