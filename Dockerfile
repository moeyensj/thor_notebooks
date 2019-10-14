FROM moeyensj/thor:latest
MAINTAINER Joachim Moeyens <moeyensj@gmail.com>

# Set shell to bash
SHELL ["/bin/bash", "-c"]

# Expose port for Jupyter
EXPOSE 1719

# Download THOR Notebooks
RUN cd projects \
	&& git clone https://github.com/moeyensj/thor_notebooks.git --depth=1

# Install additional requirements into conda enviroment
RUN cd projects/thor_notebooks \
	&& conda install -c defaults -c conda-forge --file additional_requirements.txt --y

# Set work directory and entry point 
WORKDIR /projects
ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--port=1719", "--allow-root", "--no-browser"]