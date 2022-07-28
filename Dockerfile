FROM frolvlad/alpine-miniconda3:python3.7

# Create the environment:

COPY environment.yml .
RUN conda env create -f environment.yml
ENV PATH /opt/conda/envs/demo/bin:$PATH
RUN /bin/sh -c "source activate demo"
RUN apk add sudo git
RUN pip install git+https://github.com/openai/CLIP.git
ADD . .



RUN mkdir wikiart
RUN ls wikiart

CMD streamlit run streamlit_app.py --server.port 8050
