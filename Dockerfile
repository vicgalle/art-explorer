FROM frolvlad/alpine-miniconda3:python3.7

# Create the environment:

COPY environment.yml .
RUN conda env create -f environment.yml
ENV PATH /opt/conda/envs/demo/bin:$PATH
RUN /bin/sh -c "source activate demo"

ADD . .

RUN apk add sudo

RUN mkdir wikiart
RUN ls wikiart

CMD streamlit run streamlit_app.py --server.port 8003
