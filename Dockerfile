FROM rpy2/base-ubuntu

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
RUN Rscript -e 'install.packages("multiway", repos="https://cloud.r-project.org")'
    
EXPOSE 8080:8080
EXPOSE 8888:8888
COPY . .

