# CrewATLAS - MollahLab Internal Repo

## Requirements

### Docker:
Full environment and requisite Python packages available on docker, pull from docker with
```
docker pull lucharles46/mollahlab:NLP
```

### Data
The data is processed via TDMs from Nature and Cell. API keys are in the notebooks -- NOTE: for external use remove API keys. Do not share API keys under any circumstance.

### NER Models
We are using BioBert and PubMedBert from huggingface. Dependencies are included in the docker image and calls are made in above notebook



