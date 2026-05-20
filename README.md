[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/QDjEejvC)
# CS-552 - Open Project Milestone

Welcome to the milestone submission for the MNLP open project!

As detailed in the [open project description](https://docs.google.com/document/d/1NI4UKsasYuFLxOGGzsAweCbW0XOEtc59/edit#heading=h.8ww6fxjeedkw), the milestone submission is a checkpoint to ensure you are making meaningful progress. You should have a working pipeline and initial results by this point.

## Deliverables
Please refer to the [project description - Part II - Deliverables](https://docs.google.com/document/d/1NI4UKsasYuFLxOGGzsAweCbW0XOEtc59/edit#heading=h.oyoo4j9crv0k) for details.

## Rubric
Please refer to the [project description - Part II - Rubric](https://docs.google.com/document/d/1NI4UKsasYuFLxOGGzsAweCbW0XOEtc59/edit#heading=h.psuj40jgnavg) for details.

## Repo Structure
The repository is organized as follows:

### [`rcp_support/`](rcp_support)
We provide a step by step guide to help you set up and use the RCP framework effectively in the [rcp_support](rcp_support).
This includes instructions for submitting runai jobs and building your own docker images if needed.
Read [rcp_support documentation](rcp_support/README.md) carefully to ensure you can successfully run your experiments on the cluster.

### [`notebooks/`](notebooks)
As part of the milestone submission, you are required to submit a notebook per team member. 
Put these notebooks in the [notebooks](notebooks) directory.

Use this structure for the notebook submissions:

```text
notebooks/
  submit.sh # (one per team, to launch the notebook environment)
  <first_name>_<last_name>_<sciper>.ipynb
  <first_name>_<last_name>_<sciper>.ipynb
  <first_name>_<last_name>_<sciper>.ipynb
  <first_name>_<last_name>_<sciper>.ipynb
```

See [rcp_support/README.md](rcp_support/README.md) for submit script details.

### [`report/`](report)
Put your milestone report in the [report](report) directory. 

### [`code/`](code)
Put all the code files in the [code](code) directory. You are free to organize your code as you see fit in this directory.
