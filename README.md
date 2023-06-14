# Detecting the Underlying Human Values and Analysing their Persuasiveness in Online Debates using Large Language Models: A Sequence-to-Sequence Approach

## Dependencies
* All the dependencies are listed in [requirements.txt](requirements.txt)

## Task 1
### To classify human values using sequence-to-sequence approach
* The following dataset was used for this task - [Touché23-ValueEval](https://zenodo.org/record/7879430)

### The following scripts/notebooks can be run
* Run [task_1/augmentation.py](task_1/augmentation.py) to generate augmented data
* Run [task_1/prompt.py](task_1/prompt.py) to generate dataset files with prompts
* Run [task_1/train.py](task_1/train.py) to train the model
* Run [task_1/test.py](task_1/test.py) to test the model

## Task 2
### To find the underlying human values by online debaters
* The following dataset was used for this task - [Webis-CMV-20](https://zenodo.org/record/3778298#.ZHX069JBxH7)

### The following scripts/notebooks can be run
* Run [task_2/create_csv_data.py](task_2/create_csv_data.py) to generate csv data from json
* Run [task_2/post_process_csv_dataset.py](task_2/post_process_csv_dataset.py) to postprocess csv data file to remove empty/duplicate comments
* Run [task_2/prompt.py][task_2/prompt.py] to generate csv data file with prompts
* Run [task_2/create_dataset_for_top_k_authors.py](task_2/create_dataset_for_top_k_authors.py) to  create hugging face dataset files for top authors
* Run [task_2/test.py](task_2/test.py) to predict human values expressed by online debaters on their comments
* Run [task_2/results_task2/final_evaluation.ipynb](task_2/results_task2/final_evaluation.ipynb) to evaluate task 2
