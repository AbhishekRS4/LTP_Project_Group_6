import wandb
run = wandb.init()
artifact = run.use_artifact('language-technology-project/Touche23/model-g9ixfcas:v0', type='model')
artifact_dir = artifact.download()
