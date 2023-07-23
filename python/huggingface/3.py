from datasets import load_metric
metric = load_metric('glue', 'mrpc')
print(metric.inputs_description)