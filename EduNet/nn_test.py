from nn_visualizer import visualize, make_blobs_dataset, make_xor_dataset

visualize(
  dataset=make_blobs_dataset,  # any function returning (X, y)
  architecture=[2, 5, 5, 1],
  epochs=200,
  alpha=0.6,
  seed=None,
)